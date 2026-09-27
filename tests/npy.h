// train.cpp - 测试专用：最小 .npy 读取器（仅支持 C 序、小端 f32/i32）
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace trc_test {

struct NpyArray {
    std::vector<int64_t> shape;   // .npy 中的形状（C 序）
    std::vector<uint8_t> raw;     // 原始数据（小端）
    std::string          descr;   // "<f4" / "<i4"
};

inline bool npy_load(const std::string& path, NpyArray& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char magic[6] = {0};
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        return false;
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    f.read((char*) &major, 1);
    f.read((char*) &minor, 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        f.read((char*) &len16, 2);
        header_len = len16;
    } else {
        f.read((char*) &header_len, 4);
    }

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (!f) {
        return false;
    }

    // descr
    const char* d = std::strstr(header.c_str(), "'descr'");
    if (d == nullptr) {
        return false;
    }
    // 只支持 C 序（防止 np.save 把非连续视图写成 fortran_order=True）
    if (std::strstr(header.c_str(), "'fortran_order': True") != nullptr) {
        return false;
    }
    const char* colon = std::strchr(d, ':');
    const char* v1 = colon ? std::strchr(colon, '\'') : nullptr;
    const char* v2 = v1 ? std::strchr(v1 + 1, '\'') : nullptr;
    if (v1 == nullptr || v2 == nullptr) {
        return false;
    }
    out.descr.assign(v1 + 1, v2);

    // shape
    const char* s = std::strstr(header.c_str(), "'shape'");
    if (s == nullptr) {
        return false;
    }
    const char* lp = std::strchr(s, '(');
    const char* rp = lp ? std::strchr(lp, ')') : nullptr;
    if (lp == nullptr || rp == nullptr) {
        return false;
    }
    out.shape.clear();
    const char* p = lp + 1;
    while (p < rp) {
        while (p < rp && (*p == ' ' || *p == ',')) {
            ++p;
        }
        if (p >= rp) {
            break;
        }
        char* endp = nullptr;
        const long long v = std::strtoll(p, &endp, 10);
        if (endp == p) {
            break;
        }
        out.shape.push_back((int64_t) v);
        p = endp;
    }

    // data
    const std::streamoff data_off = 6 + 2 + (major == 1 ? 2 : 4) + (std::streamoff) header_len;
    f.seekg(0, std::ios::end);
    const std::streamoff total = f.tellg();
    if (total < data_off) {
        return false;
    }
    out.raw.resize((size_t) (total - data_off));
    f.seekg(data_off);
    f.read((char*) out.raw.data(), (std::streamsize) out.raw.size());
    return (bool) f;
}

} // namespace trc_test
