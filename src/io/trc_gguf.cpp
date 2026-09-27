// train.cpp - GGUF 读写实现（与 ggml v0.23.0 对齐，M1.6a）
//
// 参考 ggml v0.23.0 的 src/gguf.cpp、ggml-quants.c、ggml-common.h
// 生成端语义逐条对齐：头布局、KV 编码、tensor info、数据段对齐、类型 id、量化块布局。
#define _CRT_SECURE_NO_WARNINGS

#include "traincpp/trc_gguf.h"

#include "core/trc_float.h"
#include "core/trc_impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace traincpp {

namespace {

constexpr uint32_t GGUF_VERSION_WRITE     = 3;
constexpr uint32_t GGUF_VERSION_MIN       = 2;
constexpr uint32_t GGUF_VERSION_MAX       = 3;
constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;

// GGUF 元数据类型字节数（BOOL 落盘 1 字节；STRING/ARRAY 无固定大小 → 0）
int64_t gguf_type_size(GgufType t) {
    switch (t) {
        case GgufType::U8: case GgufType::I8: case GgufType::BOOL: return 1;
        case GgufType::U16: case GgufType::I16: return 2;
        case GgufType::U32: case GgufType::I32: case GgufType::F32: return 4;
        case GgufType::U64: case GgufType::I64: case GgufType::F64: return 8;
        default: return 0;
    }
}

bool host_is_little_endian() {
    const uint16_t one = 1;
    uint8_t b = 0;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

int file_seek64(std::FILE* f, uint64_t off) {
#ifdef _MSC_VER
    return _fseeki64(f, (int64_t) off, SEEK_SET);
#else
    return fseeko(f, (off_t) off, SEEK_SET);
#endif
}

// ---------------- 读取辅助 ----------------

struct ReadFile {
    std::FILE* f    = nullptr;
    uint64_t   size = 0;
    uint64_t   pos  = 0;
    bool       fail = false;
};

bool rf_seek(ReadFile& r, uint64_t off) {
    if (r.fail || off > r.size) { r.fail = true; return false; }
    if (file_seek64(r.f, off) != 0) { r.fail = true; return false; }
    r.pos = off;
    return true;
}

bool rf_read(ReadFile& r, void* dst, size_t n) {
    if (r.fail) return false;
    if (n == 0) return true;
    if ((uint64_t) n > r.size - r.pos) { r.fail = true; return false; }
    if (std::fread(dst, 1, n, r.f) != n) { r.fail = true; return false; }
    r.pos += (uint64_t) n;
    return true;
}

template <typename T>
bool rf_read_pod(ReadFile& r, T& v) {
    return rf_read(r, &v, sizeof(v));
}

// 字符串：u64 字节长度 + UTF-8（无 NUL），与 ggml 一致
bool rf_read_str(ReadFile& r, std::string& s) {
    uint64_t len = 0;
    if (!rf_read_pod(r, len)) return false;
    if (len > r.size - r.pos) { r.fail = true; return false; }
    s.resize((size_t) len);
    return rf_read(r, s.data(), (size_t) len);
}

// ---------------- 内存中的 KV ----------------

struct GgufKv {
    std::string key;
    GgufType    type = GgufType::U8;

    uint64_t    u64 = 0;    // 全部整数类型的位模式（含 i8..i64 补码）
    double      f64 = 0.0;  // F32/F64
    bool        b   = false;
    std::string str;

    GgufType    arr_type = GgufType::U8;
    int64_t     arr_n    = 0;
    std::vector<int32_t>     arr_i32;
    std::vector<int64_t>     arr_i64;
    std::vector<float>       arr_f32;
    std::vector<std::string> arr_str;
    std::vector<uint8_t>     arr_raw;  // 其他数组类型（1 字节/BOOL，其余按原字节）
};

struct GgufRTensor {
    std::string name;
    Type        type    = TYPE_F32;
    int64_t     ne[MAX_DIMS] = {1, 1, 1, 1};
    size_t      offset  = 0;
    size_t      nbytes  = 0;
};

// ---------------- 写出辅助 ----------------

struct WriteFile {
    std::FILE* f   = nullptr;
    uint64_t   pos = 0;
    bool       ok  = true;

    bool put(const void* p, size_t n) {
        if (!ok) return false;
        if (n != 0 && std::fwrite(p, 1, n, f) != n) { ok = false; return false; }
        pos += (uint64_t) n;
        return true;
    }
    bool put_u8(uint8_t v)   { return put(&v, 1); }
    bool put_u32(uint32_t v) { return put(&v, 4); }
    bool put_i32(int32_t v)  { return put(&v, 4); }
    bool put_u64(uint64_t v) { return put(&v, 8); }
    bool put_i64(int64_t v)  { return put(&v, 8); }
    bool put_str(const std::string& s) {
        return put_u64((uint64_t) s.size()) && put(s.data(), s.size());
    }
    // 以 0 填充到 target 字节位置
    bool pad_to(uint64_t target) {
        static const uint8_t zeros[64] = {0};
        while (ok && pos < target) {
            const uint64_t chunk = std::min<uint64_t>(target - pos, (uint64_t) sizeof(zeros));
            put(zeros, (size_t) chunk);
        }
        return ok;
    }
    // 以 0 填充到 alignment 的整数倍
    bool pad_to_alignment(uint64_t alignment) {
        return pad_to((uint64_t) align_up((size_t) pos, (size_t) alignment));
    }
};

// ---------------- KV 序列化 ----------------

size_t kv_serialized_size(const GgufKv& kv) {
    size_t n = 8 + kv.key.size() + 4;
    switch (kv.type) {
        case GgufType::U8: case GgufType::I8: case GgufType::BOOL: n += 1; break;
        case GgufType::U16: case GgufType::I16: n += 2; break;
        case GgufType::U32: case GgufType::I32: case GgufType::F32: n += 4; break;
        case GgufType::U64: case GgufType::I64: case GgufType::F64: n += 8; break;
        case GgufType::STRING: n += 8 + kv.str.size(); break;
        case GgufType::ARRAY: {
            n += 4 + 8;
            if (kv.arr_type == GgufType::STRING) {
                for (const std::string& s : kv.arr_str) {
                    n += 8 + s.size();
                }
            } else {
                n += (size_t) kv.arr_n * (size_t) gguf_type_size(kv.arr_type);
            }
        } break;
        default: break;
    }
    return n;
}

bool write_kv(WriteFile& wf, const GgufKv& kv) {
    if (!wf.put_str(kv.key)) return false;
    if (!wf.put_u32((uint32_t) kv.type)) return false;
    switch (kv.type) {
        case GgufType::U8:
        case GgufType::I8:   return wf.put_u8((uint8_t) kv.u64);
        case GgufType::U16:  { const uint16_t v = (uint16_t) kv.u64; return wf.put(&v, 2); }
        case GgufType::I16:  { const int16_t  v = (int16_t)  kv.u64; return wf.put(&v, 2); }
        case GgufType::U32:  return wf.put_u32((uint32_t) kv.u64);
        case GgufType::I32:  return wf.put_i32((int32_t) kv.u64);
        case GgufType::F32:  { const float v = (float) kv.f64; return wf.put(&v, 4); }
        case GgufType::BOOL: return wf.put_u8(kv.b ? 1 : 0);
        case GgufType::U64:  return wf.put_u64(kv.u64);
        case GgufType::I64:  return wf.put_i64((int64_t) kv.u64);
        case GgufType::F64:  return wf.put(&kv.f64, 8);
        case GgufType::STRING: return wf.put_str(kv.str);
        case GgufType::ARRAY: {
            if (!wf.put_u32((uint32_t) kv.arr_type)) return false;
            if (!wf.put_u64((uint64_t) kv.arr_n)) return false;
            if (kv.arr_type == GgufType::STRING) {
                for (const std::string& s : kv.arr_str) {
                    if (!wf.put_str(s)) return false;
                }
                return true;
            }
            if (kv.arr_type == GgufType::I32) {
                return kv.arr_i32.empty() || wf.put(kv.arr_i32.data(), kv.arr_i32.size() * 4);
            }
            if (kv.arr_type == GgufType::I64) {
                return kv.arr_i64.empty() || wf.put(kv.arr_i64.data(), kv.arr_i64.size() * 8);
            }
            if (kv.arr_type == GgufType::F32) {
                return kv.arr_f32.empty() || wf.put(kv.arr_f32.data(), kv.arr_f32.size() * 4);
            }
            return false;  // 写端暂不支持其他数组类型
        }
        default: return false;
    }
}

// 解析一个 KV（含数组）。失败返回 false（r.fail 也会置位）。
bool parse_kv(ReadFile& r, GgufKv& kv) {
    if (!rf_read_str(r, kv.key)) return false;

    int32_t type_raw = 0;
    if (!rf_read_pod(r, type_raw)) return false;
    if (type_raw < 0 || type_raw > (int32_t) GgufType::F64) return false;
    kv.type = (GgufType) type_raw;

    switch (kv.type) {
        case GgufType::U8: case GgufType::I8: case GgufType::BOOL: {
            uint8_t v = 0;
            if (!rf_read_pod(r, v)) return false;
            kv.u64 = v;
            kv.b   = (v != 0);
        } break;
        case GgufType::U16: case GgufType::I16: {
            uint16_t v = 0;
            if (!rf_read_pod(r, v)) return false;
            kv.u64 = v;
        } break;
        case GgufType::U32: case GgufType::I32: {
            uint32_t v = 0;
            if (!rf_read_pod(r, v)) return false;
            kv.u64 = v;
        } break;
        case GgufType::U64: case GgufType::I64: {
            uint64_t v = 0;
            if (!rf_read_pod(r, v)) return false;
            kv.u64 = v;
        } break;
        case GgufType::F32: {
            float v = 0.0f;
            if (!rf_read_pod(r, v)) return false;
            kv.f64 = (double) v;
        } break;
        case GgufType::F64: {
            double v = 0.0;
            if (!rf_read_pod(r, v)) return false;
            kv.f64 = v;
        } break;
        case GgufType::STRING: {
            if (!rf_read_str(r, kv.str)) return false;
        } break;
        case GgufType::ARRAY: {
            int32_t et_raw = 0;
            if (!rf_read_pod(r, et_raw)) return false;
            if (et_raw < 0 || et_raw > (int32_t) GgufType::F64 ||
                et_raw == (int32_t) GgufType::ARRAY) {
                return false;  // 不支持嵌套数组
            }
            kv.arr_type = (GgufType) et_raw;

            uint64_t n = 0;
            if (!rf_read_pod(r, n)) return false;
            if (n > (uint64_t) INT64_MAX) return false;
            kv.arr_n = (int64_t) n;

            if (kv.arr_type == GgufType::STRING) {
                kv.arr_str.reserve((size_t) std::min<uint64_t>(n, 1024));
                for (uint64_t i = 0; i < n; ++i) {
                    std::string s;
                    if (!rf_read_str(r, s)) return false;
                    kv.arr_str.push_back(std::move(s));
                }
            } else if (kv.arr_type == GgufType::BOOL) {
                if (n > r.size - r.pos) return false;
                kv.arr_raw.resize((size_t) n);
                if (!rf_read(r, kv.arr_raw.data(), (size_t) n)) return false;
            } else {
                const int64_t es = gguf_type_size(kv.arr_type);
                if (es <= 0) return false;
                if (n > 0 && (uint64_t) es > (r.size - r.pos) / n) return false;
                const size_t bytes = (size_t) (n * (uint64_t) es);
                std::vector<uint8_t> raw(bytes);
                if (!rf_read(r, raw.data(), bytes)) return false;
                if (kv.arr_type == GgufType::I32) {
                    kv.arr_i32.resize((size_t) n);
                    if (bytes > 0) std::memcpy(kv.arr_i32.data(), raw.data(), bytes);
                } else if (kv.arr_type == GgufType::I64) {
                    kv.arr_i64.resize((size_t) n);
                    if (bytes > 0) std::memcpy(kv.arr_i64.data(), raw.data(), bytes);
                } else if (kv.arr_type == GgufType::F32) {
                    kv.arr_f32.resize((size_t) n);
                    if (bytes > 0) std::memcpy(kv.arr_f32.data(), raw.data(), bytes);
                } else {
                    kv.arr_raw = std::move(raw);
                }
            }
        } break;
        default: return false;
    }
    return true;
}

// ---------------- 张量数据反量化（对照 ggml-quants.c dequantize_row_*） ----------------

void dequantize_q4_0(const uint8_t* src, int64_t n_blocks, float* out) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        uint16_t dh = 0;
        std::memcpy(&dh, src, 2);
        src += 2;
        const float    d  = fp16_to_fp32(dh);
        const uint8_t* qs = src;
        src += 16;
        float* y = out + b * 32;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >> 4) - 8;
            y[j]      = (float) x0 * d;
            y[j + 16] = (float) x1 * d;
        }
    }
}

void dequantize_q4_1(const uint8_t* src, int64_t n_blocks, float* out) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        uint16_t dh = 0, mh = 0;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        src += 4;
        const float    d  = fp16_to_fp32(dh);
        const float    m  = fp16_to_fp32(mh);
        const uint8_t* qs = src;
        src += 16;
        float* y = out + b * 32;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (qs[j] & 0x0F);
            const int x1 = (qs[j] >> 4);
            y[j]      = (float) x0 * d + m;
            y[j + 16] = (float) x1 * d + m;
        }
    }
}

void dequantize_q5_0(const uint8_t* src, int64_t n_blocks, float* out) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        uint16_t dh = 0;
        std::memcpy(&dh, src, 2);
        uint32_t qh = 0;
        std::memcpy(&qh, src + 2, 4);
        src += 6;
        const float    d  = fp16_to_fp32(dh);
        const uint8_t* qs = src;
        src += 16;
        float* y = out + b * 32;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = (uint8_t) (((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh_1 = (uint8_t) ((qh >> (j + 12)) & 0x10);
            const int     x0   = ((qs[j] & 0x0F) | xh_0) - 16;
            const int     x1   = ((qs[j] >> 4) | xh_1) - 16;
            y[j]      = (float) x0 * d;
            y[j + 16] = (float) x1 * d;
        }
    }
}

void dequantize_q5_1(const uint8_t* src, int64_t n_blocks, float* out) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        uint16_t dh = 0, mh = 0;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        uint32_t qh = 0;
        std::memcpy(&qh, src + 4, 4);
        src += 8;
        const float    d  = fp16_to_fp32(dh);
        const float    m  = fp16_to_fp32(mh);
        const uint8_t* qs = src;
        src += 16;
        float* y = out + b * 32;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = (uint8_t) (((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh_1 = (uint8_t) ((qh >> (j + 12)) & 0x10);
            const int     x0   = (qs[j] & 0x0F) | xh_0;
            const int     x1   = (qs[j] >> 4) | xh_1;
            y[j]      = (float) x0 * d + m;
            y[j + 16] = (float) x1 * d + m;
        }
    }
}

void dequantize_q8_0(const uint8_t* src, int64_t n_blocks, float* out) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        uint16_t dh = 0;
        std::memcpy(&dh, src, 2);
        src += 2;
        const float   d  = fp16_to_fp32(dh);
        const int8_t* qs = (const int8_t*) src;
        src += 32;
        float* y = out + b * 32;
        for (int j = 0; j < 32; ++j) {
            y[j] = (float) qs[j] * d;
        }
    }
}

} // namespace

// ---------------------------------------------------------------- 读句柄

struct GgufFile {
    std::vector<GgufKv>         kvs;
    std::vector<std::string>    tensor_names;  // infos[i].name 指向此处（reserve 后地址稳定）
    std::vector<GgufTensorInfo> infos;
    std::vector<uint8_t>        data;          // load_data=true 时载入的数据段（可能不含尾部填充）
    uint32_t version     = 0;
    uint32_t alignment   = GGUF_DEFAULT_ALIGNMENT;
    uint64_t data_offset = 0;
    bool     has_data    = false;
    std::unordered_map<std::string, int64_t> key_map;
    std::unordered_map<std::string, int64_t> tensor_map;
};

// ---------------------------------------------------------------- 写句柄

struct GgufWriter {
    struct WTensor {
        std::string   name;
        const Tensor* tensor   = nullptr;
        Type          out_type = TYPE_F32;
        int64_t       ne[MAX_DIMS] = {1, 1, 1, 1};
        int           n_dims   = 1;
        size_t        nbytes   = 0;
    };

    std::vector<GgufKv>  kvs;
    std::vector<WTensor> tensors;
};

namespace {

// ---------------- 解析 ----------------

int64_t tensor_nelements_of(const GgufTensorInfo& info) {
    return info.ne[0] * info.ne[1] * info.ne[2] * info.ne[3];
}

int64_t tensor_nelements_of(const GgufRTensor& t) {
    return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
}

// 解析张量元信息；遵循 ggml 的校验（类型范围/ne 整除块大小/元素数溢出）
bool parse_tensor_info(ReadFile& r, GgufRTensor& t) {
    if (!rf_read_str(r, t.name)) return false;
    if (t.name.empty() || t.name.size() >= (size_t) MAX_NAME) return false;

    uint32_t n_dims = 0;
    if (!rf_read_pod(r, n_dims)) return false;
    if (n_dims < 1 || n_dims > (uint32_t) MAX_DIMS) return false;

    uint64_t ne64[MAX_DIMS] = {1, 1, 1, 1};
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (!rf_read_pod(r, ne64[i])) return false;
        if (ne64[i] == 0 || ne64[i] > (uint64_t) INT64_MAX) return false;
    }

    int32_t type_raw = 0;
    if (!rf_read_pod(r, type_raw)) return false;
    if (type_raw < 0 || type_raw >= (int32_t) TYPE_COUNT) return false;
    t.type = (Type) type_raw;

    const int64_t blck = type_blck_size(t.type);
    const size_t  es   = type_size(t.type);
    if (blck <= 0 || es == 0) return false;  // 本库尚未实现的类型（如 K-quant）

    uint64_t offset = 0;
    if (!rf_read_pod(r, offset)) return false;
    if (offset > (uint64_t) SIZE_MAX) return false;
    t.offset = (size_t) offset;

    for (int i = 0; i < MAX_DIMS; ++i) {
        t.ne[i] = (i < (int) n_dims) ? (int64_t) ne64[i] : 1;
    }
    if (t.ne[0] % blck != 0) return false;

    // 元素数溢出检查（ggml 用 ggml_nelements 校验）
    int64_t n_elems = 1;
    for (int i = 0; i < MAX_DIMS; ++i) {
        if (t.ne[i] > INT64_MAX / n_elems) return false;
        n_elems *= t.ne[i];
    }

    // 字节数与核心 tensor_nbytes 公式一致
    size_t nb[MAX_DIMS] = {0, 0, 0, 0};
    nb[0] = es;
    for (int i = 1; i < MAX_DIMS; ++i) {
        nb[i] = nb[i - 1] * (size_t) (t.ne[i - 1] / blck);
    }
    size_t nbytes = (size_t) (t.ne[0] / blck) * es;
    for (int i = 1; i < MAX_DIMS; ++i) {
        if (t.ne[i] > 1) {
            nbytes += (size_t) (t.ne[i] - 1) * nb[i];
        }
    }
    t.nbytes = nbytes;
    return true;
}

GgufFile* gguf_parse(ReadFile& r, bool load_data) {
    char magic[4] = {0};
    if (!rf_read(r, magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        TRC_LOG_ERROR("GGUF: 魔数不正确（不是 GGUF 文件）");
        return nullptr;
    }

    uint32_t version = 0;
    if (!rf_read_pod(r, version)) {
        TRC_LOG_ERROR("GGUF: 文件头不完整");
        return nullptr;
    }
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        TRC_LOG_ERROR("GGUF: 不支持的版本 %u（仅支持 v2/v3）", version);
        return nullptr;
    }

    uint64_t n_tensors = 0;
    uint64_t n_kv      = 0;
    if (!rf_read_pod(r, n_tensors) || !rf_read_pod(r, n_kv)) {
        TRC_LOG_ERROR("GGUF: 文件头不完整");
        return nullptr;
    }

    // 计数合理性：避免损坏文件导致巨量分配（KV >= 13 字节、tensor info >= 32 字节）
    const uint64_t remain0 = r.size - r.pos;
    if (n_kv > remain0 / 13 || n_tensors > remain0 / 32) {
        TRC_LOG_ERROR("GGUF: 元数据/张量计数超出文件大小");
        return nullptr;
    }

    GgufFile* gf = new GgufFile();
    auto fail = [&](const char* why) -> GgufFile* {
        TRC_LOG_ERROR("GGUF: 解析失败（%s）", why);
        delete gf;
        return nullptr;
    };

    gf->version = version;

    gf->kvs.reserve((size_t) std::min<uint64_t>(n_kv, 4096));
    for (uint64_t i = 0; i < n_kv; ++i) {
        GgufKv kv;
        if (!parse_kv(r, kv)) {
            return fail("元数据 KV 损坏");
        }
        if (kv.key.empty()) {
            return fail("元数据 key 为空");
        }
        auto ins = gf->key_map.emplace(kv.key, (int64_t) i);
        if (!ins.second) {
            return fail("元数据 key 重复");
        }
        gf->kvs.push_back(std::move(kv));
    }

    // 对齐（general.alignment，与 ggml 相同：u32、2 的幂）
    if (auto it = gf->key_map.find("general.alignment"); it != gf->key_map.end()) {
        const GgufKv& kv = gf->kvs[(size_t) it->second];
        if (kv.type != GgufType::U32) {
            return fail("general.alignment 类型必须为 u32");
        }
        const uint32_t a = (uint32_t) kv.u64;
        if (a == 0 || (a & (a - 1)) != 0 || a > (UINT32_C(1) << 30)) {
            return fail("general.alignment 非法");
        }
        gf->alignment = a;
    }

    std::vector<GgufRTensor> tinfos;
    tinfos.reserve((size_t) std::min<uint64_t>(n_tensors, 65536));
    uint64_t expected_off = 0;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufRTensor t;
        if (!parse_tensor_info(r, t)) {
            return fail("张量元信息损坏或不支持的类型");
        }
        // 与 ggml 一致：offset 必须等于前序张量按对齐累计的位置（连续无洞）
        if ((uint64_t) t.offset != expected_off) {
            return fail("张量 offset 不连续");
        }
        expected_off += (uint64_t) align_up(t.nbytes, (size_t) gf->alignment);
        tinfos.push_back(std::move(t));
    }

    // 数据段起始 = 对齐后的当前位置；允许无张量文件缺少尾部填充
    const uint64_t pos_after_ti   = r.pos;
    uint64_t       data_off       = (uint64_t) align_up((size_t) pos_after_ti, (size_t) gf->alignment);
    uint64_t       last_end       = 0;
    if (!tinfos.empty()) {
        last_end = (uint64_t) tinfos.back().offset + (uint64_t) tinfos.back().nbytes;
        if (data_off > r.size || last_end > r.size - data_off) {
            return fail("张量数据段超出文件大小");
        }
    } else if (data_off > r.size) {
        data_off = r.size;
    }
    gf->data_offset = data_off;

    // 登记张量（名字与查找索引）；names 先 reserve 保证 c_str 指针稳定
    gf->tensor_names.reserve(tinfos.size());
    gf->infos.reserve(tinfos.size());
    for (size_t i = 0; i < tinfos.size(); ++i) {
        auto ins = gf->tensor_map.emplace(tinfos[i].name, (int64_t) i);
        if (!ins.second) {
            return fail("张量名重复");
        }
        gf->tensor_names.push_back(tinfos[i].name);
    }
    for (size_t i = 0; i < tinfos.size(); ++i) {
        GgufTensorInfo info;
        info.name = gf->tensor_names[i].c_str();
        info.type = tinfos[i].type;
        for (int d = 0; d < MAX_DIMS; ++d) {
            info.ne[d] = tinfos[i].ne[d];
        }
        info.offset = tinfos[i].offset;
        info.nbytes = tinfos[i].nbytes;
        gf->infos.push_back(info);
    }

    if (load_data) {
        const uint64_t span = std::min<uint64_t>(expected_off, r.size - data_off);
        gf->data.resize((size_t) span);
        if (span > 0) {
            if (!rf_seek(r, data_off) || !rf_read(r, gf->data.data(), (size_t) span)) {
                gf->data.clear();
                return fail("张量数据读取失败");
            }
        }
        gf->has_data = true;
    }
    return gf;
}

// 查找 KV（类型必须匹配；缺失/类型不符时打印错误并返回 nullptr）
const GgufKv* find_kv(const GgufFile* f, const char* key, GgufType want) {
    const auto it = f->key_map.find(key);
    if (it == f->key_map.end()) {
        TRC_LOG_ERROR("GGUF: 缺少元数据键 %s", key);
        return nullptr;
    }
    const GgufKv& kv = f->kvs[(size_t) it->second];
    if (kv.type != want) {
        TRC_LOG_ERROR("GGUF: 键 %s 类型不符（实际 %d，期望 %d）", key, (int) kv.type, (int) want);
        return nullptr;
    }
    return &kv;
}

// ---------------- 写出 ----------------

size_t tensor_info_serialized_size(const GgufWriter::WTensor& wt) {
    return 8 + wt.name.size() + 4 + 8 * (size_t) wt.n_dims + 4 + 8;
}

bool write_tensor_info(WriteFile& wf, const GgufWriter::WTensor& wt, uint64_t offset) {
    if (!wf.put_str(wt.name)) return false;
    if (!wf.put_u32((uint32_t) wt.n_dims)) return false;
    for (int i = 0; i < wt.n_dims; ++i) {
        if (!wf.put_i64(wt.ne[i])) return false;
    }
    if (!wf.put_u32((uint32_t) wt.out_type)) return false;
    return wf.put_u64(offset);
}

// 写张量数据（原样分块拷贝；F32->F16 分块转换）
bool write_tensor_data(WriteFile& wf, const GgufWriter::WTensor& wt) {
    const Tensor* t = wt.tensor;
    if (wt.out_type == t->type) {
        constexpr size_t CHUNK = 1u << 16;
        std::vector<uint8_t> buf(CHUNK);
        size_t done = 0;
        while (done < wt.nbytes) {
            const size_t cur = std::min<size_t>(CHUNK, wt.nbytes - done);
            tensor_get(t, buf.data(), done, cur);
            if (!wf.put(buf.data(), cur)) return false;
            done += cur;
        }
        return true;
    }

    // F32 -> F16（ggml 写端不做转换，本库导出提供这一便利；数值用核心 fp32_to_fp16）
    const int64_t n      = tensor_nelements(t);
    constexpr int64_t CHUNK = 1 << 14;
    std::vector<float>    ftmp((size_t) CHUNK);
    std::vector<uint16_t> htmp((size_t) CHUNK);
    int64_t done = 0;
    while (done < n) {
        const int64_t cur = std::min<int64_t>(CHUNK, n - done);
        tensor_get(t, ftmp.data(), (size_t) done * sizeof(float), (size_t) cur * sizeof(float));
        for (int64_t i = 0; i < cur; ++i) {
            htmp[(size_t) i] = fp32_to_fp16(ftmp[(size_t) i]);
        }
        if (!wf.put(htmp.data(), (size_t) cur * sizeof(uint16_t))) return false;
        done += cur;
    }
    return true;
}

size_t writer_kv_find(const GgufWriter* w, const char* key) {
    for (size_t i = 0; i < w->kvs.size(); ++i) {
        if (w->kvs[i].key == key) return i;
    }
    return (size_t) -1;
}

GgufKv& writer_kv_slot(GgufWriter* w, const char* key, GgufType type) {
    const size_t idx = writer_kv_find(w, key);
    if (idx != (size_t) -1) {
        GgufKv& kv = w->kvs[idx];
        kv = GgufKv{};
        kv.key  = key;
        kv.type = type;
        return kv;
    }
    w->kvs.push_back(GgufKv{});
    GgufKv& kv = w->kvs.back();
    kv.key  = key;
    kv.type = type;
    return kv;
}

} // namespace

// ---------------------------------------------------------------- 公共 API：读

GgufFile* gguf_open(const char* path, bool load_data) {
    TRC_ASSERT(path != nullptr, "gguf_open: path 为空");
    if (!host_is_little_endian()) {
        TRC_LOG_ERROR("GGUF: 仅支持小端主机（大端平台未实现字节交换）");
        return nullptr;
    }

    std::error_code ec;
    const uint64_t  file_size = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        TRC_LOG_ERROR("GGUF: 无法访问文件 %s", path);
        return nullptr;
    }

    std::FILE* fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        TRC_LOG_ERROR("GGUF: 无法打开文件 %s", path);
        return nullptr;
    }

    ReadFile r;
    r.f    = fp;
    r.size = file_size;

    GgufFile* gf = gguf_parse(r, load_data);
    std::fclose(fp);
    return gf;
}

void gguf_close(GgufFile* f) {
    delete f;
}

int64_t gguf_version(const GgufFile* f) {
    return (int64_t) f->version;
}

size_t gguf_alignment(const GgufFile* f) {
    return (size_t) f->alignment;
}

size_t gguf_data_offset(const GgufFile* f) {
    return (size_t) f->data_offset;
}

int64_t gguf_kv_count(const GgufFile* f) {
    return (int64_t) f->kvs.size();
}

int64_t gguf_find_key(const GgufFile* f, const char* key) {
    const auto it = f->key_map.find(key);
    return it == f->key_map.end() ? -1 : it->second;
}

const char* gguf_key(const GgufFile* f, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) f->kvs.size(), "gguf_key: 下标越界 %lld",
               (long long) index);
    return f->kvs[(size_t) index].key.c_str();
}

GgufType gguf_kv_type(const GgufFile* f, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) f->kvs.size(), "gguf_kv_type: 下标越界 %lld",
               (long long) index);
    return f->kvs[(size_t) index].type;
}

GgufType gguf_arr_type(const GgufFile* f, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) f->kvs.size(), "gguf_arr_type: 下标越界 %lld",
               (long long) index);
    const GgufKv& kv = f->kvs[(size_t) index];
    if (kv.type != GgufType::ARRAY) {
        TRC_LOG_ERROR("GGUF: 键 %s 不是数组", kv.key.c_str());
        return GgufType::U8;
    }
    return kv.arr_type;
}

int64_t gguf_arr_count(const GgufFile* f, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) f->kvs.size(), "gguf_arr_count: 下标越界 %lld",
               (long long) index);
    const GgufKv& kv = f->kvs[(size_t) index];
    if (kv.type != GgufType::ARRAY) {
        TRC_LOG_ERROR("GGUF: 键 %s 不是数组", kv.key.c_str());
        return 0;
    }
    return kv.arr_n;
}

bool gguf_has_key(const GgufFile* f, const char* key) {
    return f->key_map.find(key) != f->key_map.end();
}

uint8_t gguf_get_u8(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::U8);
    return kv != nullptr ? (uint8_t) kv->u64 : 0;
}

int8_t gguf_get_i8(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::I8);
    return kv != nullptr ? (int8_t) (uint8_t) kv->u64 : 0;
}

uint16_t gguf_get_u16(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::U16);
    return kv != nullptr ? (uint16_t) kv->u64 : 0;
}

int16_t gguf_get_i16(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::I16);
    return kv != nullptr ? (int16_t) (uint16_t) kv->u64 : 0;
}

uint32_t gguf_get_u32(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::U32);
    return kv != nullptr ? (uint32_t) kv->u64 : 0;
}

int32_t gguf_get_i32(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::I32);
    return kv != nullptr ? (int32_t) (uint32_t) kv->u64 : 0;
}

float gguf_get_f32(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::F32);
    return kv != nullptr ? (float) kv->f64 : 0.0f;
}

bool gguf_get_bool(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::BOOL);
    return kv != nullptr && kv->b;
}

uint64_t gguf_get_u64(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::U64);
    return kv != nullptr ? kv->u64 : 0;
}

int64_t gguf_get_i64(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::I64);
    return kv != nullptr ? (int64_t) kv->u64 : 0;
}

double gguf_get_f64(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::F64);
    return kv != nullptr ? kv->f64 : 0.0;
}

const char* gguf_get_str(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv(f, key, GgufType::STRING);
    return kv != nullptr ? kv->str.c_str() : nullptr;
}

const int32_t* gguf_get_arr_i32(const GgufFile* f, const char* key, int64_t* n) {
    const GgufKv* kv = find_kv(f, key, GgufType::ARRAY);
    if (kv == nullptr) return nullptr;
    if (kv->arr_type != GgufType::I32) {
        TRC_LOG_ERROR("GGUF: 键 %s 的数组元素类型不是 i32", key);
        return nullptr;
    }
    if (n != nullptr) *n = kv->arr_n;
    return kv->arr_i32.data();
}

const int64_t* gguf_get_arr_i64(const GgufFile* f, const char* key, int64_t* n) {
    const GgufKv* kv = find_kv(f, key, GgufType::ARRAY);
    if (kv == nullptr) return nullptr;
    if (kv->arr_type != GgufType::I64) {
        TRC_LOG_ERROR("GGUF: 键 %s 的数组元素类型不是 i64", key);
        return nullptr;
    }
    if (n != nullptr) *n = kv->arr_n;
    return kv->arr_i64.data();
}

const float* gguf_get_arr_f32(const GgufFile* f, const char* key, int64_t* n) {
    const GgufKv* kv = find_kv(f, key, GgufType::ARRAY);
    if (kv == nullptr) return nullptr;
    if (kv->arr_type != GgufType::F32) {
        TRC_LOG_ERROR("GGUF: 键 %s 的数组元素类型不是 f32", key);
        return nullptr;
    }
    if (n != nullptr) *n = kv->arr_n;
    return kv->arr_f32.data();
}

const char* gguf_get_arr_str(const GgufFile* f, const char* key, int64_t index) {
    const GgufKv* kv = find_kv(f, key, GgufType::ARRAY);
    if (kv == nullptr) return nullptr;
    if (kv->arr_type != GgufType::STRING) {
        TRC_LOG_ERROR("GGUF: 键 %s 的数组元素类型不是字符串", key);
        return nullptr;
    }
    if (index < 0 || index >= kv->arr_n) {
        TRC_LOG_ERROR("GGUF: 键 %s 数组下标越界 %lld", key, (long long) index);
        return nullptr;
    }
    return kv->arr_str[(size_t) index].c_str();
}

int64_t gguf_tensor_count(const GgufFile* f) {
    return (int64_t) f->infos.size();
}

int64_t gguf_find_tensor(const GgufFile* f, const char* name) {
    const auto it = f->tensor_map.find(name);
    return it == f->tensor_map.end() ? -1 : it->second;
}

const GgufTensorInfo* gguf_tensor_info(const GgufFile* f, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) f->infos.size(), "gguf_tensor_info: 下标越界 %lld",
               (long long) index);
    return &f->infos[(size_t) index];
}

int64_t gguf_tensor_nelements(const GgufFile* f, int64_t index) {
    return tensor_nelements_of(*gguf_tensor_info(f, index));
}

bool gguf_tensor_to_f32(const GgufFile* f, const char* name, float* out, int64_t n_elems) {
    TRC_ASSERT(f != nullptr && name != nullptr && out != nullptr, "gguf_tensor_to_f32: 参数为空");
    const int64_t ti = gguf_find_tensor(f, name);
    if (ti < 0) {
        TRC_LOG_ERROR("GGUF: 张量不存在 %s", name);
        return false;
    }
    const GgufTensorInfo& info = f->infos[(size_t) ti];
    const int64_t n = tensor_nelements_of(info);
    if (n_elems < n) {
        TRC_LOG_ERROR("GGUF: 输出缓冲区太小（%lld < %lld）", (long long) n_elems, (long long) n);
        return false;
    }
    if (!f->has_data) {
        TRC_LOG_ERROR("GGUF: 打开时未加载数据（load_data=false），无法读取张量 %s", name);
        return false;
    }
    if (info.offset + info.nbytes > f->data.size()) {
        TRC_LOG_ERROR("GGUF: 张量 %s 数据越界", name);
        return false;
    }
    const uint8_t* src = f->data.data() + info.offset;

    switch (info.type) {
        case TYPE_F32:
            std::memcpy(out, src, (size_t) n * sizeof(float));
            return true;
        case TYPE_F16:
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h = 0;
                std::memcpy(&h, src + (size_t) i * 2, 2);
                out[i] = fp16_to_fp32(h);
            }
            return true;
        case TYPE_BF16:
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h = 0;
                std::memcpy(&h, src + (size_t) i * 2, 2);
                out[i] = bf16_to_fp32(h);
            }
            return true;
        case TYPE_F64:
            for (int64_t i = 0; i < n; ++i) {
                double v = 0.0;
                std::memcpy(&v, src + (size_t) i * 8, 8);
                out[i] = (float) v;
            }
            return true;
        case TYPE_I8:
            for (int64_t i = 0; i < n; ++i) {
                out[i] = (float) ((const int8_t*) src)[i];
            }
            return true;
        case TYPE_I16:
            for (int64_t i = 0; i < n; ++i) {
                int16_t v = 0;
                std::memcpy(&v, src + (size_t) i * 2, 2);
                out[i] = (float) v;
            }
            return true;
        case TYPE_I32:
            for (int64_t i = 0; i < n; ++i) {
                int32_t v = 0;
                std::memcpy(&v, src + (size_t) i * 4, 4);
                out[i] = (float) v;
            }
            return true;
        case TYPE_I64:
            for (int64_t i = 0; i < n; ++i) {
                int64_t v = 0;
                std::memcpy(&v, src + (size_t) i * 8, 8);
                out[i] = (float) v;
            }
            return true;
        case TYPE_Q4_0: dequantize_q4_0(src, n / 32, out); return true;
        case TYPE_Q4_1: dequantize_q4_1(src, n / 32, out); return true;
        case TYPE_Q5_0: dequantize_q5_0(src, n / 32, out); return true;
        case TYPE_Q5_1: dequantize_q5_1(src, n / 32, out); return true;
        case TYPE_Q8_0: dequantize_q8_0(src, n / 32, out); return true;
        default:
            TRC_LOG_ERROR("GGUF: 读取端暂不支持类型 %s（张量 %s）", type_name(info.type), name);
            return false;
    }
}

bool gguf_load_tensor(const GgufFile* f, const char* name, Tensor* dst) {
    TRC_ASSERT(f != nullptr && name != nullptr && dst != nullptr, "gguf_load_tensor: 参数为空");
    const int64_t ti = gguf_find_tensor(f, name);
    if (ti < 0) {
        TRC_LOG_ERROR("GGUF: 张量不存在 %s", name);
        return false;
    }
    const GgufTensorInfo& info = f->infos[(size_t) ti];

    if (!tensor_is_contiguous(dst)) {
        TRC_LOG_ERROR("GGUF: 目标张量 %s 必须连续", name);
        return false;
    }
    if (tensor_nelements(dst) != tensor_nelements_of(info)) {
        TRC_LOG_ERROR("GGUF: 张量 %s 元素数不匹配（目标 %lld，文件 %lld）", name,
                      (long long) tensor_nelements(dst), (long long) tensor_nelements_of(info));
        return false;
    }
    if (!f->has_data) {
        TRC_LOG_ERROR("GGUF: 打开时未加载数据（load_data=false），无法载入张量 %s", name);
        return false;
    }
    if (info.offset + info.nbytes > f->data.size()) {
        TRC_LOG_ERROR("GGUF: 张量 %s 数据越界", name);
        return false;
    }

    if (dst->type == info.type) {
        tensor_set(dst, f->data.data() + info.offset, 0, info.nbytes);
        return true;
    }
    if (dst->type == TYPE_F32) {
        std::vector<float> tmp((size_t) tensor_nelements_of(info));
        if (!gguf_tensor_to_f32(f, name, tmp.data(), (int64_t) tmp.size())) {
            return false;
        }
        tensor_set(dst, tmp.data(), 0, tmp.size() * sizeof(float));
        return true;
    }

    TRC_LOG_ERROR("GGUF: 张量 %s 类型不匹配（目标 %s，文件 %s），目标须为 F32 或同类型", name,
                  type_name(dst->type), type_name(info.type));
    return false;
}

Tensor* gguf_new_tensor(Context* ctx, const GgufFile* f, const char* name, Type type) {
    TRC_ASSERT(ctx != nullptr && name != nullptr, "gguf_new_tensor: 参数为空");
    const int64_t ti = gguf_find_tensor(f, name);
    if (ti < 0) {
        TRC_LOG_ERROR("GGUF: 张量不存在 %s", name);
        return nullptr;
    }
    if (!type_is_supported(type)) {
        TRC_LOG_ERROR("GGUF: 不支持创建类型 %s（一期仅基本类型）", type_name(type));
        return nullptr;
    }
    if (std::strlen(name) >= (size_t) MAX_NAME) {
        TRC_LOG_ERROR("GGUF: 张量名过长（%zu >= %d）：%s", std::strlen(name), MAX_NAME, name);
        return nullptr;
    }
    const GgufTensorInfo& info = f->infos[(size_t) ti];

    int n_dims = 1;
    for (int i = 1; i < MAX_DIMS; ++i) {
        if (info.ne[i] > 1) n_dims = i + 1;
    }
    const int64_t blck = type_blck_size(type);
    if (blck > 1 && info.ne[0] % blck != 0) {
        TRC_LOG_ERROR("GGUF: 张量 %s 的 ne0=%lld 不是类型 %s 块大小的整数倍", name,
                      (long long) info.ne[0], type_name(type));
        return nullptr;
    }

    Tensor* t = new_tensor_nd(ctx, type, n_dims, info.ne);
    tensor_set_name(t, "%s", name);
    return t;
}

bool gguf_validate(const char* path) {
    TRC_ASSERT(path != nullptr, "gguf_validate: path 为空");

    GgufFile* f = gguf_open(path, /*load_data=*/true);
    if (f == nullptr) {
        TRC_LOG_ERROR("GGUF 校验失败：%s", path);
        return false;
    }

    bool   ok       = true;
    size_t expected = 0;
    for (int64_t i = 0; i < gguf_tensor_count(f); ++i) {
        const GgufTensorInfo* info = gguf_tensor_info(f, i);
        if (info->offset != expected || info->offset + info->nbytes > f->data.size()) {
            TRC_LOG_ERROR("GGUF 校验失败：张量 %s 偏移/数据范围异常（offset=%zu nbytes=%zu 数据段=%zu）",
                          info->name, info->offset, info->nbytes, f->data.size());
            ok = false;
            break;
        }
        expected += align_up(info->nbytes, gguf_alignment(f));
    }

    if (ok) {
        TRC_LOG_INFO("GGUF 校验通过：%s（版本 %lld，%lld 个 KV，%lld 个张量，对齐 %zu）", path,
                     (long long) gguf_version(f), (long long) gguf_kv_count(f),
                     (long long) gguf_tensor_count(f), gguf_alignment(f));
    }
    gguf_close(f);
    return ok;
}

// ---------------------------------------------------------------- 公共 API：写

GgufWriter* gguf_writer_new() {
    return new GgufWriter();
}

void gguf_writer_free(GgufWriter* w) {
    delete w;
}

void gguf_writer_set_arch(GgufWriter* w, const char* arch) {
    gguf_writer_set_str(w, "general.architecture", arch);
}

namespace {

void writer_set_uint(GgufWriter* w, const char* key, GgufType type, uint64_t v) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_*: 参数为空");
    GgufKv& kv = writer_kv_slot(w, key, type);
    kv.u64 = v;
}

} // namespace

void gguf_writer_set_u8(GgufWriter* w, const char* key, uint8_t v) {
    writer_set_uint(w, key, GgufType::U8, (uint64_t) v);
}

void gguf_writer_set_i8(GgufWriter* w, const char* key, int8_t v) {
    writer_set_uint(w, key, GgufType::I8, (uint64_t) (int64_t) v);
}

void gguf_writer_set_u16(GgufWriter* w, const char* key, uint16_t v) {
    writer_set_uint(w, key, GgufType::U16, (uint64_t) v);
}

void gguf_writer_set_i16(GgufWriter* w, const char* key, int16_t v) {
    writer_set_uint(w, key, GgufType::I16, (uint64_t) (int64_t) v);
}

void gguf_writer_set_u32(GgufWriter* w, const char* key, uint32_t v) {
    writer_set_uint(w, key, GgufType::U32, (uint64_t) v);
}

void gguf_writer_set_i32(GgufWriter* w, const char* key, int32_t v) {
    writer_set_uint(w, key, GgufType::I32, (uint64_t) (int64_t) v);
}

void gguf_writer_set_f32(GgufWriter* w, const char* key, float v) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_f32: 参数为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::F32);
    kv.f64 = (double) v;
}

void gguf_writer_set_bool(GgufWriter* w, const char* key, bool v) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_bool: 参数为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::BOOL);
    kv.b = v;
}

void gguf_writer_set_u64(GgufWriter* w, const char* key, uint64_t v) {
    writer_set_uint(w, key, GgufType::U64, v);
}

void gguf_writer_set_i64(GgufWriter* w, const char* key, int64_t v) {
    writer_set_uint(w, key, GgufType::I64, (uint64_t) v);
}

void gguf_writer_set_f64(GgufWriter* w, const char* key, double v) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_f64: 参数为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::F64);
    kv.f64 = v;
}

void gguf_writer_set_str(GgufWriter* w, const char* key, const char* value) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key && value != nullptr,
               "gguf_writer_set_str: 参数为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::STRING);
    kv.str = value;
}

void gguf_writer_set_arr_i32(GgufWriter* w, const char* key, const int32_t* v, int64_t n) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_arr_i32: 参数为空");
    TRC_ASSERT(n >= 0 && (n == 0 || v != nullptr), "gguf_writer_set_arr_i32: 数组为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::ARRAY);
    kv.arr_type = GgufType::I32;
    kv.arr_n    = n;
    if (n > 0) {
        kv.arr_i32.assign(v, v + n);
    }
}

void gguf_writer_set_arr_i64(GgufWriter* w, const char* key, const int64_t* v, int64_t n) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_arr_i64: 参数为空");
    TRC_ASSERT(n >= 0 && (n == 0 || v != nullptr), "gguf_writer_set_arr_i64: 数组为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::ARRAY);
    kv.arr_type = GgufType::I64;
    kv.arr_n    = n;
    if (n > 0) {
        kv.arr_i64.assign(v, v + n);
    }
}

void gguf_writer_set_arr_f32(GgufWriter* w, const char* key, const float* v, int64_t n) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_arr_f32: 参数为空");
    TRC_ASSERT(n >= 0 && (n == 0 || v != nullptr), "gguf_writer_set_arr_f32: 数组为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::ARRAY);
    kv.arr_type = GgufType::F32;
    kv.arr_n    = n;
    if (n > 0) {
        kv.arr_f32.assign(v, v + n);
    }
}

void gguf_writer_set_arr_str(GgufWriter* w, const char* key, const char* const* v, int64_t n) {
    TRC_ASSERT(w != nullptr && key != nullptr && *key, "gguf_writer_set_arr_str: 参数为空");
    TRC_ASSERT(n >= 0 && (n == 0 || v != nullptr), "gguf_writer_set_arr_str: 数组为空");
    GgufKv& kv = writer_kv_slot(w, key, GgufType::ARRAY);
    kv.arr_type = GgufType::STRING;
    kv.arr_n    = n;
    kv.arr_str.clear();
    kv.arr_str.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        TRC_ASSERT(v[i] != nullptr, "gguf_writer_set_arr_str: 第 %lld 个字符串为空", (long long) i);
        kv.arr_str.push_back(v[i]);
    }
}

void gguf_writer_add_tensor(GgufWriter* w, const char* name, const Tensor* t, Type out_type) {
    TRC_ASSERT(w != nullptr && name != nullptr && t != nullptr, "gguf_writer_add_tensor: 参数为空");
    TRC_ASSERT(*name != '\0', "gguf_writer_add_tensor: 张量名不能为空");
    TRC_ASSERT(std::strlen(name) < (size_t) MAX_NAME,
               "gguf_writer_add_tensor: 张量名过长（%zu >= %d）：%s", std::strlen(name), MAX_NAME, name);
    for (const GgufWriter::WTensor& wt : w->tensors) {
        TRC_ASSERT(wt.name != name, "gguf_writer_add_tensor: 张量名重复 %s", name);
    }
    TRC_ASSERT(tensor_is_contiguous(t), "gguf_writer_add_tensor: 张量 %s 必须连续（先 cont 物化）",
               name);
    TRC_ASSERT(t->data != nullptr || t->buffer != nullptr,
               "gguf_writer_add_tensor: 张量 %s 尚未分配数据", name);

    const bool same_type   = (out_type == t->type);
    const bool f32_to_f16  = (out_type == TYPE_F16 && t->type == TYPE_F32);
    TRC_ASSERT(same_type || f32_to_f16, "gguf_writer_add_tensor: 不支持的类型转换 %s -> %s",
               type_name(t->type), type_name(out_type));

    GgufWriter::WTensor wt;
    wt.name     = name;
    wt.tensor   = t;
    wt.out_type = out_type;
    wt.n_dims   = tensor_n_dims(t);
    for (int i = 0; i < MAX_DIMS; ++i) {
        wt.ne[i] = t->ne[i];
    }
    wt.nbytes = same_type ? tensor_nbytes(t) : (size_t) tensor_nelements(t) * sizeof(uint16_t);
    w->tensors.push_back(std::move(wt));
}

int64_t gguf_writer_tensor_count(const GgufWriter* w) {
    return (int64_t) w->tensors.size();
}

bool gguf_writer_write(GgufWriter* w, const char* path) {
    TRC_ASSERT(w != nullptr && path != nullptr, "gguf_writer_write: 参数为空");

    const uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;

    uint64_t kv_bytes = 0;
    for (const GgufKv& kv : w->kvs) {
        kv_bytes += (uint64_t) kv_serialized_size(kv);
    }
    uint64_t ti_bytes = 0;
    for (const GgufWriter::WTensor& wt : w->tensors) {
        ti_bytes += (uint64_t) tensor_info_serialized_size(wt);
    }
    const uint64_t data_offset =
        (uint64_t) align_up((size_t) (24 + kv_bytes + ti_bytes), (size_t) alignment);

    std::FILE* fp = std::fopen(path, "wb");
    if (fp == nullptr) {
        TRC_LOG_ERROR("GGUF: 无法写入文件 %s", path);
        return false;
    }

    WriteFile wf;
    wf.f = fp;

    wf.put("GGUF", 4);
    wf.put_u32(GGUF_VERSION_WRITE);
    wf.put_u64((uint64_t) w->tensors.size());
    wf.put_u64((uint64_t) w->kvs.size());
    for (const GgufKv& kv : w->kvs) {
        if (!write_kv(wf, kv)) break;
    }
    uint64_t offset = 0;
    for (const GgufWriter::WTensor& wt : w->tensors) {
        if (!write_tensor_info(wf, wt, offset)) break;
        offset += (uint64_t) align_up(wt.nbytes, (size_t) alignment);
    }
    wf.pad_to(data_offset);
    for (const GgufWriter::WTensor& wt : w->tensors) {
        if (!write_tensor_data(wf, wt)) break;
        wf.pad_to_alignment(alignment);
    }

    if (std::fflush(fp) != 0) wf.ok = false;
    if (std::fclose(fp) != 0) wf.ok = false;

    if (!wf.ok) {
        TRC_LOG_ERROR("GGUF: 写出失败 %s", path);
        std::remove(path);
        return false;
    }
    return true;
}

} // namespace traincpp
