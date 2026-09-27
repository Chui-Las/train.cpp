// traincpp 图像分类演示 - 共享工具（仅示例使用，不是库的一部分）
//
// 内容：
//   - 最小图片解码：BMP（24/32 位未压缩）、PGM（P2/P5）、PPM（P3/P6）-> 灰度 [0,1]
//   - PGM（P5）写出、最近邻缩放
//   - 内置合成数据集（横条纹 / 竖条纹 / 棋盘格），以及磁盘数据集扫描
//   - 小型 CNN 的构建与前向（Conv2d x3 + Linear x2）
//   - 命令行参数辅助
#pragma once

#include "traincpp/traincpp.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace demo {

namespace fs = std::filesystem;

inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// ---------------------------------------------------------------- 字节读取

inline bool read_file_bytes(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    if (n <= 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    in.read(reinterpret_cast<char*>(out.data()), n);
    return static_cast<bool>(in);
}

// ---------------------------------------------------------------- Netpbm（PGM/PPM）解析

// 从字节流取下一个 ASCII token（跳过空白与 '#' 注释）
inline bool pbm_next_token(const std::vector<unsigned char>& d, size_t& pos, std::string& tok) {
    while (pos < d.size()) {
        const unsigned char c = d[pos];
        if (c == '#') {
            while (pos < d.size() && d[pos] != '\n') {
                ++pos;
            }
        } else if (std::isspace(c) != 0) {
            ++pos;
        } else {
            break;
        }
    }
    if (pos >= d.size()) {
        return false;
    }
    const size_t start = pos;
    while (pos < d.size() && std::isspace(d[pos]) == 0 && d[pos] != '#') {
        ++pos;
    }
    tok.assign(reinterpret_cast<const char*>(d.data()) + start, pos - start);
    return true;
}

// 解析 PGM/PPM，输出灰度 [0,1]（行主序：外层 y，内层 x）
inline bool load_netpbm(const std::vector<unsigned char>& d, int& w, int& h, std::vector<float>& gray) {
    size_t pos = 0;
    std::string magic;
    if (!pbm_next_token(d, pos, magic) || magic.size() != 2 || magic[0] != 'P') {
        return false;
    }
    std::string tok;
    if (!pbm_next_token(d, pos, tok)) {
        return false;
    }
    w = std::atoi(tok.c_str());
    if (!pbm_next_token(d, pos, tok)) {
        return false;
    }
    h = std::atoi(tok.c_str());
    if (!pbm_next_token(d, pos, tok)) {
        return false;
    }
    const int maxval = std::atoi(tok.c_str());
    if (w <= 0 || h <= 0 || maxval <= 0) {
        return false;
    }
    const size_t npix = (size_t) w * (size_t) h;
    gray.assign(npix, 0.0f);
    const float scale = 1.0f / (float) maxval;

    if (magic == "P5" || magic == "P6") {
        if (pos < d.size() && std::isspace(d[pos]) != 0) {
            ++pos;  // 头部与二进制数据之间恰好一个空白
        }
        const int channels = (magic == "P6") ? 3 : 1;
        if (d.size() - pos < npix * (size_t) channels) {
            return false;
        }
        for (size_t i = 0; i < npix; ++i) {
            if (channels == 1) {
                gray[i] = (float) d[pos + i] * scale;
            } else {
                const float r = (float) d[pos + i * 3 + 0];
                const float g = (float) d[pos + i * 3 + 1];
                const float b = (float) d[pos + i * 3 + 2];
                gray[i] = (0.299f * r + 0.587f * g + 0.114f * b) * scale;
            }
        }
        return true;
    }

    // ASCII 变体（P2/P3）
    const int channels = (magic == "P3") ? 3 : 1;
    for (size_t i = 0; i < npix; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            if (!pbm_next_token(d, pos, tok)) {
                return false;
            }
            const float v = (float) std::atoi(tok.c_str());
            sum += (channels == 1) ? v : ((c == 0) ? 0.299f * v : ((c == 1) ? 0.587f * v : 0.114f * v));
        }
        gray[i] = sum * scale;
    }
    return true;
}

// ---------------------------------------------------------------- BMP 解析

inline uint32_t rd_u32(const unsigned char* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}
inline uint16_t rd_u16(const unsigned char* p) {
    return (uint16_t) ((uint16_t) p[0] | ((uint16_t) p[1] << 8));
}

inline bool load_bmp(const std::vector<unsigned char>& d, int& w, int& h, std::vector<float>& gray) {
    if (d.size() < 54 || d[0] != 'B' || d[1] != 'M') {
        return false;
    }
    const uint32_t data_off = rd_u32(&d[10]);
    const uint32_t hdr_size = rd_u32(&d[14]);
    if (hdr_size < 40) {
        return false;  // 仅支持 BITMAPINFOHEADER 及以上
    }
    const int32_t  wi = (int32_t) rd_u32(&d[18]);
    const int32_t  hi = (int32_t) rd_u32(&d[22]);
    const uint16_t bpp = rd_u16(&d[28]);
    const uint32_t comp = rd_u32(&d[30]);
    if (comp != 0 || (bpp != 24 && bpp != 32) || wi <= 0 || hi == 0) {
        return false;  // 仅支持未压缩 24/32 位
    }
    const int width    = wi;
    const int height   = (hi < 0) ? -hi : hi;
    const bool topdown = (hi < 0);
    const size_t stride = ((size_t) width * bpp / 8 + 3) & ~(size_t) 3;
    if (d.size() < data_off + stride * (size_t) height) {
        return false;
    }
    w = width;
    h = height;
    gray.assign((size_t) width * (size_t) height, 0.0f);
    const int bytes = bpp / 8;
    for (int y = 0; y < height; ++y) {
        const int src_row = topdown ? y : (height - 1 - y);
        const unsigned char* row = d.data() + data_off + (size_t) src_row * stride;
        for (int x = 0; x < width; ++x) {
            const unsigned char b = row[(size_t) x * bytes + 0];
            const unsigned char g = row[(size_t) x * bytes + 1];
            const unsigned char r = row[(size_t) x * bytes + 2];
            gray[(size_t) y * (size_t) width + x] =
                (0.299f * (float) r + 0.587f * (float) g + 0.114f * (float) b) / 255.0f;
        }
    }
    return true;
}

inline bool load_gray(const std::string& path, int& w, int& h, std::vector<float>& gray) {
    std::vector<unsigned char> d;
    if (!read_file_bytes(path, d)) {
        return false;
    }
    if (d.size() >= 2 && d[0] == 'B' && d[1] == 'M') {
        return load_bmp(d, w, h, gray);
    }
    if (d.size() >= 2 && d[0] == 'P') {
        return load_netpbm(d, w, h, gray);
    }
    return false;
}

inline void resize_nearest(const std::vector<float>& src, int sw, int sh, int dw, int dh,
                           std::vector<float>& dst) {
    dst.assign((size_t) dw * (size_t) dh, 0.0f);
    for (int y = 0; y < dh; ++y) {
        const int sy = (int) ((int64_t) y * sh / dh);
        for (int x = 0; x < dw; ++x) {
            const int sx = (int) ((int64_t) x * sw / dw);
            dst[(size_t) y * (size_t) dw + x] = src[(size_t) sy * (size_t) sw + sx];
        }
    }
}

inline bool load_gray_resized(const std::string& path, int W, int H, std::vector<float>& out) {
    int w = 0, h = 0;
    std::vector<float> gray;
    if (!load_gray(path, w, h, gray)) {
        return false;
    }
    if (w == W && h == H) {
        out = std::move(gray);
    } else {
        resize_nearest(gray, w, h, W, H, out);
    }
    return true;
}

inline bool save_pgm(const std::string& path, const std::vector<float>& gray, int w, int h) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P5\n" << w << " " << h << "\n255\n";
    std::vector<unsigned char> row((size_t) w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            row[(size_t) x] = (unsigned char) (clamp01(gray[(size_t) y * (size_t) w + x]) * 255.0f + 0.5f);
        }
        out.write(reinterpret_cast<const char*>(row.data()), w);
    }
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------- 合成数据

inline const char* synth_class_name(int c) {
    switch (c) {
        case 0: return "horizontal";  // 横条纹
        case 1: return "vertical";    // 竖条纹
        case 2: return "checker";     // 棋盘格
        default: return "class";
    }
}

// 生成一张合成图（灰度 [0,1]），类别语义：0 横条纹、1 竖条纹、2 棋盘格
inline void make_synthetic(int cls, int W, int H, traincpp::Rng* rng, std::vector<float>& out) {
    out.resize((size_t) W * (size_t) H);
    const float freq  = 2.0f + traincpp::rng_uniform01(rng) * 4.0f;   // 2..6
    const float phase = traincpp::rng_uniform01(rng) * 6.2831853f;
    const float bright = 0.65f + traincpp::rng_uniform01(rng) * 0.3f;
    const float two_pi = 6.2831853f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float v;
            if (cls == 0) {
                v = 0.5f + 0.5f * std::sin(two_pi * (float) y * freq / (float) H + phase);
            } else if (cls == 1) {
                v = 0.5f + 0.5f * std::sin(two_pi * (float) x * freq / (float) W + phase);
            } else {
                const int cx = (int) ((float) x * freq / (float) W);
                const int cy = (int) ((float) y * freq / (float) H);
                v = (((cx + cy) & 1) != 0) ? 1.0f : 0.0f;
            }
            v = v * bright + (traincpp::rng_uniform01(rng) - 0.5f) * 0.12f;
            out[(size_t) y * (size_t) W + x] = clamp01(v);
        }
    }
}

inline std::string class_dir_name(int c) {
    if (c < 3) {
        return synth_class_name(c);
    }
    return "class" + std::to_string(c);
}

// 把合成数据集写成 PGM 文件：<root>/<class>/<class>_<i>.pgm
inline bool generate_synthetic_dataset(const std::string& root, int classes, int per_class, int W,
                                       int H, uint32_t seed) {
    traincpp::Rng rng;
    traincpp::rng_seed(&rng, seed);
    for (int c = 0; c < classes; ++c) {
        const fs::path dir = fs::path(root) / class_dir_name(c);
        std::error_code ec;
        fs::create_directories(dir, ec);
        for (int i = 0; i < per_class; ++i) {
            std::vector<float> img;
            make_synthetic(c, W, H, &rng, img);
            const fs::path file = dir / (class_dir_name(c) + "_" + std::to_string(i) + ".pgm");
            if (!save_pgm(file.string(), img, W, H)) {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------- 数据集

struct Dataset {
    int                        W = 0;
    int                        H = 0;
    int                        channels = 1;
    std::vector<std::string>   class_names;
    std::vector<int>           labels;
    std::vector<std::vector<float>> images;  // 每张为 W*H 灰度（行主序）
};

inline bool has_image_ext(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return e == ".bmp" || e == ".pgm" || e == ".ppm" || e == ".pnm";
}

// 扫描 <root>/<类名>/*.{bmp,pgm,ppm}，缩放到 WxH 灰度
inline bool load_dataset(const std::string& root, int W, int H, Dataset& out, std::string& err) {
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        err = "数据目录不存在: " + root;
        return false;
    }
    std::vector<fs::path> class_dirs;
    for (const auto& e : fs::directory_iterator(root, ec)) {
        if (e.is_directory()) {
            class_dirs.push_back(e.path());
        }
    }
    std::sort(class_dirs.begin(), class_dirs.end());
    if (class_dirs.empty()) {
        err = "数据目录下没有类别子目录: " + root;
        return false;
    }
    out = Dataset{};
    out.W = W;
    out.H = H;
    out.channels = 1;
    for (const fs::path& dir : class_dirs) {
        out.class_names.push_back(dir.filename().string());
        std::vector<fs::path> files;
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            if (e.is_regular_file() && has_image_ext(e.path())) {
                files.push_back(e.path());
            }
        }
        std::sort(files.begin(), files.end());
        for (const fs::path& f : files) {
            std::vector<float> img;
            if (!load_gray_resized(f.string(), W, H, img)) {
                std::fprintf(stderr, "[demo] 跳过无法读取的图片: %s\n", f.string().c_str());
                continue;
            }
            out.images.push_back(std::move(img));
            out.labels.push_back((int) out.class_names.size() - 1);
        }
    }
    if (out.images.empty()) {
        err = "数据目录下没有可用图片: " + root;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- 小型 CNN

// 输入 [S,S,1,N] -> Conv2d(1->8) -> Conv2d(8->16,s2) -> Conv2d(16->16,s2)
//               -> reshape -> Linear(F->64) -> Linear(64->K) -> logits [K,N]
// 刻意不使用池化（pool_2d_back 无 Vulkan 实现），确保 GPU 可训练。
struct ConvModel {
    traincpp::Conv2d    c1{};
    traincpp::Conv2d    c2{};
    traincpp::Conv2d    c3{};
    traincpp::Linear    fc1{};
    traincpp::Linear    fc2{};
    traincpp::ParamList params{};
    int64_t             size    = 0;
    int64_t             classes = 0;
    int64_t             flat    = 0;
};

inline int64_t conv_out_size(int64_t n, int64_t p, int64_t k, int64_t s, int64_t d) {
    return (n + 2 * p - d * (k - 1) - 1) / s + 1;
}

inline void model_init(traincpp::Context* ctx, ConvModel& m, int64_t S, int64_t K,
                       traincpp::Rng* rng) {
    m.size    = S;
    m.classes = K;
    const int64_t s2 = conv_out_size(S, 1, 3, 2, 1);
    const int64_t s3 = conv_out_size(s2, 1, 3, 2, 1);
    m.flat = s3 * s3 * 16;

    traincpp::conv2d_init(ctx, &m.c1, 1, 8, 3, 3, 1, 1, 1, 1, 1, 1, true, rng, "c1");
    traincpp::conv2d_init(ctx, &m.c2, 8, 16, 3, 3, 2, 2, 1, 1, 1, 1, true, rng, "c2");
    traincpp::conv2d_init(ctx, &m.c3, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, true, rng, "c3");
    traincpp::linear_init(ctx, &m.fc1, m.flat, 64, true, rng, "fc1");
    traincpp::linear_init(ctx, &m.fc2, 64, K, true, rng, "fc2");

    traincpp::conv2d_params(&m.c1, &m.params);
    traincpp::conv2d_params(&m.c2, &m.params);
    traincpp::conv2d_params(&m.c3, &m.params);
    traincpp::linear_params(&m.fc1, &m.params);
    traincpp::linear_params(&m.fc2, &m.params);
}

inline traincpp::Tensor* model_forward(traincpp::Context* ctx, const ConvModel& m,
                                       traincpp::Tensor* x) {
    traincpp::Tensor* y = traincpp::relu(ctx, traincpp::conv2d_forward(ctx, &m.c1, x));
    y = traincpp::relu(ctx, traincpp::conv2d_forward(ctx, &m.c2, y));
    y = traincpp::relu(ctx, traincpp::conv2d_forward(ctx, &m.c3, y));
    const int64_t N = x->ne[3];
    y = traincpp::reshape_2d(ctx, y, m.flat, N);
    y = traincpp::relu(ctx, traincpp::linear_forward(ctx, &m.fc1, y));
    return traincpp::linear_forward(ctx, &m.fc2, y);
}

// ---------------------------------------------------------------- 命令行辅助

inline bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (key == argv[i]) {
            return true;
        }
    }
    return false;
}

inline std::string arg_value(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return def;
}

inline int64_t arg_int(int argc, char** argv, const std::string& key, int64_t def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return std::strtoll(argv[i + 1], nullptr, 10);
        }
    }
    return def;
}

inline traincpp::Device* pick_device(const std::string& want) {
    if (want == "gpu" || want == "vulkan") {
        traincpp::Device* d = traincpp::device_by_type(traincpp::DeviceType::GPU);
        if (d == nullptr) {
            d = traincpp::device_by_type(traincpp::DeviceType::IGPU);
        }
        if (d != nullptr) {
            return d;
        }
        std::fprintf(stderr, "[demo] 未找到 Vulkan 设备，回退 CPU\n");
    }
    return traincpp::device_cpu();
}

}  // namespace demo
