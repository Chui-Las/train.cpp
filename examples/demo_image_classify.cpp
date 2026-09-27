// traincpp 图像分类演示 - 加载模型并对图片分类
//
// 用法：
//   demo_image_classify --model demo_model.gguf <图片> [<图片> ...] [--device cpu|gpu]
//
// 支持图片格式：BMP（24/32 位未压缩）、PGM、PPM；自动转灰度并缩放到模型输入尺寸。
#include "demo_image_common.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace traincpp;
using namespace demo;

int main(int argc, char** argv) {
    const std::string model_path = arg_value(argc, argv, "--model", "");
    const std::string device     = arg_value(argc, argv, "--device", "cpu");
    if (model_path.empty()) {
        std::fprintf(stderr,
                     "用法: demo_image_classify --model <model.gguf> <图片> [<图片> ...] "
                     "[--device cpu|gpu]\n");
        return 1;
    }

    std::vector<std::string> images;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" || a == "--device") {
            ++i;  // 跳过值
        } else if (!a.empty() && a[0] != '-') {
            images.push_back(a);
        }
    }
    if (images.empty()) {
        std::fprintf(stderr, "未提供图片路径\n");
        return 1;
    }

    GgufFile* f = gguf_open(model_path.c_str(), true);
    if (f == nullptr) {
        std::fprintf(stderr, "无法打开模型: %s\n", model_path.c_str());
        return 1;
    }
    const int64_t S      = (int64_t) gguf_get_u32(f, "demo.image_size");
    const int64_t K      = (int64_t) gguf_get_u32(f, "demo.class_count");
    if (S <= 0 || K <= 1) {
        std::fprintf(stderr, "模型元数据缺失或非法（image_size=%lld, class_count=%lld）\n",
                     (long long) S, (long long) K);
        gguf_close(f);
        return 1;
    }
    std::vector<std::string> class_names;
    {
        const int64_t ki = gguf_find_key(f, "demo.classes");
        const int64_t nc = (ki >= 0) ? gguf_arr_count(f, ki) : 0;
        for (int64_t i = 0; i < nc; ++i) {
            const char* s = gguf_get_arr_str(f, "demo.classes", i);
            class_names.push_back(s != nullptr ? s : "?");
        }
    }
    if ((int64_t) class_names.size() != K) {
        std::fprintf(stderr, "类别名数量与 class_count 不一致\n");
        gguf_close(f);
        return 1;
    }

    Device* dev = pick_device(device);
    Context* ctx = context_new(1 << 24);
    Rng      rng;
    rng_seed(&rng, 1u);
    ConvModel model;
    model_init(ctx, model, S, K, &rng);

    Tensor* x      = new_tensor_4d(ctx, TYPE_F32, S, S, 1, 1);
    Tensor* logits = model_forward(ctx, model, x);
    Tensor* probs  = soft_max(ctx, logits);
    tensor_set_name(x, "input");

    Graph*  g   = graph_new(ctx);
    graph_build_forward_expand(ctx, g, probs);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    // 按名载入权重
    for (int64_t i = 0; i < param_list_count(&model.params); ++i) {
        Tensor* p = param_list_get(&model.params, i);
        if (!gguf_load_tensor(f, p->name, p)) {
            std::fprintf(stderr, "载入权重失败: %s\n", p->name);
            return 1;
        }
    }
    gguf_close(f);

    std::printf("模型: %s（输入 %lldx%lld，%lld 类）\n", model_path.c_str(), (long long) S,
                (long long) S, (long long) K);
    std::printf("设备: %s\n", dev->props().name.c_str());

    std::vector<float> xbuf((size_t) (S * S));
    std::vector<float> out((size_t) K);
    for (const std::string& path : images) {
        if (!load_gray_resized(path, (int) S, (int) S, xbuf)) {
            std::fprintf(stderr, "[跳过] 无法读取图片: %s\n", path.c_str());
            continue;
        }
        tensor_set(x, xbuf.data(), 0, xbuf.size() * sizeof(float));
        dev->graph_compute(g);
        tensor_get(probs, out.data(), 0, out.size() * sizeof(float));

        int best = 0;
        for (int64_t k = 1; k < K; ++k) {
            if (out[(size_t) k] > out[(size_t) best]) {
                best = (int) k;
            }
        }
        std::printf("%s\n  预测: %s  (%.1f%%)\n", path.c_str(), class_names[(size_t) best].c_str(),
                    (double) out[(size_t) best] * 100.0);
        std::printf("  概率:");
        for (int64_t k = 0; k < K; ++k) {
            std::printf(" %s=%.3f", class_names[(size_t) k].c_str(), (double) out[(size_t) k]);
        }
        std::printf("\n");
    }

    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 0;
}
