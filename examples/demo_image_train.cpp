// traincpp 图像分类演示 - 训练并导出模型
//
// 用法：
//   demo_image_train --synthetic [--classes 3] [--per-class 60] [--size 24] \
//                    [--epochs 20] [--batch 16] [--device cpu|gpu] [--export model.gguf]
//   demo_image_train --data <目录> [...]        # 目录下每个子目录为一类
//   demo_image_train gen-data --out <目录> [--classes 3] [--per-class 60] [--size 24]
//
// 数据目录格式：<目录>/<类别名>/*.bmp|.pgm|.ppm（灰度/彩色均可，自动转灰度并缩放）。
// 训练结束后用 demo_image_classify 加载导出的 GGUF 对图片分类。
#include "demo_image_common.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace traincpp;
using namespace demo;

int main(int argc, char** argv) {
    const std::string cmd = (argc > 1 && argv[1][0] != '-') ? argv[1] : "train";

    if (cmd == "gen-data") {
        const std::string out = arg_value(argc, argv, "--out", "demo_data");
        const int classes     = (int) arg_int(argc, argv, "--classes", 3);
        const int per_class   = (int) arg_int(argc, argv, "--per-class", 60);
        const int size        = (int) arg_int(argc, argv, "--size", 24);
        if (!generate_synthetic_dataset(out, classes, per_class, size, size, 1234u)) {
            std::fprintf(stderr, "生成数据失败: %s\n", out.c_str());
            return 1;
        }
        std::printf("已生成合成数据集: %s（%d 类 x %d 张，%dx%d PGM）\n", out.c_str(), classes,
                    per_class, size, size);
        return 0;
    }

    const int64_t S       = arg_int(argc, argv, "--size", 24);
    const int64_t epochs  = arg_int(argc, argv, "--epochs", 20);
    int64_t       batch   = arg_int(argc, argv, "--batch", 16);
    const std::string device = arg_value(argc, argv, "--device", "cpu");
    const std::string data   = arg_value(argc, argv, "--data", "");
    const std::string export_path = arg_value(argc, argv, "--export", "demo_model.gguf");

    if ((S % 4) != 0) {
        std::fprintf(stderr, "--size 需为 4 的倍数（当前 %lld）\n", (long long) S);
        return 1;
    }

    // ---------- 数据集 ----------
    Dataset ds;
    if (data.empty()) {
        const int classes   = (int) std::min<int64_t>(3, arg_int(argc, argv, "--classes", 3));
        const int per_class = (int) arg_int(argc, argv, "--per-class", 60);
        Rng rng;
        rng_seed(&rng, 1234u);
        ds.W = (int) S;
        ds.H = (int) S;
        ds.channels = 1;
        for (int c = 0; c < classes; ++c) {
            ds.class_names.push_back(class_dir_name(c));
            for (int i = 0; i < per_class; ++i) {
                std::vector<float> img;
                make_synthetic(c, (int) S, (int) S, &rng, img);
                ds.images.push_back(std::move(img));
                ds.labels.push_back(c);
            }
        }
        std::printf("使用内置合成数据集：%d 类 x %d 张\n", classes, per_class);
    } else {
        std::string err;
        if (!load_dataset(data, (int) S, (int) S, ds, err)) {
            std::fprintf(stderr, "加载数据失败: %s\n", err.c_str());
            return 1;
        }
        std::printf("加载数据集: %s（%zu 张，%zu 类）\n", data.c_str(), ds.images.size(),
                    ds.class_names.size());
    }

    const int64_t N = (int64_t) ds.images.size();
    const int64_t K = (int64_t) ds.class_names.size();
    if (N < 2 || K < 2) {
        std::fprintf(stderr, "样本或类别过少（N=%lld, K=%lld）\n", (long long) N, (long long) K);
        return 1;
    }
    if (batch > N) {
        batch = N;
    }

    Device* dev = pick_device(device);
    if (dev == nullptr) {
        std::fprintf(stderr, "CPU 后端不可用\n");
        return 1;
    }
    std::printf("设备: %s（%s）\n", dev->props().name.c_str(), dev->props().description.c_str());

    // ---------- 模型与图 ----------
    Context* ctx = context_new(1 << 24);
    Rng      rng;
    rng_seed(&rng, 20260101u);
    ConvModel model;
    model_init(ctx, model, S, K, &rng);

    Tensor* x      = new_tensor_4d(ctx, TYPE_F32, S, S, 1, batch);
    Tensor* target = new_tensor_2d(ctx, TYPE_F32, K, batch);
    tensor_set_name(x, "input");
    tensor_set_name(target, "target");

    Tensor* logits = model_forward(ctx, model, x);
    Tensor* loss   = cross_entropy_loss(ctx, logits, target);
    tensor_set_loss(loss);

    Graph*  g   = graph_new(ctx);
    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    AdamwOptions opt_opts;
    opt_opts.lr           = 0.005f;
    opt_opts.weight_decay = 0.0f;
    Optimizer* opt = optim_adamw_new(ctx, model.params.items.data(), param_list_count(&model.params),
                                     opt_opts);

    // ---------- 训练/评估辅助 ----------
    std::mt19937 shuffle_rng(1234u);
    std::vector<int> order((size_t) N);
    std::iota(order.begin(), order.end(), 0);

    std::vector<float> xbuf((size_t) (batch * S * S));
    std::vector<float> tbuf((size_t) (batch * K));

    auto fill_batch = [&](const std::vector<int>& idx, int64_t count) {
        std::fill(xbuf.begin(), xbuf.end(), 0.0f);
        std::fill(tbuf.begin(), tbuf.end(), 0.0f);
        for (int64_t b = 0; b < count; ++b) {
            const int    sample = idx[(size_t) b];
            const float* src    = ds.images[(size_t) sample].data();
            std::copy(src, src + S * S, xbuf.begin() + (size_t) (b * S * S));
            tbuf[(size_t) (b * K + ds.labels[(size_t) sample])] = 1.0f;
        }
        tensor_set(x, xbuf.data(), 0, xbuf.size() * sizeof(float));
        tensor_set(target, tbuf.data(), 0, tbuf.size() * sizeof(float));
    };

    auto evaluate = [&](float& out_loss, float& out_acc) {
        const int64_t steps = N / batch;
        double sum_loss = 0.0;
        int64_t correct = 0;
        std::vector<float> probs((size_t) K);
        for (int64_t s = 0; s < steps; ++s) {
            std::vector<int> idx((size_t) batch);
            for (int64_t b = 0; b < batch; ++b) {
                idx[(size_t) b] = (int) (s * batch + b);
            }
            fill_batch(idx, batch);
            graph_reset(g);
            dev->graph_compute(g);
            float lv = 0.0f;
            tensor_get(loss, &lv, 0, sizeof(lv));
            sum_loss += (double) lv;
            for (int64_t b = 0; b < batch; ++b) {
                tensor_get(logits, probs.data(), (size_t) (b * K) * sizeof(float), K * sizeof(float));
                int best = 0;
                for (int64_t k = 1; k < K; ++k) {
                    if (probs[(size_t) k] > probs[(size_t) best]) {
                        best = (int) k;
                    }
                }
                if (best == ds.labels[(size_t) idx[(size_t) b]]) {
                    ++correct;
                }
            }
        }
        out_loss = (float) (sum_loss / (double) steps);
        out_acc  = steps > 0 ? (float) correct / (float) (steps * batch) : 0.0f;
    };

    // ---------- 训练循环 ----------
    for (int64_t e = 1; e <= epochs; ++e) {
        std::shuffle(order.begin(), order.end(), shuffle_rng);
        double sum_loss = 0.0;
        const int64_t steps = N / batch;
        for (int64_t s = 0; s < steps; ++s) {
            std::vector<int> idx((size_t) batch);
            for (int64_t b = 0; b < batch; ++b) {
                idx[(size_t) b] = order[(size_t) (s * batch + b)];
            }
            fill_batch(idx, batch);
            graph_reset(g);
            dev->graph_compute(g);
            float lv = 0.0f;
            tensor_get(loss, &lv, 0, sizeof(lv));
            sum_loss += (double) lv;
            optim_step(opt);
        }
        float eval_loss = 0.0f, eval_acc = 0.0f;
        evaluate(eval_loss, eval_acc);
        std::printf("epoch %3lld/%lld  train_loss %.4f  eval_loss %.4f  eval_acc %.1f%%\n",
                    (long long) e, (long long) epochs, sum_loss / (double) steps, (double) eval_loss,
                    (double) eval_acc * 100.0);
    }

    // ---------- 导出 GGUF ----------
    {
        GgufWriter* w = gguf_writer_new();
        gguf_writer_set_arch(w, "demo.image_classifier");
        gguf_writer_set_u32(w, "demo.image_size", (uint32_t) S);
        gguf_writer_set_u32(w, "demo.channels", 1);
        gguf_writer_set_u32(w, "demo.class_count", (uint32_t) K);
        std::vector<const char*> names;
        names.reserve(ds.class_names.size());
        for (const std::string& s : ds.class_names) {
            names.push_back(s.c_str());
        }
        gguf_writer_set_arr_str(w, "demo.classes", names.data(), (int64_t) names.size());
        for (int64_t i = 0; i < param_list_count(&model.params); ++i) {
            Tensor* p = param_list_get(&model.params, i);
            gguf_writer_add_tensor(w, p->name, p, TYPE_F32);
        }
        const bool wrote = gguf_writer_write(w, export_path.c_str());
        gguf_writer_free(w);
        if (!wrote) {
            std::fprintf(stderr, "导出模型失败: %s\n", export_path.c_str());
            return 1;
        }
        gguf_validate(export_path.c_str());
        std::printf("已导出模型: %s\n", export_path.c_str());
    }

    optim_free(opt);
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    std::printf("训练完成。用 demo_image_classify --model %s <图片> 进行分类。\n",
                export_path.c_str());
    return 0;
}
