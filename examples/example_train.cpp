// train.cpp - 训练示例：小 MLP 回归（Linear+ReLU+Linear + MSE + AdamW）
//
// 演示一条完整、可复用的训练链路（全部使用公共 API）：
//   1) nn 层系统搭模型；ParamList 收集参数（*_init 已自动标记 TENSOR_FLAG_PARAM）
//   2) 前向 → graph_build_backward_expand → buffer_alloc_ctx_tensors（顺序固定）
//   3) 训练循环：graph_reset → graph_compute → optim_step
//   4) checkpoint 保存 / 探测 / 恢复（新 session 载入后 loss 一致）
//   5) GGUF 导出 + gguf_validate 自检 + gguf_tensor_to_f32 回读比对（ggml 生态可读）
#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace traincpp;

namespace {

std::vector<float> read_f32(const Tensor* t) {
    std::vector<float> v((size_t) tensor_nelements(t));
    tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

float read_scalar(const Tensor* t) {
    float v = 0.0f;
    tensor_get(t, &v, 0, sizeof(float));
    return v;
}

// 一个独立训练会话 = Context + Graph + Buffer + 模型 + 优化器
struct Session {
    static constexpr int64_t IN    = 4;
    static constexpr int64_t HID   = 8;
    static constexpr int64_t OUT   = 1;
    static constexpr int64_t BATCH = 16;

    Context* ctx = nullptr;
    Graph*   g   = nullptr;
    Buffer*  buf = nullptr;
    Device*  dev = nullptr;
    Rng      rng;

    Linear    l1{};
    Linear    l2{};
    ParamList params{};
    Tensor*   x      = nullptr;
    Tensor*   target = nullptr;
    Tensor*   loss   = nullptr;

    Optimizer* opt = nullptr;

    Session(Device* d, uint32_t seed) : dev(d) {
        ctx = context_new(1 << 22);
        g   = graph_new(ctx);
        rng_seed(&rng, seed);

        // 1) 模型：Linear(4→8) → ReLU → Linear(8→1)
        linear_init(ctx, &l1, IN, HID, true, &rng, "fc1");
        linear_init(ctx, &l2, HID, OUT, true, &rng, "fc2");

        // 2) 收集参数（*_init 已自动标记可训练；冻结/解冻用 param_list_set_param）
        ParamList p1;
        ParamList p2;
        linear_params(&l1, &p1);
        linear_params(&l2, &p2);
        param_list_merge(&params, &p1);
        param_list_merge(&params, &p2);

        x      = new_tensor_2d(ctx, TYPE_F32, IN, BATCH);
        target = new_tensor_2d(ctx, TYPE_F32, OUT, BATCH);

        Tensor* pred = linear_forward(ctx, &l2, relu(ctx, linear_forward(ctx, &l1, x)));
        loss = mse_loss(ctx, pred, target);
        tensor_set_loss(loss);

        // 3) 图构建顺序：前向 → 反向展开 → 统一分配（不可交换）
        graph_build_forward_expand(ctx, g, loss);
        graph_build_backward_expand(g);
        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

        // 4) 数据：y = 2*x0 - 3*x1 + x2 + 0.5*x3 + 1
        std::vector<float> vx((size_t) (IN * BATCH));
        std::vector<float> vt((size_t) BATCH);
        for (int64_t i = 0; i < BATCH; ++i) {
            const float x0 = std::sin((float) i * 0.7f);
            const float x1 = std::cos((float) i * 0.31f);
            const float x2 = (float) (i % 5) * 0.2f - 0.4f;
            const float x3 = (float) (i % 3) * 0.5f - 0.5f;
            vx[(size_t) (i * IN + 0)] = x0;
            vx[(size_t) (i * IN + 1)] = x1;
            vx[(size_t) (i * IN + 2)] = x2;
            vx[(size_t) (i * IN + 3)] = x3;
            vt[(size_t) i] = 2.0f * x0 - 3.0f * x1 + x2 + 0.5f * x3 + 1.0f;
        }
        tensor_set(x, vx.data(), 0, vx.size() * sizeof(float));
        tensor_set(target, vt.data(), 0, vt.size() * sizeof(float));

        // 5) 优化器（主机端实现，CPU/Vulkan 后端通用）
        AdamwOptions o;
        o.lr           = 0.05f;
        o.weight_decay = 0.0f;
        opt = optim_adamw_new(ctx, params.items.data(), param_list_count(&params), o);
    }

    ~Session() {
        if (opt != nullptr) {
            optim_free(opt);
        }
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }

    void train_step() {
        graph_reset(g);          // 清零梯度 + LOSS 播种
        dev->graph_compute(g);   // 一次 compute 完成前向 + 反向
        optim_step(opt);         // 更新参数
    }

    float eval_loss() {
        graph_reset(g);
        dev->graph_compute(g);
        return read_scalar(loss);
    }
};

} // namespace

int main() {
    Device* dev = device_cpu();
    if (dev == nullptr) {
        std::fprintf(stderr, "CPU 后端未启用\n");
        return 1;
    }

    // ---------- 训练 ----------
    Session a(dev, 1234u);
    const float loss0 = a.eval_loss();
    std::printf("训练前 loss = %.6g\n", loss0);

    constexpr int STEPS = 2000;
    for (int i = 1; i <= STEPS; ++i) {
        a.train_step();
        if (i % 500 == 0) {
            std::printf("  step %4d  loss = %.6g\n", i, a.eval_loss());
        }
    }

    // ---------- checkpoint 保存 / 探测 ----------
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::string ckpt_path  = (dir / "traincpp_example_train.ckpt.gguf").string();
    const std::string model_path = (dir / "traincpp_example_train.gguf").string();

    if (!checkpoint_save(ckpt_path.c_str(), CheckpointItems{a.opt, &a.rng, nullptr}, "example.mlp")) {
        std::fprintf(stderr, "checkpoint 保存失败\n");
        return 1;
    }
    CheckpointInfo info;
    if (checkpoint_probe(ckpt_path.c_str(), &info)) {
        std::printf("checkpoint: arch=%s 优化器=%s step=%lld 参数=%lld\n", info.model_arch,
                    info.optimizer_type, (long long) info.step, (long long) info.param_count);
    }

    // ---------- 恢复：新 session（不同初值）载入后 loss 应一致 ----------
    {
        Session b(dev, 4321u);
        optim_alloc_state(b.opt);   // 文件含优化器状态时，载入前先分配状态
        if (!checkpoint_load(ckpt_path.c_str(), CheckpointItems{b.opt, &b.rng, nullptr})) {
            std::fprintf(stderr, "checkpoint 载入失败\n");
            return 1;
        }
        const float loss_a = a.eval_loss();
        const float loss_b = b.eval_loss();
        std::printf("恢复校验: 原 loss %.6g / 载入后 %.6g（差 %.3g）\n", loss_a, loss_b,
                    std::fabs((double) loss_a - (double) loss_b));

        // ---------- GGUF 导出 + 自检 + 回读 ----------
        GgufWriter* w = gguf_writer_new();
        gguf_writer_set_arch(w, "example.mlp");
        gguf_writer_set_u32(w, "example.in_features", (uint32_t) Session::IN);
        gguf_writer_set_u32(w, "example.hidden", (uint32_t) Session::HID);
        for (int64_t i = 0; i < param_list_count(&b.params); ++i) {
            Tensor* p = param_list_get(&b.params, i);
            gguf_writer_add_tensor(w, p->name, p, TYPE_F32);
        }
        if (!gguf_writer_write(w, model_path.c_str())) {
            std::fprintf(stderr, "GGUF 导出失败\n");
            gguf_writer_free(w);
            return 1;
        }
        gguf_writer_free(w);

        if (!gguf_validate(model_path.c_str())) {
            std::fprintf(stderr, "GGUF 自检失败\n");
            return 1;
        }

        GgufFile* f = gguf_open(model_path.c_str(), true);
        bool same = false;
        if (f != nullptr) {
            Tensor* p0 = param_list_get(&b.params, 0);
            const std::vector<float> ref = read_f32(p0);
            std::vector<float> got(ref.size(), 0.0f);
            same = gguf_tensor_to_f32(f, p0->name, got.data(), (int64_t) got.size());
            for (size_t i = 0; same && i < got.size(); ++i) {
                same = got[i] == ref[i];
            }
            gguf_close(f);
        }
        std::printf("GGUF 导出+回读: %s\n", same ? "逐位一致" : "不一致");
        if (!same) {
            return 1;
        }
    }

    std::printf("示例完成。产物：\n  %s\n  %s\n", ckpt_path.c_str(), model_path.c_str());
    return 0;
}
