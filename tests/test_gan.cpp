// train.cpp - 双图双优化器训练（M2.3e）：冻结/解冻 + detach 输入（通用 GAN 用例）
//
// 协议（R4：禁止多图共享参数/梯度路径）：
//   1) G 图：G(z) → D_frozen(fake) → LSGAN 损失；构建反向前冻结 D 参数（param_list_set_param(false)）
//   2) D 图：D(real) 与 D(fake) → LSGAN 损失；构建反向前冻结 G 参数 → D 图不回流 G 的梯度（detach）
//   3) 两图各自的 backward 构建完成后恢复标志；训练循环交替 graph_reset + compute + optim_step
// 断言：冻结参数无 grad/grad_acc；G 步只改 G 参数、D 步只改 D 参数；200 步对抗训练 loss 下降。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace traincpp;
using namespace trc_test;

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

bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

} // namespace

TRC_TEST(gan_dual_graph_training) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    constexpr int64_t ZDIM = 4;
    constexpr int64_t HID  = 16;
    constexpr int64_t B    = 8;

    Context* ctx = context_new(1 << 22);
    Rng      rng;
    rng_seed(&rng, 2026);

    Linear g1{}, g2{}, d1{}, d2{};
    linear_init(ctx, &g1, ZDIM, HID, true, &rng, "g1");
    linear_init(ctx, &g2, HID, 1, true, &rng, "g2");
    linear_init(ctx, &d1, 1, HID, true, &rng, "d1");
    linear_init(ctx, &d2, HID, 1, true, &rng, "d2");

    ParamList g_params, d_params;
    linear_params(&g1, &g_params);
    linear_params(&g2, &g_params);
    linear_params(&d1, &d_params);
    linear_params(&d2, &d_params);

    Tensor* z      = new_tensor_2d(ctx, TYPE_F32, ZDIM, B);
    Tensor* real   = new_tensor_2d(ctx, TYPE_F32, 1, B);
    Tensor* t_real = new_tensor_2d(ctx, TYPE_F32, 1, B);
    Tensor* t_fake = new_tensor_2d(ctx, TYPE_F32, 1, B);

    // ---------------- G 图：G(z) → D_frozen(fake) → LSGAN ----------------
    Graph*  gg     = graph_new(ctx);
    Tensor* fake   = linear_forward(ctx, &g2, relu(ctx, linear_forward(ctx, &g1, z)));
    Tensor* d_fake_g = linear_forward(ctx, &d2, relu(ctx, linear_forward(ctx, &d1, fake)));
    Tensor* g_loss = mse_loss(ctx, d_fake_g, t_real);
    tensor_set_loss(g_loss);

    param_list_set_param(&d_params, false);  // 冻结 D：G 图不建 D 的梯度
    graph_build_forward_expand(ctx, gg, g_loss);
    graph_build_backward_expand(gg);

    for (int64_t i = 0; i < param_list_count(&d_params); ++i) {
        Tensor* p = param_list_get(&d_params, i);
        TRC_EXPECT(p->grad == nullptr);
        TRC_EXPECT(p->grad_acc == nullptr);
    }
    std::vector<Tensor*> g_acc;
    for (int64_t i = 0; i < param_list_count(&g_params); ++i) {
        Tensor* p = param_list_get(&g_params, i);
        TRC_EXPECT(p->grad_acc != nullptr);
        g_acc.push_back(p->grad_acc);
    }

    // ---------------- D 图：D(real) / D(fake) → LSGAN ----------------
    Graph* dg = graph_new(ctx);
    param_list_set_param(&d_params, true);
    param_list_set_param(&g_params, false);  // 冻结 G：D 图不回流 G 的梯度（等价 detach）
    Tensor* d_real = linear_forward(ctx, &d2, relu(ctx, linear_forward(ctx, &d1, real)));
    Tensor* d_fake = linear_forward(ctx, &d2, relu(ctx, linear_forward(ctx, &d1, fake)));
    Tensor* d_loss = add(ctx, mse_loss(ctx, d_real, t_real), mse_loss(ctx, d_fake, t_fake));
    tensor_set_loss(d_loss);
    graph_build_forward_expand(ctx, dg, d_loss);
    graph_build_backward_expand(dg);
    param_list_set_param(&g_params, true);  // 恢复（只影响后续展开）

    // D 图构建未给 G 参数重分配累加器（仍指向 G 图的那些）
    for (int64_t i = 0; i < param_list_count(&g_params); ++i) {
        TRC_EXPECT(param_list_get(&g_params, i)->grad_acc == g_acc[(size_t) i]);
    }
    for (int64_t i = 0; i < param_list_count(&d_params); ++i) {
        TRC_EXPECT(param_list_get(&d_params, i)->grad_acc != nullptr);
    }

    // ---------------- 分配 + 优化器 + 数据 ----------------
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    AdamwOptions go;
    go.lr           = 0.02f;
    go.weight_decay = 0.0f;
    AdamwOptions do_;
    do_.lr           = 0.02f;
    do_.weight_decay = 0.0f;
    Optimizer* g_opt = optim_adamw_new(ctx, g_params.items.data(), param_list_count(&g_params), go);
    Optimizer* d_opt = optim_adamw_new(ctx, d_params.items.data(), param_list_count(&d_params), do_);

    uint32_t seed = 12345u;
    auto     rnd  = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> zv((size_t) (ZDIM * B));
    std::vector<float> rv((size_t) B);
    std::vector<float> tv((size_t) B, 1.0f);
    std::vector<float> fv((size_t) B, -1.0f);
    for (float& v : zv) {
        v = rnd();
    }
    for (float& v : rv) {
        v = 0.8f + 0.2f * rnd();
    }
    tensor_set(z, zv.data(), 0, zv.size() * sizeof(float));
    tensor_set(real, rv.data(), 0, rv.size() * sizeof(float));
    tensor_set(t_real, tv.data(), 0, tv.size() * sizeof(float));
    tensor_set(t_fake, fv.data(), 0, fv.size() * sizeof(float));

    // ---------------- 双优化器隔离性 ----------------
    const std::vector<float> d2_before = read_f32(d2.weight);
    const std::vector<float> g1_before = read_f32(g1.weight);

    graph_reset(gg);
    dev->graph_compute(gg);
    optim_step(g_opt);
    const std::vector<float> g1_after_g = read_f32(g1.weight);
    TRC_EXPECT(g1_after_g != g1_before);          // G 步更新了 G
    TRC_EXPECT(read_f32(d2.weight) == d2_before);  // 未触碰 D
    for (int64_t i = 0; i < param_list_count(&d_params); ++i) {
        // G 图不得向 D 的梯度累加器写入任何值（累加器由 graph_reset(gg) 清零）
        Tensor* p = param_list_get(&d_params, i);
        if (p->grad_acc != nullptr) {
            const std::vector<float> gacc = read_f32(p->grad_acc);
            for (float v : gacc) {
                TRC_EXPECT(v == 0.0f);
            }
        }
    }

    graph_reset(dg);
    dev->graph_compute(dg);
    TRC_EXPECT(read_f32(g1.weight) == g1_after_g);  // D 前向/反向不改 G 参数
    optim_step(d_opt);
    TRC_EXPECT(read_f32(d2.weight) != d2_before);   // D 步更新了 D
    TRC_EXPECT(read_f32(g1.weight) == g1_after_g);  // D 步不碰 G

    // ---------------- 对抗训练 ----------------
    graph_reset(gg);
    dev->graph_compute(gg);
    const float g0 = read_scalar(g_loss);
    graph_reset(dg);
    dev->graph_compute(dg);
    const float d0 = read_scalar(d_loss);

    constexpr int STEPS = 300;
    float         g_best = g0;
    for (int i = 0; i < STEPS; ++i) {
        graph_reset(gg);
        dev->graph_compute(gg);
        optim_step(g_opt);
        graph_reset(dg);
        dev->graph_compute(dg);
        optim_step(d_opt);

        graph_reset(gg);
        dev->graph_compute(gg);
        g_best = std::min(g_best, read_scalar(g_loss));
    }
    graph_reset(dg);
    dev->graph_compute(dg);
    const float d1v = read_scalar(d_loss);

    std::printf("    GAN（双图双 AdamW）: G loss %.4g → min %.4g（%d 步）；D loss %.4g → %.4g\n",
                (double) g0, (double) g_best, STEPS, (double) d0, (double) d1v);
    TRC_EXPECT(std::isfinite(g_best) && std::isfinite(d1v));
    TRC_EXPECT(g_best < 0.8f * g0);  // G 学会骗过 D（LSGAN 目标 → D(fake)=1）
    TRC_EXPECT(all_finite(read_f32(g1.weight)));
    TRC_EXPECT(all_finite(read_f32(d2.weight)));

    optim_free(g_opt);
    optim_free(d_opt);
    buffer_free(buf);
    graph_free(dg);
    graph_free(gg);
    context_free(ctx);
}
