// train.cpp - 二期验收（M2.4）：通用双网络对抗训练端到端（合成数据）
//
// 覆盖（§7.8 A/B）：
//   生成器 G：ConvTranspose1d(1→C) → LeakyReLU → ConvTranspose1d(C→1)（层内 N=B≥2 切片 + concat）
//   判别器 D：多周期三分支（周期 2/3/5），每支
//       WeightNorm 重参数化 Conv1d（w = g·v/‖v‖）+ LeakyReLU → grouped Conv1d(g=2) + LeakyReLU
//       三分支沿时间维 concat → Conv1d(1×1) 打分
//   反射填充：G 输出经 pad_ext(PAD_REFLECT) 后入 D（real 输入同长）
//   训练：双图双优化器（G 图冻结 D、D 图冻结 G；R4 协议）+ 双 AdamW + 双 CosineAnnealingLR
//   断点续训：checkpoint v2 双分区（G/D + 调度器 + RNG）→ 新会话载入续训 → 参数/损失/lr 逐位一致
//   GGUF：全部参数（含 weight_g/weight_v）导出 + gguf_validate + gguf_tensor_to_f32 回读逐位
//   断言：冻结参数无 grad/grad_acc；两优化器互不污染（逐位）；训练 loss 下降且参数有限
//
// 数据确定性：全部由下标公式生成（无随机训练数据），为断点续训"逐位等价"与 GGUF 回读提供基线。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
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

std::vector<std::vector<float>> snapshot(const ParamList& list) {
    std::vector<std::vector<float>> out;
    for (int64_t i = 0; i < param_list_count(&list); ++i) {
        out.push_back(read_f32(param_list_get(&list, i)));
    }
    return out;
}

// 逐位比较快照与当前参数（用于"优化器隔离/冻结不回流"断言）
bool snapshot_equal(const std::vector<std::vector<float>>& expected, const ParamList& list) {
    if (expected.size() != (size_t) param_list_count(&list)) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (read_f32(param_list_get(&list, (int64_t) i)) != expected[i]) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------- 网络结构常量

constexpr int64_t B      = 4;   // batch（验收要求 ≥2）
constexpr int64_t G_IN   = 8;   // 噪声长度
constexpr int64_t G_CH   = 8;   // 生成器中层通道
constexpr int64_t PAD_L  = 2;   // reflect 左填充
constexpr int64_t PAD_R  = 1;   // reflect 右填充
constexpr int64_t D_OC   = 4;   // 判别器分支通道
constexpr int64_t D_GRP  = 2;   // 判别器分组卷积 groups

// G 输出长度：((G_IN-1)*2-2*1+4) 再第二次转置卷积；reflect 填充后入 D
constexpr int64_t FAKE_LEN = ((G_IN - 1) * 2 - 2 + 4 - 1) * 2 - 2 + 4;  // 32
constexpr int64_t D_IN_LEN = FAKE_LEN + PAD_L + PAD_R;                  // 35

// 判别器分支：wn Conv1d(K=KS[i], pad=0) → grouped Conv1d(K=3, pad=0)，长度 = L-KS[i]+1-3+1
constexpr int64_t D_KS[3]  = {3, 5, 3};
constexpr int64_t D_LEN    = (D_IN_LEN - D_KS[0] + 1 - 3 + 1) + (D_IN_LEN - D_KS[1] + 1 - 3 + 1) +
                             (D_IN_LEN - D_KS[2] + 1 - 3 + 1);  // 91

// 确定性数据（纯下标公式；同一表达式在 CPU/Vulkan 主机构造，逐位一致）
float z_value(int64_t t, int64_t n) {
    return (float) ((t * 7 + n * 5) % 13) * 0.1f - 0.6f;
}

float real_value(int64_t t, int64_t n) {
    return (float) ((t * 3 + n * 7) % 11) * 0.12f - 0.55f;
}

// ---------------------------------------------------------------- 双图会话

struct Session {
    Device* dev = nullptr;
    Context* ctx = nullptr;
    Rng      rng;
    Graph*   gg = nullptr;  // G 图：G(z) → D_frozen(fake) → loss
    Graph*   dg = nullptr;  // D 图：D(real) / D(fake) → loss
    Buffer*  buf = nullptr;

    // 生成器
    ConvTranspose1d g1{}, g2{};
    // 判别器：每周期一支（weightnorm 重参数化 conv + grouped conv）
    WeightNorm wn[3]      = {};
    Tensor*    w_wn[3]    = {nullptr, nullptr, nullptr};
    Tensor*    d_bias[3]  = {nullptr, nullptr, nullptr};
    Conv1d     wnc[3]     = {};  // 仅承载 stride/padding/out_channels/bias，权重走 weightnorm_forward
    Conv1d     dgrp[3]    = {};
    Conv1d     dout{};

    ParamList g_params, d_params;

    Tensor* z      = nullptr;  // [G_IN,1,B]
    Tensor* real   = nullptr;  // [D_IN_LEN,1,B]
    Tensor* t_real = nullptr;  // [D_LEN,1,B]
    Tensor* t_fake = nullptr;  // [D_LEN,1,B]
    Tensor* g_loss = nullptr;
    Tensor* d_loss = nullptr;
    Tensor* x_in   = nullptr;  // reflect pad 后 D 输入（G 输出）
    Tensor* d_fake = nullptr;  // G 图中的 D(fake) 打分

    Optimizer*   g_opt   = nullptr;
    Optimizer*   d_opt   = nullptr;
    LrScheduler* g_sched = nullptr;
    LrScheduler* d_sched = nullptr;

    CheckpointOptimizer parts[2] = {};  // 长生命周期（CheckpointItems 持有裸指针）

    Session(Device* d, uint64_t seed) : dev(d) {
        ctx = context_new(1 << 22);
        rng_seed(&rng, seed);
        const float kaiming_a = std::sqrt(5.0f);

        // ---------- 生成器：ConvTranspose1d ×2（N>1 层内切片 + concat） ----------
        convtranspose1d_init(ctx, &g1, 1, G_CH, 4, 2, 1, 1, true, &rng, "gen.1");
        convtranspose1d_init(ctx, &g2, G_CH, 1, 4, 2, 1, 1, true, &rng, "gen.2");

        // ---------- 判别器：3 支多周期（weight_norm + grouped conv） ----------
        const char* period_names[3] = {"disc.p2", "disc.p3", "disc.p5"};
        const char* group_names[3]  = {"disc.g2", "disc.g3", "disc.g5"};
        for (int i = 0; i < 3; ++i) {
            Tensor* v = new_tensor_3d(ctx, TYPE_F32, D_KS[i], 1, D_OC);
            Tensor* g = new_tensor_3d(ctx, TYPE_F32, 1, 1, D_OC);
            weightnorm_init(ctx, &wn[i], v, g, period_names[i]);
            nn_init_kaiming_uniform(ctx, v, kaiming_a, &rng);
            nn_init_constant(ctx, g, 1.0f);  // 分配后 weightnorm_sync_g 覆盖为 ‖v‖

            d_bias[i] = new_tensor_1d(ctx, TYPE_F32, D_OC);
            tensor_set_name(d_bias[i], "%s.bias", period_names[i]);
            nn_init_uniform(ctx, d_bias[i], -0.2f, 0.2f, &rng);

            wnc[i].in_channels  = 1;
            wnc[i].out_channels = D_OC;
            wnc[i].kernel_size  = D_KS[i];
            wnc[i].stride       = 1;
            wnc[i].padding      = 0;
            wnc[i].dilation     = 1;
            wnc[i].groups       = 1;
            wnc[i].has_bias     = true;
            wnc[i].bias         = d_bias[i];

            // grouped conv：IC=OC=D_OC、groups=D_GRP（要求整除）
            conv1d_init_groups(ctx, &dgrp[i], D_OC, D_OC, 3, 1, 0, 1, D_GRP, true, &rng,
                               group_names[i]);
        }
        conv1d_init(ctx, &dout, D_OC, 1, 1, 1, 0, 1, true, &rng, "disc.out");

        // 重参数化权重（两支共享同一 w 节点：real/fake 前向各用一次，梯度自动累加）
        for (int i = 0; i < 3; ++i) {
            w_wn[i] = weightnorm_forward(ctx, &wn[i]);
        }

        z      = new_tensor_3d(ctx, TYPE_F32, G_IN, 1, B);
        real   = new_tensor_3d(ctx, TYPE_F32, D_IN_LEN, 1, B);
        t_real = new_tensor_3d(ctx, TYPE_F32, D_LEN, 1, B);
        t_fake = new_tensor_3d(ctx, TYPE_F32, D_LEN, 1, B);
        tensor_set_name(z, "z");
        tensor_set_name(real, "real");
        tensor_set_name(t_real, "target_real");
        tensor_set_name(t_fake, "target_fake");

        // ---------- 参数收集 ----------
        convtranspose1d_params(&g1, &g_params);
        convtranspose1d_params(&g2, &g_params);
        for (int i = 0; i < 3; ++i) {
            weightnorm_params(&wn[i], &d_params);
            param_list_add(&d_params, d_bias[i]);
            conv1d_params(&dgrp[i], &d_params);
        }
        conv1d_params(&dout, &d_params);

        // ---------- G 图：G(z) → reflect pad → D_frozen(fake) → LSGAN ----------
        Tensor* fake = convtranspose1d_forward(
            ctx, &g2, leaky_relu(ctx, convtranspose1d_forward(ctx, &g1, z), 0.1f));
        tensor_set_name(fake, "fake");
        x_in = pad_ext(ctx, fake, (int) PAD_L, (int) PAD_R, 0, 0, 0, 0, 0, 0, PAD_REFLECT);

        gg     = graph_new(ctx);
        d_fake = d_forward(x_in);
        g_loss = mse_loss(ctx, d_fake, t_real);
        tensor_set_loss(g_loss);

        param_list_set_param(&d_params, false);  // 冻结 D：G 图不建 D 的梯度
        graph_build_forward_expand(ctx, gg, g_loss);
        graph_build_backward_expand(gg);
    }

    // D 图（在 G 图反向构建之后调用；用于验证"冻结 D"确实成立）
    void build_d_graph() {
        dg = graph_new(ctx);
        param_list_set_param(&d_params, true);
        param_list_set_param(&g_params, false);  // 冻结 G：D 图不回流 G 的梯度（等价 detach）
        Tensor* d_real   = d_forward(real);
        Tensor* d_fake_d = d_forward(x_in);
        d_loss = add(ctx, mse_loss(ctx, d_real, t_real), mse_loss(ctx, d_fake_d, t_fake));
        tensor_set_loss(d_loss);
        graph_build_forward_expand(ctx, dg, d_loss);
        graph_build_backward_expand(dg);
        param_list_set_param(&g_params, true);  // 只影响后续展开
    }

    // 统一分配 + 主机端同步 ‖v‖ + 双优化器/双调度器
    void alloc_and_setup() {
        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
        for (int i = 0; i < 3; ++i) {
            weightnorm_sync_g(&wn[i]);
        }

        AdamwOptions go;
        go.lr           = 0.01f;
        go.weight_decay = 0.0f;
        AdamwOptions do_;
        do_.lr           = 0.01f;
        do_.weight_decay = 0.0f;
        g_opt   = optim_adamw_new(ctx, g_params.items.data(), param_list_count(&g_params), go);
        d_opt   = optim_adamw_new(ctx, d_params.items.data(), param_list_count(&d_params), do_);
        g_sched = lr_scheduler_cosine_new(g_opt, 150, 0.0f);
        d_sched = lr_scheduler_cosine_new(d_opt, 300, 0.0f);
    }

    // 优化器状态分配（checkpoint 载入前必须）
    void alloc_state() {
        optim_alloc_state(g_opt);
        optim_alloc_state(d_opt);
    }

    // checkpoint v2 双优化器分区（parts 为成员，返回的 items 生命周期安全）
    CheckpointItems items() {
        parts[0] = CheckpointOptimizer{"G", g_opt, g_sched};
        parts[1] = CheckpointOptimizer{"D", d_opt, d_sched};
        CheckpointItems it{};
        it.rng             = &rng;
        it.optimizers      = parts;
        it.optimizer_count = 2;
        return it;
    }

    ~Session() {
        if (d_sched != nullptr) {
            lr_scheduler_free(d_sched);
        }
        if (g_sched != nullptr) {
            lr_scheduler_free(g_sched);
        }
        if (d_opt != nullptr) {
            optim_free(d_opt);
        }
        if (g_opt != nullptr) {
            optim_free(g_opt);
        }
        buffer_free(buf);
        graph_free(dg);
        graph_free(gg);
        context_free(ctx);
    }

    // D 前向：3 支多周期 concat → 1×1 打分；x 为 [D_IN_LEN,1,B]
    Tensor* d_forward(Tensor* x) {
        Tensor* y = nullptr;
        for (int i = 0; i < 3; ++i) {
            Tensor* h = conv1d_forward_weight(ctx, &wnc[i], w_wn[i], x);
            h         = leaky_relu(ctx, h, 0.1f);
            h         = conv1d_forward(ctx, &dgrp[i], h);
            h         = leaky_relu(ctx, h, 0.1f);
            y         = (y == nullptr) ? h : concat(ctx, y, h, 0);
        }
        return conv1d_forward(ctx, &dout, y);
    }

    // 确定性数据写入（必须先 buffer_alloc）
    void write_data() {
        std::vector<float> vz((size_t) (G_IN * B));
        for (int64_t t = 0; t < G_IN; ++t) {
            for (int64_t n = 0; n < B; ++n) {
                vz[(size_t) (t + n * G_IN)] = z_value(t, n);
            }
        }
        std::vector<float> vr((size_t) (D_IN_LEN * B));
        for (int64_t t = 0; t < D_IN_LEN; ++t) {
            for (int64_t n = 0; n < B; ++n) {
                vr[(size_t) (t + n * D_IN_LEN)] = real_value(t, n);
            }
        }
        const int64_t n_targets = D_LEN * B;
        std::vector<float> vtr((size_t) n_targets, 1.0f);
        std::vector<float> vtf((size_t) n_targets, -1.0f);
        tensor_set(z, vz.data(), 0, vz.size() * sizeof(float));
        tensor_set(real, vr.data(), 0, vr.size() * sizeof(float));
        tensor_set(t_real, vtr.data(), 0, vtr.size() * sizeof(float));
        tensor_set(t_fake, vtf.data(), 0, vtf.size() * sizeof(float));
    }

    float eval_g() {
        graph_reset(gg);
        dev->graph_compute(gg);
        return read_scalar(g_loss);
    }

    float eval_d() {
        graph_reset(dg);
        dev->graph_compute(dg);
        return read_scalar(d_loss);
    }

    void g_step() {
        graph_reset(gg);
        dev->graph_compute(gg);
        optim_step(g_opt);
        lr_scheduler_step(g_sched);
    }

    void d_step() {
        graph_reset(dg);
        dev->graph_compute(dg);
        optim_step(d_opt);
        lr_scheduler_step(d_sched);
    }
};

} // namespace

// ---------------------------------------------------------------- 验收用例

TRC_TEST(acceptance_dual_network_e2e) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Session s(dev, 20260922);

    std::printf("    验收网络：G(ConvTranspose1d×2) + D(多周期 %lld 支, weight_norm+grouped conv) | "
                "batch=%lld, fake_len=%lld→pad %lld→D_in=%lld, D_out=%lld\n",
                (long long) 3, (long long) B, (long long) FAKE_LEN, (long long) PAD_L,
                (long long) D_IN_LEN, (long long) D_LEN);

    // ---------- 冻结协议（G 图反向已建、D 图尚未建）：D 参数无梯度、G 参数有累加器 ----------
    for (int64_t i = 0; i < param_list_count(&s.d_params); ++i) {
        Tensor* p = param_list_get(&s.d_params, i);
        TRC_EXPECT(p->grad == nullptr);
        TRC_EXPECT(p->grad_acc == nullptr);
    }
    std::vector<Tensor*> g_acc;
    for (int64_t i = 0; i < param_list_count(&s.g_params); ++i) {
        Tensor* p = param_list_get(&s.g_params, i);
        TRC_EXPECT(p->grad_acc != nullptr);
        g_acc.push_back(p->grad_acc);
    }

    // ---------- D 图：D(real)/D(fake) 反向；验证未给 G 重分配累加器、未回流 G ----------
    s.build_d_graph();
    for (int64_t i = 0; i < param_list_count(&s.g_params); ++i) {
        TRC_EXPECT(param_list_get(&s.g_params, i)->grad_acc == g_acc[(size_t) i]);
    }
    for (int64_t i = 0; i < param_list_count(&s.d_params); ++i) {
        TRC_EXPECT(param_list_get(&s.d_params, i)->grad_acc != nullptr);
    }
    s.alloc_and_setup();
    s.write_data();

    // ---------- 双优化器隔离性（逐位） ----------
    const auto d_before = snapshot(s.d_params);
    const auto g_before = snapshot(s.g_params);
    s.g_step();
    const auto g_after_g = snapshot(s.g_params);
    TRC_EXPECT(g_after_g != g_before);            // G 步更新了 G
    TRC_EXPECT(snapshot_equal(d_before, s.d_params));  // 未触碰 D
    // G 图不得向 D 的梯度累加器写入任何值
    for (int64_t i = 0; i < param_list_count(&s.d_params); ++i) {
        Tensor* p = param_list_get(&s.d_params, i);
        if (p->grad_acc != nullptr) {
            for (float v : read_f32(p->grad_acc)) {
                TRC_EXPECT(v == 0.0f);
            }
        }
    }
    s.d_step();
    TRC_EXPECT(snapshot_equal(g_after_g, s.g_params));  // D 步不碰 G
    TRC_EXPECT(!snapshot_equal(d_before, s.d_params));  // D 步更新了 D

    // ---------- 对抗训练（第一阶段） ----------
    const float g0 = s.eval_g();
    const float d0 = s.eval_d();
    constexpr int STEPS_A = 100;
    constexpr int STEPS_B = 100;  // 存档后两会话各续训的步数
    float g_best = g0;
    for (int i = 0; i < STEPS_A; ++i) {
        s.g_step();
        s.d_step();
        g_best = std::min(g_best, s.eval_g());
    }
    const float d_mid = s.eval_d();
    std::printf("    对抗训练：G loss %.4g → min %.4g（%d 步）；D loss %.4g → %.4g\n", (double) g0,
                (double) g_best, STEPS_A, (double) d0, (double) d_mid);

    // ---------- 断点续训：存档（双优化器 + 双调度器 + RNG） ----------
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::string ckpt = (dir / "traincpp_acceptance_gan.ckpt.gguf").string();
    if (!checkpoint_save(ckpt.c_str(), s.items(), "acceptance.gan")) {
        trc_test::add_failure("acceptance: checkpoint 保存失败");
        return;
    }
    CheckpointInfo info;
    TRC_EXPECT(checkpoint_probe(ckpt.c_str(), &info));
    TRC_EXPECT_EQ(info.version, 2);
    TRC_EXPECT_EQ(info.optimizer_count, 2);
    TRC_EXPECT(checkpoint_verify(ckpt.c_str(), s.items()));
    std::printf("    checkpoint：schema v%d 分区=%lld step=%lld 参数=%lld\n", (int) info.version,
                (long long) info.optimizer_count, (long long) info.step,
                (long long) info.param_count);

    // 原会话继续训练 STEPS_B 步（不中断基线）
    for (int i = 0; i < STEPS_B; ++i) {
        s.g_step();
        s.d_step();
    }
    const auto  g_final  = snapshot(s.g_params);
    const auto  d_final  = snapshot(s.d_params);
    const float g1       = s.eval_g();
    const float d1       = s.eval_d();
    TRC_EXPECT(optim_step_count(s.g_opt) == optim_step_count(s.d_opt));

    // ---------- 新会话：载入 → 续训 STEPS_B 步 → 参数/损失逐位一致 ----------
    {
        Session b(dev, 777);  // 不同初始化种子：载入后应完全被存档覆盖
        b.build_d_graph();
        b.alloc_and_setup();
        b.alloc_state();
        if (!checkpoint_load(ckpt.c_str(), b.items())) {
            trc_test::add_failure("acceptance: checkpoint 载入失败");
            return;
        }
        b.write_data();
        for (int i = 0; i < STEPS_B; ++i) {
            b.g_step();
            b.d_step();
        }
        TRC_EXPECT(snapshot_equal(g_final, b.g_params));  // 续训后逐位相等
        TRC_EXPECT(snapshot_equal(d_final, b.d_params));
        TRC_EXPECT(b.eval_g() == g1);
        TRC_EXPECT(b.eval_d() == d1);
        TRC_EXPECT(optim_step_count(s.g_opt) == optim_step_count(b.g_opt));
        TRC_EXPECT(optim_step_count(s.d_opt) == optim_step_count(b.d_opt));
        TRC_EXPECT(optim_get_lr(s.g_opt) == optim_get_lr(b.g_opt));
        TRC_EXPECT(optim_get_lr(s.d_opt) == optim_get_lr(b.d_opt));
        std::printf("    断点续训：新会话载入 + %d 步后参数逐位一致；G/D loss %.6g / %.6g\n",
                    STEPS_B, (double) g1, (double) d1);
    }

    // ---------- 训练有效性与数值健康 ----------
    TRC_EXPECT(std::isfinite(g_best) && std::isfinite(d1));
    TRC_EXPECT(g_best < g0);  // G 学会骗过 D（LSGAN 目标 → D(fake)=1）
    for (int64_t i = 0; i < param_list_count(&s.g_params); ++i) {
        TRC_EXPECT(all_finite(read_f32(param_list_get(&s.g_params, i))));
    }
    for (int64_t i = 0; i < param_list_count(&s.d_params); ++i) {
        TRC_EXPECT(all_finite(read_f32(param_list_get(&s.d_params, i))));
    }

    // ---------- GGUF 导出 + 自检 + 回读（ggml 生态互操作；weight_g/weight_v 原样导出） ----------
    const std::string model_path = (dir / "traincpp_acceptance_model.gguf").string();
    const int64_t     n_params   = param_list_count(&s.g_params) + param_list_count(&s.d_params);
    {
        GgufWriter* w = gguf_writer_new();
        gguf_writer_set_arch(w, "acceptance.gan");
        gguf_writer_set_u32(w, "acceptance.batch", (uint32_t) B);
        gguf_writer_set_u32(w, "acceptance.d_in_len", (uint32_t) D_IN_LEN);
        gguf_writer_set_u32(w, "acceptance.d_out_len", (uint32_t) D_LEN);
        for (int64_t i = 0; i < param_list_count(&s.g_params); ++i) {
            Tensor* p = param_list_get(&s.g_params, i);
            gguf_writer_add_tensor(w, p->name, p, TYPE_F32);
        }
        for (int64_t i = 0; i < param_list_count(&s.d_params); ++i) {
            Tensor* p = param_list_get(&s.d_params, i);
            gguf_writer_add_tensor(w, p->name, p, TYPE_F32);
        }
        TRC_EXPECT_EQ(gguf_writer_tensor_count(w), n_params);
        const bool wrote = gguf_writer_write(w, model_path.c_str());
        gguf_writer_free(w);
        if (!wrote) {
            trc_test::add_failure("acceptance: GGUF 导出失败");
            return;
        }
    }
    TRC_EXPECT(gguf_validate(model_path.c_str()));
    {
        GgufFile* f = gguf_open(model_path.c_str(), true);
        TRC_EXPECT(f != nullptr);
        if (f != nullptr) {
            TRC_EXPECT_EQ(gguf_tensor_count(f), n_params);
            bool all_same = true;
            for (int pass = 0; pass < 2; ++pass) {
                const ParamList& list = (pass == 0) ? s.g_params : s.d_params;
                for (int64_t i = 0; i < param_list_count(&list); ++i) {
                    Tensor*                  p   = param_list_get(&list, i);
                    const std::vector<float> ref = read_f32(p);
                    std::vector<float>       got(ref.size(), 0.0f);
                    bool same = gguf_tensor_to_f32(f, p->name, got.data(), (int64_t) got.size());
                    for (size_t k = 0; same && k < got.size(); ++k) {
                        same = (got[k] == ref[k]);
                    }
                    all_same = all_same && same;
                }
            }
            gguf_close(f);
            TRC_EXPECT(all_same);
        }
    }
    std::printf("    GGUF：导出 %lld 张量 + 自检 + 回读逐位一致（%s）\n", (long long) n_params,
                model_path.c_str());
}
