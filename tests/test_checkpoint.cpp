// train.cpp - checkpoint 测试（M1.6b）
//
// 覆盖：
//   1. save/probe/load round-trip：参数与优化器状态逐位一致，step/RNG/调度器 epoch 恢复
//   2. 断点续训等价性：中断 + 恢复后的训练轨迹 == 不中断轨迹（AdamW + Cosine / SGD momentum）
//   3. 错误路径：缺文件、非 checkpoint 文件、参数数量/优化器类型/超参不匹配
#include "test_util.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace traincpp;

namespace {

std::string ckpt_tmp_path(const char* name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    return (dir / (std::string("trc_checkpoint_test_") + name + ".gguf")).string();
}

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

void fill_random(Tensor* t, Rng* rng) {
    std::vector<float> v((size_t) tensor_nelements(t));
    for (float& x : v) {
        x = rng_uniform(rng, -0.5f, 0.5f);
    }
    tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

void expect_same(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    if (a.size() != b.size()) {
        TRC_EXPECT(false && what);
        return;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            TRC_EXPECT(false && what);
            return;
        }
    }
}

// 逐位相等（用于续训等价断言；expect_same 已是精确比较，这里显式命名语义）
void expect_exact(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    expect_same(a, b, what);
}

// 固定小 MLP：x[IN,BATCH] -> Linear -> relu -> Linear -> mse（训练一次 = 前向+反向+优化器一步）
struct TrainSession {
    static constexpr int64_t IN    = 4;
    static constexpr int64_t HID   = 8;
    static constexpr int64_t OUT   = 2;
    static constexpr int64_t BATCH = 6;

    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Graph*   g   = nullptr;
    Buffer*  buf = nullptr;
    Rng      rng;

    Tensor* w1 = nullptr;
    Tensor* b1 = nullptr;
    Tensor* w2 = nullptr;
    Tensor* b2 = nullptr;
    Tensor* x  = nullptr;
    Tensor* t  = nullptr;
    Tensor* loss = nullptr;

    Optimizer*   opt   = nullptr;
    LrScheduler* sched = nullptr;

    TrainSession(Device* d, uint64_t seed, bool use_sgd, bool use_sched) : dev(d) {
        ctx = context_new(1 << 22);
        g   = graph_new(ctx);
        rng_seed(&rng, seed);

        w1 = new_tensor_2d(ctx, TYPE_F32, IN, HID);
        b1 = new_tensor_1d(ctx, TYPE_F32, HID);
        w2 = new_tensor_2d(ctx, TYPE_F32, HID, OUT);
        b2 = new_tensor_1d(ctx, TYPE_F32, OUT);
        tensor_set_name(w1, "w1");
        tensor_set_name(b1, "b1");
        tensor_set_name(w2, "w2");
        tensor_set_name(b2, "b2");
        tensor_set_param(w1);
        tensor_set_param(b1);
        tensor_set_param(w2);
        tensor_set_param(b2);

        x = new_tensor_2d(ctx, TYPE_F32, IN, BATCH);
        t = new_tensor_2d(ctx, TYPE_F32, OUT, BATCH);
        tensor_set_name(x, "x");
        tensor_set_name(t, "target");

        Tensor* h    = relu(ctx, add(ctx, mul_mat(ctx, w1, x), b1));
        Tensor* pred = add(ctx, mul_mat(ctx, w2, h), b2);
        loss = mse_loss(ctx, pred, t);
        tensor_set_loss(loss);

        graph_build_forward_expand(ctx, g, loss);
        graph_build_backward_expand(g);
        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

        Tensor* params[4] = {w1, b1, w2, b2};
        if (use_sgd) {
            SgdOptions o;
            o.lr       = 0.05f;
            o.momentum = 0.9f;
            opt = optim_sgd_new(ctx, params, 4, o);
        } else {
            AdamwOptions o;
            o.lr = 0.01f;
            opt  = optim_adamw_new(ctx, params, 4, o);
        }
        if (use_sched) {
            sched = lr_scheduler_cosine_new(opt, 50, 0.0f);
        }

        fill_random(w1, &rng);
        fill_random(b1, &rng);
        fill_random(w2, &rng);
        fill_random(b2, &rng);

        std::vector<float> vx((size_t) (IN * BATCH));
        std::vector<float> vt((size_t) (OUT * BATCH));
        for (size_t i = 0; i < vx.size(); ++i) {
            vx[i] = (float) (i * 7 % 11) * 0.1f - 0.5f;
        }
        for (size_t i = 0; i < vt.size(); ++i) {
            vt[i] = (float) (i * 5 % 7) * 0.25f - 0.5f;
        }
        tensor_set(x, vx.data(), 0, vx.size() * sizeof(float));
        tensor_set(t, vt.data(), 0, vt.size() * sizeof(float));
    }

    ~TrainSession() {
        if (sched != nullptr) {
            lr_scheduler_free(sched);
        }
        optim_free(opt);
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }

    void step() {
        graph_reset(g);
        dev->graph_compute(g);
        optim_step(opt);
        if (sched != nullptr) {
            lr_scheduler_step(sched);
        }
    }
};

// 双网络双优化器/双调度器（M2.3f）：G/D 各自独立的图与优化器；数据确定性、不共享参数
struct DualSession {
    static constexpr int64_t IN    = 3;
    static constexpr int64_t HID   = 6;
    static constexpr int64_t OUT   = 2;
    static constexpr int64_t BATCH = 4;

    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Graph*   gg  = nullptr;
    Graph*   dg  = nullptr;
    Buffer*  buf = nullptr;
    Rng      rng;

    Linear    g1{}, g2{}, d1{}, d2{};
    ParamList g_params, d_params;

    Tensor* z      = nullptr;
    Tensor* tg     = nullptr;
    Tensor* td     = nullptr;
    Tensor* g_loss = nullptr;
    Tensor* d_loss = nullptr;

    Optimizer*   g_opt   = nullptr;
    Optimizer*   d_opt   = nullptr;
    LrScheduler* g_sched = nullptr;
    LrScheduler* d_sched = nullptr;

    CheckpointOptimizer parts[2] = {};

    DualSession(Device* d, uint64_t seed) : dev(d) {
        ctx = context_new(1 << 22);
        rng_seed(&rng, seed);

        linear_init(ctx, &g1, IN, HID, true, &rng, "g1");
        linear_init(ctx, &g2, HID, OUT, true, &rng, "g2");
        linear_init(ctx, &d1, IN, HID, true, &rng, "d1");
        linear_init(ctx, &d2, HID, OUT, true, &rng, "d2");
        linear_params(&g1, &g_params);
        linear_params(&g2, &g_params);
        linear_params(&d1, &d_params);
        linear_params(&d2, &d_params);

        z  = new_tensor_2d(ctx, TYPE_F32, IN, BATCH);
        tg = new_tensor_2d(ctx, TYPE_F32, OUT, BATCH);
        td = new_tensor_2d(ctx, TYPE_F32, OUT, BATCH);
        tensor_set_name(z, "z");
        tensor_set_name(tg, "tg");
        tensor_set_name(td, "td");

        gg     = graph_new(ctx);
        g_loss = mse_loss(ctx, linear_forward(ctx, &g2, relu(ctx, linear_forward(ctx, &g1, z))), tg);
        tensor_set_loss(g_loss);
        graph_build_forward_expand(ctx, gg, g_loss);
        graph_build_backward_expand(gg);

        dg     = graph_new(ctx);
        d_loss = mse_loss(ctx, linear_forward(ctx, &d2, relu(ctx, linear_forward(ctx, &d1, z))), td);
        tensor_set_loss(d_loss);
        graph_build_forward_expand(ctx, dg, d_loss);
        graph_build_backward_expand(dg);

        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

        AdamwOptions go;
        go.lr           = 0.02f;
        go.weight_decay = 0.0f;
        AdamwOptions do_;
        do_.lr           = 0.01f;
        do_.weight_decay = 0.0f;
        g_opt   = optim_adamw_new(ctx, g_params.items.data(), param_list_count(&g_params), go);
        d_opt   = optim_adamw_new(ctx, d_params.items.data(), param_list_count(&d_params), do_);
        g_sched = lr_scheduler_cosine_new(g_opt, 20, 0.0f);
        d_sched = lr_scheduler_cosine_new(d_opt, 40, 0.0f);

        // 确定性数据（与构造种子无关，保证两个会话输入一致）
        std::vector<float> vz((size_t) (IN * BATCH));
        std::vector<float> vg((size_t) (OUT * BATCH));
        std::vector<float> vd((size_t) (OUT * BATCH));
        for (size_t i = 0; i < vz.size(); ++i) {
            vz[i] = (float) (i * 7 % 13) * 0.1f - 0.5f;
        }
        for (size_t i = 0; i < vg.size(); ++i) {
            vg[i] = (float) (i * 3 % 5) * 0.25f - 0.5f;
        }
        for (size_t i = 0; i < vd.size(); ++i) {
            vd[i] = (float) (i * 5 % 7) * 0.2f - 0.4f;
        }
        tensor_set(z, vz.data(), 0, vz.size() * sizeof(float));
        tensor_set(tg, vg.data(), 0, vg.size() * sizeof(float));
        tensor_set(td, vd.data(), 0, vd.size() * sizeof(float));
    }

    ~DualSession() {
        lr_scheduler_free(d_sched);
        lr_scheduler_free(g_sched);
        optim_free(d_opt);
        optim_free(g_opt);
        buffer_free(buf);
        graph_free(dg);
        graph_free(gg);
        context_free(ctx);
    }

    void alloc_state() {
        optim_alloc_state(g_opt);
        optim_alloc_state(d_opt);
    }

    void step() {
        graph_reset(gg);
        dev->graph_compute(gg);
        optim_step(g_opt);
        lr_scheduler_step(g_sched);

        graph_reset(dg);
        dev->graph_compute(dg);
        optim_step(d_opt);
        lr_scheduler_step(d_sched);
    }

    CheckpointItems items() {
        parts[0] = CheckpointOptimizer{"G", g_opt, g_sched};
        parts[1] = CheckpointOptimizer{"D", d_opt, d_sched};
        CheckpointItems it{};
        it.rng             = &rng;
        it.optimizers      = parts;
        it.optimizer_count = 2;
        return it;
    }
};

} // namespace

// ---------------------------------------------------------------- round-trip

TRC_TEST(checkpoint_roundtrip) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("roundtrip");
    std::remove(path.c_str());

    TrainSession a(dev, 1234, /*use_sgd=*/false, /*use_sched=*/true);
    for (int i = 0; i < 8; ++i) {
        a.step();
    }

    const std::vector<float> w1a = read_f32(a.w1);
    const std::vector<float> b1a = read_f32(a.b1);
    const std::vector<float> w2a = read_f32(a.w2);
    const std::vector<float> b2a = read_f32(a.b2);
    const uint64_t rng_a   = a.rng.state;
    const int64_t  step_a  = optim_step_count(a.opt);
    const int64_t  epoch_a = lr_scheduler_epoch(a.sched);
    const float    lr_a    = optim_get_lr(a.opt);

    TRC_EXPECT(checkpoint_save(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}, "test.arch"));

    CheckpointInfo info;
    TRC_EXPECT(checkpoint_probe(path.c_str(), &info));
    TRC_EXPECT_EQ(info.version, 2);
    TRC_EXPECT_EQ(info.optimizer_count, 1);
    TRC_EXPECT_EQ(info.step, 8);
    TRC_EXPECT_EQ(info.param_count, 4);
    TRC_EXPECT_EQ(info.group_count, 1);
    TRC_EXPECT(std::string(info.optimizer_type) == "adamw");
    TRC_EXPECT(std::string(info.model_arch) == "test.arch");
    TRC_EXPECT(info.has_rng && info.has_scheduler);
    TRC_EXPECT_EQ(info.scheduler_epoch, 8);

    // 新会话用不同初始化；载入后应逐位还原
    TrainSession b(dev, 999, false, true);
    optim_alloc_state(b.opt);
    TRC_EXPECT(checkpoint_load(path.c_str(), CheckpointItems{b.opt, &b.rng, b.sched}));
    expect_same(w1a, read_f32(b.w1), "w1 载入不一致");
    expect_same(b1a, read_f32(b.b1), "b1 载入不一致");
    expect_same(w2a, read_f32(b.w2), "w2 载入不一致");
    expect_same(b2a, read_f32(b.b2), "b2 载入不一致");

    TRC_EXPECT_EQ(optim_state_count(a.opt), 8);  // AdamW：4 参数 × 2
    for (int64_t i = 0; i < optim_state_count(a.opt); ++i) {
        expect_same(read_f32(optim_state(a.opt, i)), read_f32(optim_state(b.opt, i)),
                    "优化器状态载入不一致");
    }
    TRC_EXPECT_EQ(optim_step_count(b.opt), step_a);
    TRC_EXPECT(b.rng.state == rng_a);
    TRC_EXPECT_EQ(lr_scheduler_epoch(b.sched), epoch_a);
    TRC_EXPECT(optim_get_lr(b.opt) == lr_a);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------- 断点续训等价性

TRC_TEST(checkpoint_resume_trajectory) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("resume");
    std::remove(path.c_str());

    TrainSession a(dev, 7, false, true);
    for (int i = 0; i < 10; ++i) {
        a.step();
    }
    TRC_EXPECT(checkpoint_save(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}, nullptr));
    for (int i = 0; i < 10; ++i) {
        a.step();
    }
    const std::vector<float> w1a = read_f32(a.w1);
    const std::vector<float> b2a = read_f32(a.b2);
    const float loss_a = read_scalar(a.loss);

    TrainSession b(dev, 42, false, true);
    optim_alloc_state(b.opt);
    TRC_EXPECT(checkpoint_load(path.c_str(), CheckpointItems{b.opt, &b.rng, b.sched}));
    for (int i = 0; i < 10; ++i) {
        b.step();
    }
    const std::vector<float> w1b = read_f32(b.w1);
    const std::vector<float> b2b = read_f32(b.b2);
    const float loss_b = read_scalar(b.loss);

    for (size_t i = 0; i < w1a.size(); ++i) {
        TRC_EXPECT_NEAR(w1b[i], w1a[i], 1e-5);
    }
    for (size_t i = 0; i < b2a.size(); ++i) {
        TRC_EXPECT_NEAR(b2b[i], b2a[i], 1e-5);
    }
    TRC_EXPECT_NEAR(loss_b, loss_a, 1e-6);
    TRC_EXPECT_EQ(optim_step_count(b.opt), optim_step_count(a.opt));
    TRC_EXPECT(optim_get_lr(b.opt) == optim_get_lr(a.opt));
    std::printf("    断点续训：训练 20 步 loss %.6g == 中断 10 + 恢复 10 步 loss %.6g\n", loss_a, loss_b);

    std::remove(path.c_str());
}

TRC_TEST(checkpoint_sgd_momentum) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    // 分组信息访问器
    const std::string path = ckpt_tmp_path("sgd");
    std::remove(path.c_str());

    TrainSession a(dev, 21, true, false);
    TRC_EXPECT(optim_group_type(a.opt, 0) == OptimType::SGD);
    TRC_EXPECT_EQ(optim_group_param_count(a.opt, 0), 4);
    TRC_EXPECT_EQ(optim_group_state_count(a.opt, 0), 4);  // SGD：每参数 1 个 momentum
    TRC_EXPECT_NEAR(optim_get_lr_group(a.opt, 0), 0.05, 1e-7);
    TRC_EXPECT(optim_group_sgd_options(a.opt, 0).momentum == 0.9f);

    for (int i = 0; i < 6; ++i) {
        a.step();
    }
    TRC_EXPECT(checkpoint_save(path.c_str(), CheckpointItems{a.opt, nullptr, nullptr}, "sgd.test"));
    for (int i = 0; i < 6; ++i) {
        a.step();
    }
    const std::vector<float> w2a = read_f32(a.w2);

    TrainSession b(dev, 77, true, false);
    optim_alloc_state(b.opt);
    TRC_EXPECT(checkpoint_load(path.c_str(), CheckpointItems{b.opt, nullptr, nullptr}));
    TRC_EXPECT(optim_group_sgd_options(b.opt, 0).momentum == 0.9f);
    CheckpointInfo info;
    TRC_EXPECT(checkpoint_probe(path.c_str(), &info));
    TRC_EXPECT(std::string(info.optimizer_type) == "sgd");
    TRC_EXPECT(std::string(info.model_arch) == "sgd.test");
    TRC_EXPECT(!info.has_rng && !info.has_scheduler);
    for (int i = 0; i < 6; ++i) {
        b.step();
    }
    const std::vector<float> w2b = read_f32(b.w2);
    for (size_t i = 0; i < w2a.size(); ++i) {
        TRC_EXPECT_NEAR(w2b[i], w2a[i], 1e-5);
    }

    std::remove(path.c_str());
}

// ---------------------------------------------------------------- 多优化器（schema v2）

TRC_TEST(checkpoint_multi_roundtrip) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("multi_roundtrip");
    std::remove(path.c_str());

    DualSession a(dev, 2026);
    a.alloc_state();
    for (int i = 0; i < 6; ++i) {
        a.step();
    }

    // 记录全部参数/状态/元数据
    std::vector<std::vector<float>> params_a;
    std::vector<std::vector<float>> states_a;
    for (int64_t i = 0; i < optim_param_count(a.g_opt); ++i) {
        params_a.push_back(read_f32(optim_param(a.g_opt, i)));
    }
    for (int64_t i = 0; i < optim_param_count(a.d_opt); ++i) {
        params_a.push_back(read_f32(optim_param(a.d_opt, i)));
    }
    for (int64_t i = 0; i < optim_state_count(a.g_opt); ++i) {
        states_a.push_back(read_f32(optim_state(a.g_opt, i)));
    }
    for (int64_t i = 0; i < optim_state_count(a.d_opt); ++i) {
        states_a.push_back(read_f32(optim_state(a.d_opt, i)));
    }

    CheckpointItems ia = a.items();
    TRC_EXPECT(checkpoint_save(path.c_str(), ia, "dual.test"));

    CheckpointInfo info;
    TRC_EXPECT(checkpoint_probe(path.c_str(), &info));
    TRC_EXPECT_EQ(info.version, 2);
    TRC_EXPECT_EQ(info.optimizer_count, 2);
    TRC_EXPECT_EQ(info.param_count, 4);  // 第 0 个分区（G）
    TRC_EXPECT_EQ(info.group_count, 1);
    TRC_EXPECT(std::string(info.optimizer_type) == "adamw");
    TRC_EXPECT(std::string(info.model_arch) == "dual.test");
    TRC_EXPECT(info.has_rng && info.has_scheduler);
    TRC_EXPECT_EQ(info.scheduler_epoch, 6);

    // 新会话（不同初始化）载入后全部逐位还原
    DualSession b(dev, 999);
    b.alloc_state();
    CheckpointItems ib = b.items();
    TRC_EXPECT(checkpoint_load(path.c_str(), ib));

    size_t kp = 0;
    for (int64_t i = 0; i < optim_param_count(b.g_opt); ++i) {
        expect_same(params_a[kp++], read_f32(optim_param(b.g_opt, i)), "G 参数载入不一致");
    }
    for (int64_t i = 0; i < optim_param_count(b.d_opt); ++i) {
        expect_same(params_a[kp++], read_f32(optim_param(b.d_opt, i)), "D 参数载入不一致");
    }
    size_t ks = 0;
    for (int64_t i = 0; i < optim_state_count(b.g_opt); ++i) {
        expect_same(states_a[ks++], read_f32(optim_state(b.g_opt, i)), "G 状态载入不一致");
    }
    for (int64_t i = 0; i < optim_state_count(b.d_opt); ++i) {
        expect_same(states_a[ks++], read_f32(optim_state(b.d_opt, i)), "D 状态载入不一致");
    }

    TRC_EXPECT_EQ(optim_step_count(b.g_opt), optim_step_count(a.g_opt));
    TRC_EXPECT_EQ(optim_step_count(b.d_opt), optim_step_count(a.d_opt));
    for (int64_t i = 0; i < optim_param_count(b.g_opt); ++i) {
        TRC_EXPECT_EQ(optim_state_step(b.g_opt, i), optim_state_step(a.g_opt, i));
    }
    for (int64_t i = 0; i < optim_param_count(b.d_opt); ++i) {
        TRC_EXPECT_EQ(optim_state_step(b.d_opt, i), optim_state_step(a.d_opt, i));
    }
    TRC_EXPECT_EQ(lr_scheduler_epoch(b.g_sched), lr_scheduler_epoch(a.g_sched));
    TRC_EXPECT_EQ(lr_scheduler_epoch(b.d_sched), lr_scheduler_epoch(a.d_sched));
    TRC_EXPECT(optim_get_lr_group(b.g_opt, 0) == optim_get_lr_group(a.g_opt, 0));
    TRC_EXPECT(optim_get_lr_group(b.d_opt, 0) == optim_get_lr_group(a.d_opt, 0));
    TRC_EXPECT(b.rng.state == a.rng.state);
    TRC_EXPECT(checkpoint_verify(path.c_str(), ib));

    std::remove(path.c_str());
}

TRC_TEST(checkpoint_multi_resume_exact) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("multi_resume");
    std::remove(path.c_str());

    // A：训练 6 步 → 存档 → 再训练 6 步
    DualSession a(dev, 7);
    a.alloc_state();
    for (int i = 0; i < 6; ++i) {
        a.step();
    }
    CheckpointItems ia = a.items();
    TRC_EXPECT(checkpoint_save(path.c_str(), ia, "dual.resume"));
    for (int i = 0; i < 6; ++i) {
        a.step();
    }

    // B：恢复后训练 6 步，轨迹必须与 A 逐位一致
    DualSession b(dev, 42);
    b.alloc_state();
    CheckpointItems ib = b.items();
    TRC_EXPECT(checkpoint_load(path.c_str(), ib));
    for (int i = 0; i < 6; ++i) {
        b.step();
    }

    for (int64_t i = 0; i < optim_param_count(a.g_opt); ++i) {
        expect_exact(read_f32(optim_param(a.g_opt, i)), read_f32(optim_param(b.g_opt, i)),
                     "续训 G 参数不一致");
    }
    for (int64_t i = 0; i < optim_param_count(a.d_opt); ++i) {
        expect_exact(read_f32(optim_param(a.d_opt, i)), read_f32(optim_param(b.d_opt, i)),
                     "续训 D 参数不一致");
    }
    for (int64_t i = 0; i < optim_state_count(a.g_opt); ++i) {
        expect_exact(read_f32(optim_state(a.g_opt, i)), read_f32(optim_state(b.g_opt, i)),
                     "续训 G 状态不一致");
    }
    for (int64_t i = 0; i < optim_state_count(a.d_opt); ++i) {
        expect_exact(read_f32(optim_state(a.d_opt, i)), read_f32(optim_state(b.d_opt, i)),
                     "续训 D 状态不一致");
    }
    TRC_EXPECT(read_scalar(a.g_loss) == read_scalar(b.g_loss));
    TRC_EXPECT(read_scalar(a.d_loss) == read_scalar(b.d_loss));
    TRC_EXPECT_EQ(optim_step_count(b.g_opt), optim_step_count(a.g_opt));
    TRC_EXPECT_EQ(optim_step_count(b.d_opt), optim_step_count(a.d_opt));
    TRC_EXPECT_EQ(lr_scheduler_epoch(b.g_sched), lr_scheduler_epoch(a.g_sched));
    TRC_EXPECT_EQ(lr_scheduler_epoch(b.d_sched), lr_scheduler_epoch(a.d_sched));
    TRC_EXPECT(b.rng.state == a.rng.state);
    std::printf("    双优化器续训逐位等价：G loss %.6g / D loss %.6g（12 步 + 恢复 6 步）\n",
                (double) read_scalar(a.g_loss), (double) read_scalar(a.d_loss));

    std::remove(path.c_str());
}

TRC_TEST(checkpoint_v1_load_compat) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("v1_compat");
    std::remove(path.c_str());

    // 训练一个小模型，然后用旧键名手工写出 v1 文件（模拟旧版本存档）
    TrainSession a(dev, 11, /*use_sgd=*/false, /*use_sched=*/true);
    for (int i = 0; i < 3; ++i) {
        a.step();
    }

    GgufWriter* w = gguf_writer_new();
    gguf_writer_set_arch(w, "traincpp.checkpoint");
    gguf_writer_set_u32(w, "traincpp.checkpoint.version", 1);
    gguf_writer_set_str(w, "traincpp.model.arch", "v1.test");
    gguf_writer_set_str(w, "traincpp.optimizer.type", "adamw");
    gguf_writer_set_u32(w, "traincpp.optimizer.groups", 1);
    gguf_writer_set_u32(w, "traincpp.optimizer.param_count", 4);
    gguf_writer_set_i64(w, "traincpp.optimizer.step", optim_step_count(a.opt));
    gguf_writer_set_bool(w, "traincpp.optimizer.has_state", true);

    std::vector<const char*> pnames;
    std::vector<int64_t>     steps;
    for (int64_t i = 0; i < optim_param_count(a.opt); ++i) {
        pnames.push_back(optim_param(a.opt, i)->name);
        steps.push_back(optim_state_step(a.opt, i));
    }
    std::vector<const char*> snames;
    for (int64_t i = 0; i < optim_state_count(a.opt); ++i) {
        snames.push_back(optim_state(a.opt, i)->name);
    }
    gguf_writer_set_arr_str(w, "traincpp.optimizer.param_names", pnames.data(),
                            (int64_t) pnames.size());
    gguf_writer_set_arr_str(w, "traincpp.optimizer.state_names", snames.data(),
                            (int64_t) snames.size());
    gguf_writer_set_arr_i64(w, "traincpp.optimizer.state_steps", steps.data(),
                            (int64_t) steps.size());

    const AdamwOptions ao = optim_group_adamw_options(a.opt, 0);
    gguf_writer_set_f32(w, "traincpp.optimizer.group0.lr", ao.lr);
    gguf_writer_set_f32(w, "traincpp.optimizer.group0.beta1", ao.beta1);
    gguf_writer_set_f32(w, "traincpp.optimizer.group0.beta2", ao.beta2);
    gguf_writer_set_f32(w, "traincpp.optimizer.group0.eps", ao.eps);
    gguf_writer_set_f32(w, "traincpp.optimizer.group0.weight_decay", ao.weight_decay);
    gguf_writer_set_u64(w, "traincpp.rng.state", a.rng.state);
    gguf_writer_set_i64(w, "traincpp.scheduler.epoch", lr_scheduler_epoch(a.sched));

    for (int64_t i = 0; i < optim_param_count(a.opt); ++i) {
        Tensor* p = optim_param(a.opt, i);
        gguf_writer_add_tensor(w, p->name, p, TYPE_F32);
    }
    for (int64_t i = 0; i < optim_state_count(a.opt); ++i) {
        Tensor* s = optim_state(a.opt, i);
        gguf_writer_add_tensor(w, s->name, s, TYPE_F32);
    }
    TRC_EXPECT(gguf_writer_write(w, path.c_str()));
    gguf_writer_free(w);

    CheckpointInfo info;
    TRC_EXPECT(checkpoint_probe(path.c_str(), &info));
    TRC_EXPECT_EQ(info.version, 1);
    TRC_EXPECT_EQ(info.optimizer_count, 1);
    TRC_EXPECT_EQ(info.param_count, 4);
    TRC_EXPECT_EQ(info.step, 3);
    TRC_EXPECT(std::string(info.optimizer_type) == "adamw");
    TRC_EXPECT(std::string(info.model_arch) == "v1.test");
    TRC_EXPECT_EQ(info.scheduler_epoch, 3);

    // 单优化器快捷字段载入 v1
    TrainSession b(dev, 55, false, true);
    optim_alloc_state(b.opt);
    TRC_EXPECT(checkpoint_load(path.c_str(), CheckpointItems{b.opt, &b.rng, b.sched}));
    expect_same(read_f32(a.w1), read_f32(b.w1), "v1 参数 w1 载入不一致");
    expect_same(read_f32(a.b2), read_f32(b.b2), "v1 参数 b2 载入不一致");
    for (int64_t i = 0; i < optim_state_count(a.opt); ++i) {
        expect_same(read_f32(optim_state(a.opt, i)), read_f32(optim_state(b.opt, i)),
                    "v1 状态载入不一致");
    }
    TRC_EXPECT_EQ(optim_step_count(b.opt), optim_step_count(a.opt));
    TRC_EXPECT_EQ(lr_scheduler_epoch(b.sched), lr_scheduler_epoch(a.sched));
    TRC_EXPECT(b.rng.state == a.rng.state);
    TRC_EXPECT(checkpoint_verify(path.c_str(), CheckpointItems{b.opt, &b.rng, b.sched}));

    // entries 单分区也可读 v1（名字对 v1 无意义）；请求多分区必须失败
    CheckpointOptimizer one[1] = {{"G", b.opt, b.sched}};
    CheckpointItems     mi{};
    mi.optimizers      = one;
    mi.optimizer_count = 1;
    mi.rng             = &b.rng;
    TRC_EXPECT(checkpoint_load(path.c_str(), mi));
    CheckpointOptimizer two[2] = {{"G", b.opt, b.sched}, {"D", b.opt, b.sched}};
    CheckpointItems     mi2{};
    mi2.optimizers      = two;
    mi2.optimizer_count = 2;
    TRC_EXPECT(!checkpoint_load(path.c_str(), mi2));

    std::remove(path.c_str());
}

// ---------------------------------------------------------------- 自检（导出后校验）

TRC_TEST(checkpoint_self_verify) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("verify");
    std::remove(path.c_str());

    TrainSession a(dev, 31, false, true);
    for (int i = 0; i < 5; ++i) {
        a.step();
    }
    TRC_EXPECT(checkpoint_save(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}, "verify.test"));
    TRC_EXPECT(checkpoint_verify(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}));
    // 不传 rng/scheduler 时只校验模型与优化器
    TRC_EXPECT(checkpoint_verify(path.c_str(), CheckpointItems{a.opt, nullptr, nullptr}));

    // 内存中篡改一个参数元素 → 自检必须失败；还原后恢复通过
    float v = 0.0f;
    tensor_get(a.w1, &v, 0, sizeof(float));
    const float v_orig = v;
    v += 1.0f;
    tensor_set(a.w1, &v, 0, sizeof(float));
    TRC_EXPECT(!checkpoint_verify(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}));
    tensor_set(a.w1, &v_orig, 0, sizeof(float));
    TRC_EXPECT(checkpoint_verify(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}));

    // RNG 状态不符 → 自检失败
    a.rng.state ^= 0x5DEECE66Dull;
    TRC_EXPECT(!checkpoint_verify(path.c_str(), CheckpointItems{a.opt, &a.rng, a.sched}));

    std::remove(path.c_str());
}

// ---------------------------------------------------------------- 错误路径

TRC_TEST(checkpoint_errors) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::string path = ckpt_tmp_path("errors");
    std::remove(path.c_str());

    TrainSession a(dev, 5, false, false);

    // 缺文件
    CheckpointInfo info;
    TRC_EXPECT(!checkpoint_probe(path.c_str(), &info));
    TRC_EXPECT(!checkpoint_load(path.c_str(), CheckpointItems{a.opt, nullptr, nullptr}));

    // 非 checkpoint 文件（普通 GGUF）
    const std::string other = ckpt_tmp_path("other");
    GgufWriter* w = gguf_writer_new();
    gguf_writer_set_arch(w, "traincpp.not_checkpoint");
    TRC_EXPECT(gguf_writer_write(w, other.c_str()));
    gguf_writer_free(w);
    TRC_EXPECT(!checkpoint_probe(other.c_str(), &info));
    TRC_EXPECT(!checkpoint_load(other.c_str(), CheckpointItems{a.opt, nullptr, nullptr}));
    std::remove(other.c_str());

    // 正常保存一份（0 步，状态未分配 → 仅参数；schema v2 单分区）
    TRC_EXPECT(checkpoint_save(path.c_str(), CheckpointItems{a.opt, nullptr, nullptr}, nullptr));
    CheckpointInfo ok;
    TRC_EXPECT(checkpoint_probe(path.c_str(), &ok));
    TRC_EXPECT_EQ(ok.version, 2);
    TRC_EXPECT_EQ(ok.optimizer_count, 1);
    TRC_EXPECT_EQ(ok.step, 0);
    TRC_EXPECT_EQ(ok.param_count, 4);

    // 参数数量不匹配
    Context* ctx2 = context_new(1 << 20);
    Tensor*  p1   = new_tensor_1d(ctx2, TYPE_F32, 4);
    Tensor*  p2   = new_tensor_1d(ctx2, TYPE_F32, 4);
    tensor_set_name(p1, "p1");
    tensor_set_name(p2, "p2");
    tensor_set_param(p1);
    tensor_set_param(p2);
    Buffer* buf2 = buffer_alloc_ctx_tensors(ctx2, dev->default_buffer_type());
    Tensor* params2[2] = {p1, p2};
    Optimizer* opt2 = optim_adamw_new(ctx2, params2, 2, AdamwOptions{});
    TRC_EXPECT(!checkpoint_load(path.c_str(), CheckpointItems{opt2, nullptr, nullptr}));
    optim_free(opt2);
    buffer_free(buf2);
    context_free(ctx2);

    // 优化器类型不匹配（同样的 4 个参数，用 SGD 载入 AdamW 存档）
    Tensor* params4[4] = {a.w1, a.b1, a.w2, a.b2};
    Optimizer* sgd = optim_sgd_new(a.ctx, params4, 4, SgdOptions{});
    TRC_EXPECT(!checkpoint_load(path.c_str(), CheckpointItems{sgd, nullptr, nullptr}));
    optim_free(sgd);

    // 超参不匹配（AdamW beta1 不同）
    AdamwOptions bad;
    bad.beta1 = 0.5f;
    Optimizer* opt_bad = optim_adamw_new(a.ctx, params4, 4, bad);
    TRC_EXPECT(!checkpoint_load(path.c_str(), CheckpointItems{opt_bad, nullptr, nullptr}));
    optim_free(opt_bad);

    // v2 分区名不匹配（文件只有默认单分区 "optimizer"）
    CheckpointOptimizer one[1] = {{"G", a.opt, nullptr}};
    CheckpointItems     mi{};
    mi.optimizers      = one;
    mi.optimizer_count = 1;
    TRC_EXPECT(!checkpoint_load(path.c_str(), mi));

    // v2 分区数量不匹配（文件 1 个，请求 2 个）
    CheckpointOptimizer two[2] = {{"G", a.opt, nullptr}, {"D", a.opt, nullptr}};
    CheckpointItems     mi2{};
    mi2.optimizers      = two;
    mi2.optimizer_count = 2;
    TRC_EXPECT(!checkpoint_load(path.c_str(), mi2));

    // entries 参数不完整（optimizers 为空但 count 非 0）→ 收集即失败
    CheckpointItems mi3{};
    mi3.optimizer_count = 1;
    TRC_EXPECT(!checkpoint_load(path.c_str(), mi3));

    std::remove(path.c_str());
}
