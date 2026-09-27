// train.cpp - 优化器与学习率调度测试（M1.5a）
//
// 覆盖：
//   1. 黄金数据对拍：tests/golden/optim_*.npy（scripts/gen_golden.py 生成；缺失时自动跳过）
//      - SGD（momentum/dampening/nesterov/weight_decay）、AdamW（bias correction/解耦 wd）
//      - 调度器学习率序列（StepLR/ExponentialLR/CosineAnnealingLR/warmup_cosine）
//   2. 手算校验 / 梯度裁剪 / zero_grad / 参数分组
//   3. 端到端：MLP + AdamW + Cosine 调度，loss 显著下降（CPU 与 Vulkan 均可跑）
#include "npy.h"
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
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

void write_f32(Tensor* t, const std::vector<float>& v) {
    tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

float read_scalar(const Tensor* t) {
    float v = 0.0f;
    tensor_get(t, &v, 0, sizeof(float));
    return v;
}

// ---------------- 黄金数据 meta ----------------

struct OptimMeta {
    std::string kind;
    float       lr = 0.0f;
    float       momentum = 0.0f;
    float       dampening = 0.0f;
    float       weight_decay = 0.0f;
    float       beta1 = 0.9f;
    float       beta2 = 0.999f;
    float       eps = 1e-8f;
    float       gamma = 1.0f;
    float       eta_min = 0.0f;
    int64_t     nesterov = 0;
    int64_t     steps = 0;
    int64_t     nparams = 0;
    int64_t     step_size = 1;
    int64_t     t_max = 1;
    int64_t     warmup = 0;
    int64_t     total = 0;
    std::vector<std::array<int64_t, 4>> shapes;
};

bool load_meta(const std::string& path, OptimMeta& m) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string key;
    while (f >> key) {
        if (key == "kind") {
            f >> m.kind;
        } else if (key == "lr") {
            f >> m.lr;
        } else if (key == "momentum") {
            f >> m.momentum;
        } else if (key == "dampening") {
            f >> m.dampening;
        } else if (key == "weight_decay") {
            f >> m.weight_decay;
        } else if (key == "beta1") {
            f >> m.beta1;
        } else if (key == "beta2") {
            f >> m.beta2;
        } else if (key == "eps") {
            f >> m.eps;
        } else if (key == "gamma") {
            f >> m.gamma;
        } else if (key == "eta_min") {
            f >> m.eta_min;
        } else if (key == "nesterov") {
            f >> m.nesterov;
        } else if (key == "steps") {
            f >> m.steps;
        } else if (key == "nparams") {
            f >> m.nparams;
        } else if (key == "step_size") {
            f >> m.step_size;
        } else if (key == "t_max") {
            f >> m.t_max;
        } else if (key == "warmup") {
            f >> m.warmup;
        } else if (key == "total") {
            f >> m.total;
        } else if (key.rfind("shape", 0) == 0) {
            std::array<int64_t, 4> sh{1, 1, 1, 1};
            f >> sh[0] >> sh[1] >> sh[2] >> sh[3];
            m.shapes.push_back(sh);
        } else {
            std::string rest;
            std::getline(f, rest);
        }
    }
    return true;
}

// 与期望（连续 F32 数据）逐元素比较，返回最大绝对误差
double compare_f32(const Tensor* t, const float* expect, int64_t n, double atol, double rtol) {
    const std::vector<float> got = read_f32(t);
    double max_diff = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double diff = std::fabs((double) got[(size_t) i] - (double) expect[(size_t) i]);
        const double tol  = atol + rtol * std::fabs((double) expect[(size_t) i]);
        if (diff > tol) {
            trc_test::add_failure("优化器对拍: 元素 " + std::to_string(i) + " 实际 " +
                                  std::to_string(got[(size_t) i]) + " 期望 " +
                                  std::to_string(expect[(size_t) i]));
        }
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

// ---------------- 样例用的设备 fixture ----------------

struct OptimFixture {
    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Buffer*  buf = nullptr;
    Optimizer* opt = nullptr;

    explicit OptimFixture(size_t mem = 1 << 22) {
        ctx = context_new(mem);
        dev = trc_test::test_device();
    }

    ~OptimFixture() {
        if (opt != nullptr) {
            optim_free(opt);
        }
        if (buf != nullptr) {
            buffer_free(buf);
        }
        context_free(ctx);
    }

    bool device_ok() const {
        if (dev == nullptr) {
            std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
            return false;
        }
        return true;
    }

    void alloc() { buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type()); }
};

// 为参数准备合成梯度张量（优化器只要求 p->grad 同形状 F32）。
// 必须在 buffer_alloc_ctx_tensors 之前创建；数据在分配完成后写入。
Tensor* make_grad_tensor(Context* ctx, const Tensor* p) {
    return new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, p->ne);
}

} // namespace

// ---------------------------------------------------------------- 黄金数据对拍

TRC_TEST(optim_golden_trajectories) {
    const std::string dir = TRC_GOLDEN_DIR;
    const char* names[]  = {"sgd_plain", "sgd_momentum", "sgd_nesterov", "sgd_wd",
                            "adamw_basic", "adamw_nowd"};

    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    int n_ok = 0;
    int n_skip = 0;
    for (const char* name : names) {
        OptimMeta meta;
        if (!load_meta(dir + "/optim_" + name + "_meta.txt", meta)) {
            ++n_skip;
            continue;
        }
        TRC_EXPECT(meta.nparams == (int64_t) meta.shapes.size());
        if (meta.nparams != (int64_t) meta.shapes.size()) {
            continue;
        }

        OptimFixture f;
        std::vector<Tensor*> params((size_t) meta.nparams);
        std::vector<Tensor*> gts((size_t) meta.nparams);
        std::vector<std::vector<float>> grads((size_t) meta.nparams);

        for (int64_t i = 0; i < meta.nparams; ++i) {
            params[(size_t) i] = new_tensor_nd(f.ctx, TYPE_F32, MAX_DIMS, meta.shapes[(size_t) i].data());
            gts[(size_t) i]    = make_grad_tensor(f.ctx, params[(size_t) i]);
            params[(size_t) i]->grad = gts[(size_t) i];
        }
        f.alloc();

        // 载入初值、构造梯度张量
        for (int64_t i = 0; i < meta.nparams; ++i) {
            const int64_t n = tensor_nelements(params[(size_t) i]);
            NpyArray init_npy;
            const std::string p_init = dir + "/optim_" + name + "_p" + std::to_string(i) + "_init.npy";
            if (!npy_load(p_init, init_npy) || init_npy.raw.size() != (size_t) n * sizeof(float)) {
                trc_test::add_failure("优化器对拍: 初值读取失败 " + p_init);
                return;
            }
            tensor_set(params[(size_t) i], init_npy.raw.data(), 0, init_npy.raw.size());

            NpyArray grads_npy;
            const std::string p_grads = dir + "/optim_" + name + "_p" + std::to_string(i) + "_grads.npy";
            if (!npy_load(p_grads, grads_npy) ||
                grads_npy.raw.size() != (size_t) n * sizeof(float) * (size_t) meta.steps) {
                trc_test::add_failure("优化器对拍: 梯度读取失败 " + p_grads);
                return;
            }
            grads[(size_t) i].resize(grads_npy.raw.size() / sizeof(float));
            std::memcpy(grads[(size_t) i].data(), grads_npy.raw.data(), grads_npy.raw.size());
        }

        // 构造优化器
        if (meta.kind == "sgd") {
            SgdOptions o;
            o.lr           = meta.lr;
            o.momentum     = meta.momentum;
            o.dampening    = meta.dampening;
            o.weight_decay = meta.weight_decay;
            o.nesterov     = meta.nesterov != 0;
            f.opt = optim_sgd_new(f.ctx, params.data(), meta.nparams, o);
        } else {
            AdamwOptions o;
            o.lr           = meta.lr;
            o.beta1        = meta.beta1;
            o.beta2        = meta.beta2;
            o.eps          = meta.eps;
            o.weight_decay = meta.weight_decay;
            f.opt = optim_adamw_new(f.ctx, params.data(), meta.nparams, o);
        }
        TRC_EXPECT(optim_param_count(f.opt) == meta.nparams);

        // 逐步对拍
        double worst = 0.0;
        for (int64_t t = 0; t < meta.steps; ++t) {
            for (int64_t i = 0; i < meta.nparams; ++i) {
                const int64_t n = tensor_nelements(params[(size_t) i]);
                tensor_set(gts[(size_t) i], grads[(size_t) i].data() + (size_t) t * (size_t) n, 0,
                           (size_t) n * sizeof(float));
            }
            optim_step(f.opt);

            for (int64_t i = 0; i < meta.nparams; ++i) {
                const int64_t n = tensor_nelements(params[(size_t) i]);
                NpyArray traj_npy;
                const std::string p_traj = dir + "/optim_" + name + "_p" + std::to_string(i) + "_traj.npy";
                if (!npy_load(p_traj, traj_npy) ||
                    traj_npy.raw.size() != (size_t) n * sizeof(float) * (size_t) (meta.steps + 1)) {
                    trc_test::add_failure("优化器对拍: 轨迹读取失败 " + p_traj);
                    return;
                }
                const float* expect = (const float*) traj_npy.raw.data() + (size_t) (t + 1) * (size_t) n;
                worst = std::max(worst, compare_f32(params[(size_t) i], expect, n, 1e-6, 1e-5));
            }
        }
        TRC_EXPECT(optim_step_count(f.opt) == meta.steps);
        std::printf("    [%s] %s %lld 步，最大绝对误差 %.3g\n", name, meta.kind.c_str(),
                    (long long) meta.steps, worst);
        ++n_ok;
    }

    if (n_ok == 0 && n_skip > 0) {
        std::printf("    跳过：未找到优化器黄金数据（请先运行 python scripts/gen_golden.py）\n");
    } else {
        std::printf("    优化器黄金对拍：%d 个用例通过（%d 个缺失跳过）\n", n_ok, n_skip);
    }
}

TRC_TEST(optim_golden_schedulers) {
    const std::string dir = TRC_GOLDEN_DIR;
    const char* names[]  = {"step", "exponential", "cosine", "warmup_cosine"};

    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    int n_ok = 0;
    for (const char* name : names) {
        OptimMeta meta;
        if (!load_meta(dir + "/optim_sched_" + name + "_meta.txt", meta)) {
            continue;
        }

        NpyArray lr_npy;
        if (!npy_load(dir + "/optim_sched_" + name + "_lr.npy", lr_npy)) {
            trc_test::add_failure(std::string("调度器对拍: 期望 lr 读取失败 ") + name);
            continue;
        }
        const int64_t n_lr = (int64_t) (lr_npy.raw.size() / sizeof(float));
        TRC_EXPECT(n_lr == meta.steps + 1);

        OptimFixture f;
        Tensor* p = new_tensor_1d(f.ctx, TYPE_F32, 1);
        f.alloc();
        write_f32(p, {0.0f});
        f.opt = optim_sgd_new(f.ctx, &p, 1, SgdOptions{});
        optim_set_lr(f.opt, meta.lr);

        LrScheduler* sched = nullptr;
        if (meta.kind == "step") {
            sched = lr_scheduler_step_new(f.opt, meta.step_size, meta.gamma);
        } else if (meta.kind == "exponential") {
            sched = lr_scheduler_exponential_new(f.opt, meta.gamma);
        } else if (meta.kind == "cosine") {
            sched = lr_scheduler_cosine_new(f.opt, meta.t_max, meta.eta_min);
        } else {
            sched = lr_scheduler_warmup_cosine_new(f.opt, meta.warmup, meta.total, meta.eta_min);
        }

        const float* expect = (const float*) lr_npy.raw.data();
        TRC_EXPECT_NEAR(lr_scheduler_get_lr(sched), (double) expect[0], 1e-7);
        for (int64_t t = 0; t < meta.steps; ++t) {
            const float lr = lr_scheduler_step(sched);
            TRC_EXPECT_NEAR(lr, (double) expect[(size_t) (t + 1)], 1e-7);
            TRC_EXPECT_NEAR(optim_get_lr(f.opt), (double) expect[(size_t) (t + 1)], 1e-7);
        }
        TRC_EXPECT(lr_scheduler_epoch(sched) == meta.steps);
        std::printf("    [sched %s] %lld 步，lr %.6g -> %.6g\n", name, (long long) meta.steps,
                    (double) expect[0], (double) expect[(size_t) meta.steps]);
        lr_scheduler_free(sched);
        ++n_ok;
    }
    if (n_ok == 0) {
        std::printf("    跳过：未找到调度器黄金数据（请先运行 python scripts/gen_golden.py）\n");
    }
}

// ---------------------------------------------------------------- 手算校验

TRC_TEST(optim_sgd_hand_computed) {
    OptimFixture f;
    if (!f.device_ok()) {
        return;
    }
    Tensor* p  = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* gp = make_grad_tensor(f.ctx, p);
    p->grad = gp;
    f.alloc();
    write_f32(p, {1.0f});
    write_f32(gp, {2.0f});

    SgdOptions o;
    o.lr           = 0.1f;
    o.weight_decay = 0.5f;
    f.opt = optim_sgd_new(f.ctx, &p, 1, o);
    optim_step(f.opt);

    // d = g + wd·p = 2 + 0.5 = 2.5；p = 1 - 0.1·2.5 = 0.75
    TRC_EXPECT_NEAR(read_scalar(p), 0.75, 1e-6);
    TRC_EXPECT(optim_state_count(f.opt) == 1);
    TRC_EXPECT(optim_state(f.opt, 0)->data != nullptr);
}

TRC_TEST(optim_adamw_hand_computed) {
    OptimFixture f;
    if (!f.device_ok()) {
        return;
    }
    Tensor* p  = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* gp = make_grad_tensor(f.ctx, p);
    p->grad = gp;
    f.alloc();
    write_f32(p, {1.0f});
    write_f32(gp, {1.0f});

    AdamwOptions o;
    o.lr           = 1e-3f;
    o.beta1        = 0.9f;
    o.beta2        = 0.999f;
    o.eps          = 1e-8f;
    o.weight_decay = 0.0f;
    f.opt = optim_adamw_new(f.ctx, &p, 1, o);
    optim_step(f.opt);

    // m=0.1、v=0.001、bc1=0.1、bc2=0.001；p -= 0.01·(0.1/1) = 0.999
    TRC_EXPECT_NEAR(read_scalar(p), 0.999, 1e-6);

    // 状态张量应为 m（=0.1）与 v（=0.001）
    TRC_EXPECT(optim_state_count(f.opt) == 2);
    TRC_EXPECT_NEAR(read_scalar(optim_state(f.opt, 0)), 0.1, 1e-6);
    TRC_EXPECT_NEAR(read_scalar(optim_state(f.opt, 1)), 0.001, 1e-6);
}

TRC_TEST(optim_zero_grad_and_clip) {
    OptimFixture f;
    if (!f.device_ok()) {
        return;
    }
    Tensor* a  = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* b  = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* ga = make_grad_tensor(f.ctx, a);
    Tensor* gb = make_grad_tensor(f.ctx, b);
    a->grad = ga;
    b->grad = gb;
    b->grad_acc = gb;
    f.alloc();
    write_f32(a, {0.0f, 0.0f});
    write_f32(b, {0.0f});
    write_f32(ga, {3.0f, 4.0f});
    write_f32(gb, {0.0f});

    SgdOptions o;
    o.lr = 0.0f;  // 只关心裁剪与清零
    Tensor* params[2] = {a, b};
    f.opt = optim_sgd_new(f.ctx, params, 2, o);

    const float norm = optim_clip_grad_norm(f.opt, 1.0f);
    TRC_EXPECT_NEAR(norm, 5.0, 1e-6);
    const float scale = 1.0f / (5.0f + 1e-6f);
    {
        const std::vector<float> va = read_f32(ga);
        TRC_EXPECT_NEAR(va[0], 3.0f * scale, 1e-6);
        TRC_EXPECT_NEAR(va[1], 4.0f * scale, 1e-6);
    }

    optim_zero_grad(f.opt);
    for (float v : read_f32(ga)) {
        TRC_EXPECT_NEAR(v, 0.0, 1e-9);
    }
    TRC_EXPECT_NEAR(read_f32(gb)[0], 0.0, 1e-9);

    // 小范数不缩放
    write_f32(ga, {0.3f, 0.4f});
    const float norm2 = optim_clip_grad_norm(f.opt, 1.0f);
    TRC_EXPECT_NEAR(norm2, 0.5, 1e-6);
    const std::vector<float> va = read_f32(ga);
    TRC_EXPECT_NEAR(va[0], 0.3, 1e-6);
    TRC_EXPECT_NEAR(va[1], 0.4, 1e-6);
}

// 优化器在 buffer_alloc_ctx_tensors 之前创建：状态张量应随 ctx 一起分配
TRC_TEST(optim_states_allocated_by_ctx_alloc) {
    OptimFixture f;
    if (!f.device_ok()) {
        return;
    }
    Tensor* p  = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* gp = make_grad_tensor(f.ctx, p);
    p->grad = gp;

    SgdOptions o;
    o.lr = 0.1f;
    f.opt = optim_sgd_new(f.ctx, &p, 1, o);  // 此时参数与状态都还没有数据
    f.alloc();

    Tensor* st = optim_state(f.opt, 0);
    TRC_EXPECT(st->buffer != nullptr);  // 由 ctx buffer 统一分配
    TRC_EXPECT(st->data != nullptr);

    write_f32(p, {1.0f, 2.0f});
    write_f32(gp, {1.0f, 1.0f});
    optim_step(f.opt);
    const std::vector<float> v = read_f32(p);
    TRC_EXPECT_NEAR(v[0], 0.9, 1e-6);
    TRC_EXPECT_NEAR(v[1], 1.9, 1e-6);
}

TRC_TEST(optim_param_groups) {
    OptimFixture f;
    if (!f.device_ok()) {
        return;
    }
    Tensor* a  = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* b  = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* ga = make_grad_tensor(f.ctx, a);
    Tensor* gb = make_grad_tensor(f.ctx, b);
    a->grad = ga;
    b->grad = gb;
    f.alloc();
    write_f32(a, {0.0f});
    write_f32(b, {0.0f});
    write_f32(ga, {1.0f});
    write_f32(gb, {1.0f});

    SgdOptions o1;
    o1.lr = 0.1f;
    f.opt = optim_sgd_new(f.ctx, &a, 1, o1);

    AdamwOptions o2;
    o2.lr = 1e-2f;
    o2.weight_decay = 0.0f;
    optim_add_param_group_adamw(f.opt, &b, 1, o2);

    TRC_EXPECT_EQ(optim_group_count(f.opt), 2);
    TRC_EXPECT_EQ(optim_param_count(f.opt), 2);
    TRC_EXPECT_EQ(optim_state_count(f.opt), 3);  // SGD: momentum；AdamW: m、v
    TRC_EXPECT_NEAR(optim_get_lr(f.opt), 0.1, 1e-7);

    optim_step(f.opt);
    TRC_EXPECT_NEAR(read_scalar(a), -0.1, 1e-6);
    TRC_EXPECT_NEAR(read_scalar(b), -0.01, 1e-5);

    optim_set_lr(f.opt, 0.01f);
    TRC_EXPECT_NEAR(optim_get_lr(f.opt), 0.01, 1e-7);
    optim_set_lr_group(f.opt, 1, 0.02f);
    TRC_EXPECT_NEAR(optim_get_lr(f.opt), 0.01, 1e-7);
}

// ---------------------------------------------------------------- 端到端训练

namespace {

struct TrainFixture {
    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Graph*   g   = nullptr;
    Buffer*  buf = nullptr;

    explicit TrainFixture(size_t mem = 1 << 22) {
        ctx = context_new(mem);
        dev = trc_test::test_device();
        g   = graph_new(ctx);
    }

    ~TrainFixture() {
        graph_free(g);
        if (buf != nullptr) {
            buffer_free(buf);
        }
        context_free(ctx);
    }

    bool device_ok() const {
        if (dev == nullptr) {
            std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
            return false;
        }
        return true;
    }

    void build(Tensor* loss) {
        graph_build_forward_expand(ctx, g, loss);
        graph_build_backward_expand(g);
        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    }

    void step() {
        graph_reset(g);
        dev->graph_compute(g);
    }
};

} // namespace

// 两层 MLP + MSE：AdamW + Cosine 调度训练，loss 应显著下降
TRC_TEST(optim_end_to_end_mlp_adamw) {
    TrainFixture f;
    if (!f.device_ok()) {
        return;
    }

    constexpr int64_t IN = 2;
    constexpr int64_t HID = 8;
    constexpr int64_t OUT = 1;
    constexpr int64_t BATCH = 16;

    Tensor* W1 = new_tensor_2d(f.ctx, TYPE_F32, IN, HID);
    Tensor* b1 = new_tensor_2d(f.ctx, TYPE_F32, HID, 1);
    Tensor* W2 = new_tensor_2d(f.ctx, TYPE_F32, HID, OUT);
    Tensor* b2 = new_tensor_2d(f.ctx, TYPE_F32, OUT, 1);
    Tensor* x  = new_tensor_2d(f.ctx, TYPE_F32, IN, BATCH);
    Tensor* y  = new_tensor_2d(f.ctx, TYPE_F32, OUT, BATCH);

    Tensor* params[4] = {W1, b1, W2, b2};
    for (Tensor* p : params) {
        tensor_set_param(p);
    }

    Tensor* h    = relu(f.ctx, add(f.ctx, mul_mat(f.ctx, W1, x), b1));
    Tensor* pred = add(f.ctx, mul_mat(f.ctx, W2, h), b2);
    Tensor* loss = scale(f.ctx, sum(f.ctx, sqr(f.ctx, sub(f.ctx, pred, y))),
                         1.0f / (float) (OUT * BATCH));
    tensor_set_loss(loss);
    f.build(loss);

    // 确定性数据（与 test_grad::end_to_end_mlp_sgd 相同的 LCG 与 teacher）
    uint32_t seed = 12345u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> vw1((size_t) (IN * HID));
    std::vector<float> vw2((size_t) (HID * OUT));
    std::vector<float> vb1((size_t) HID, 0.0f);
    std::vector<float> vb2((size_t) OUT, 0.0f);
    std::vector<float> vx((size_t) (IN * BATCH));
    std::vector<float> vy((size_t) BATCH);
    for (float& v : vw1) {
        v = rnd() * 0.5f;
    }
    for (float& v : vw2) {
        v = rnd() * 0.5f;
    }
    for (int64_t b = 0; b < BATCH; ++b) {
        const float x0 = rnd();
        const float x1 = rnd();
        vx[(size_t) (b * IN + 0)] = x0;
        vx[(size_t) (b * IN + 1)] = x1;
        vy[(size_t) b] = std::tanh(2.0f * x0 + x1);
    }
    write_f32(W1, vw1);
    write_f32(b1, vb1);
    write_f32(W2, vw2);
    write_f32(b2, vb2);
    write_f32(x, vx);
    write_f32(y, vy);

    AdamwOptions o;
    o.lr           = 0.05f;
    o.weight_decay = 0.0f;
    Optimizer* opt = optim_adamw_new(f.ctx, params, 4, o);
    LrScheduler* sched = lr_scheduler_cosine_new(opt, 300, 0.005f);
    const float lr0 = optim_get_lr(opt);

    f.step();
    const float loss0 = read_scalar(loss);
    TRC_EXPECT(std::isfinite(loss0));
    TRC_EXPECT(loss0 > 0.0f);

    const int steps = 300;
    for (int it = 0; it < steps; ++it) {
        f.step();
        optim_clip_grad_norm(opt, 100.0f);  // 远大于实际范数：不应改变结果
        optim_step(opt);
        lr_scheduler_step(sched);
    }
    f.step();
    const float loss1 = read_scalar(loss);
    std::printf("    MLP AdamW: loss %.6g -> %.6g（%d 步，lr %.4g -> %.4g）\n", loss0, loss1, steps,
                (double) lr0, (double) optim_get_lr(opt));
    TRC_EXPECT(std::isfinite(loss1));
    TRC_EXPECT(optim_get_lr(opt) < lr0);
    TRC_EXPECT(optim_step_count(opt) == steps);

    for (Tensor* p : params) {
        for (float v : read_f32(p)) {
            TRC_EXPECT(std::isfinite(v));
        }
    }
    TRC_EXPECT(loss1 < 0.2f * loss0);

    lr_scheduler_free(sched);
    optim_free(opt);
}

// ---------------------------------------------------------------- 梯度累积（M1.5d）

namespace {

struct TrainRunResult {
    std::vector<float> params[4];
    float loss0 = 0.0f;
    float loss1 = 0.0f;
    bool  ok = false;
};

// MLP + MSE 训练：micro_batch 为每个 micro-batch 的样本数，accum 为累积数
// （损失缩放 1/accum，使累积梯度等价于全批量均值）；数据按 batch=FULL 提供，按列切片
TrainRunResult run_mlp_accum_training(int64_t micro_batch, int64_t accum,
                                      const std::vector<std::vector<float>>& xb,
                                      const std::vector<std::vector<float>>& yb, int steps) {
    constexpr int64_t IN = 2;
    constexpr int64_t HID = 8;
    constexpr int64_t OUT = 1;
    TrainRunResult res;

    TrainFixture f(1 << 22);
    if (!f.device_ok()) {
        return res;
    }

    Tensor* W1 = new_tensor_2d(f.ctx, TYPE_F32, IN, HID);
    Tensor* b1 = new_tensor_2d(f.ctx, TYPE_F32, HID, 1);
    Tensor* W2 = new_tensor_2d(f.ctx, TYPE_F32, HID, OUT);
    Tensor* b2 = new_tensor_2d(f.ctx, TYPE_F32, OUT, 1);
    Tensor* x  = new_tensor_2d(f.ctx, TYPE_F32, IN, micro_batch);
    Tensor* y  = new_tensor_2d(f.ctx, TYPE_F32, OUT, micro_batch);
    Tensor* params[4] = {W1, b1, W2, b2};
    for (Tensor* p : params) {
        tensor_set_param(p);
    }

    Tensor* h    = relu(f.ctx, add(f.ctx, mul_mat(f.ctx, W1, x), b1));
    Tensor* pred = add(f.ctx, mul_mat(f.ctx, W2, h), b2);
    Tensor* loss = scale(f.ctx, mse_loss(f.ctx, pred, y), 1.0f / (float) accum);
    tensor_set_loss(loss);
    f.build(loss);

    // 固定初始化（与 micro_batch/accum 无关，保证各次运行的初值一致）
    uint32_t seed = 777u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> vw1((size_t) (IN * HID));
    std::vector<float> vw2((size_t) (HID * OUT));
    for (float& v : vw1) {
        v = rnd() * 0.5f;
    }
    for (float& v : vw2) {
        v = rnd() * 0.5f;
    }
    write_f32(W1, vw1);
    write_f32(b1, std::vector<float>((size_t) HID, 0.0f));
    write_f32(W2, vw2);
    write_f32(b2, std::vector<float>((size_t) OUT, 0.0f));

    AdamwOptions o;
    o.lr           = 0.05f;
    o.weight_decay = 0.0f;
    Optimizer* opt = optim_adamw_new(f.ctx, params, 4, o);

    for (int it = 0; it < steps; ++it) {
        graph_reset(f.g);
        for (int64_t k = 0; k < accum; ++k) {
            graph_reset_accumulate(f.g);  // 只播种不清零：micro-batch 梯度累加
            const std::vector<float> xs(xb[(size_t) it].begin() + (size_t) (k * micro_batch * IN),
                                        xb[(size_t) it].begin() + (size_t) ((k + 1) * micro_batch * IN));
            const std::vector<float> ys(yb[(size_t) it].begin() + (size_t) (k * micro_batch),
                                        yb[(size_t) it].begin() + (size_t) ((k + 1) * micro_batch));
            write_f32(x, xs);
            write_f32(y, ys);
            f.dev->graph_compute(f.g);
            if (it == 0 && k == 0) {
                res.loss0 = read_scalar(loss) * (float) accum;  // 还原成全批量量级
            }
        }
        optim_clip_grad_norm(opt, 100.0f);
        optim_step(opt);
        if (it == steps - 1) {
            res.loss1 = read_scalar(loss) * (float) accum;
        }
    }

    for (int i = 0; i < 4; ++i) {
        res.params[i] = read_f32(params[i]);
    }
    res.ok = true;
    optim_free(opt);
    return res;
}

} // namespace

// 梯度累积等价性（优化器轨迹级）：K 个 micro-batch（损失 1/K）与全批量 AdamW 轨迹一致
TRC_TEST(optim_accumulation_matches_full_batch) {
    Device* dev = test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    constexpr int64_t FULL = 8;
    const int steps = 20;

    // 确定性数据：每步一个全批量（8 样本），teacher 同 test_grad
    uint32_t seed = 4242u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    };
    std::vector<std::vector<float>> xb((size_t) steps), yb((size_t) steps);
    for (int s = 0; s < steps; ++s) {
        xb[(size_t) s].resize((size_t) (2 * FULL));
        yb[(size_t) s].resize((size_t) FULL);
        for (int64_t b = 0; b < FULL; ++b) {
            const float x0 = rnd();
            const float x1 = rnd();
            xb[(size_t) s][(size_t) (b * 2 + 0)] = x0;
            xb[(size_t) s][(size_t) (b * 2 + 1)] = x1;
            yb[(size_t) s][(size_t) b] = std::tanh(2.0f * x0 + x1);
        }
    }

    const TrainRunResult full = run_mlp_accum_training(FULL, 1, xb, yb, steps);
    const TrainRunResult acc2 = run_mlp_accum_training(FULL / 2, 2, xb, yb, steps);
    const TrainRunResult acc4 = run_mlp_accum_training(FULL / 4, 4, xb, yb, steps);
    TRC_EXPECT(full.ok && acc2.ok && acc4.ok);

    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT(full.params[i].size() == acc2.params[i].size());
        for (size_t j = 0; j < full.params[i].size(); ++j) {
            const double v = (double) full.params[i][j];
            TRC_EXPECT_NEAR(acc2.params[i][j], v, 2e-5 * (1.0 + std::fabs(v)));
            TRC_EXPECT_NEAR(acc4.params[i][j], v, 2e-5 * (1.0 + std::fabs(v)));
        }
    }

    std::printf("    MLP AdamW 累积等价：1x 全批量 loss %.6g -> %.6g；2x/4x 累积参数一致\n",
                (double) full.loss0, (double) full.loss1);
    TRC_EXPECT(std::isfinite(full.loss1));
    TRC_EXPECT(full.loss1 < full.loss0);
}

// 训练中动态加参数组（M2.1h / R11）：已 step 后 add_param_group 必须为新组正确分配状态。
// 验收：动态加入的第二组轨迹 == 全新优化器（同超参/同初值/同梯度序列）的第二组轨迹；
//       第一组轨迹 == 单组优化器全程轨迹。
TRC_TEST(optim_add_group_after_step) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    constexpr int N = 4;
    const float   init[N] = {0.5f, -1.0f, 2.0f, 0.25f};
    const float   grads[5][N] = {
        {0.1f, -0.2f, 0.3f, -0.4f},
        {0.4f, 0.3f, -0.2f, 0.1f},
        {-0.5f, 0.25f, 0.75f, -0.125f},
        {0.05f, -0.15f, 0.35f, -0.45f},
        {0.45f, 0.35f, -0.25f, 0.15f},
    };

    SgdOptions o1;
    o1.lr           = 0.05f;
    o1.momentum     = 0.9f;
    o1.dampening    = 0.0f;
    o1.weight_decay = 0.01f;
    SgdOptions o2   = o1;
    o2.lr           = 0.02f;

    // 动态加组路径：前 3 步只有第一组；第 3 步动态加入第二组
    OptimFixture dyn_f;
    Tensor*      dw1 = new_tensor_1d(dyn_f.ctx, TYPE_F32, N);
    Tensor*      dg1 = make_grad_tensor(dyn_f.ctx, dw1);
    dw1->grad        = dg1;
    Tensor* dw2      = new_tensor_1d(dyn_f.ctx, TYPE_F32, N);
    Tensor* dg2      = make_grad_tensor(dyn_f.ctx, dw2);
    dw2->grad        = dg2;
    dyn_f.alloc();
    tensor_set(dw1, init, 0, sizeof(init));
    tensor_set(dw2, init, 0, sizeof(init));

    dyn_f.opt = optim_sgd_new(dyn_f.ctx, &dw1, 1, o1);
    for (int t = 0; t < 5; ++t) {
        if (t == 3) {
            optim_add_param_group_sgd(dyn_f.opt, &dw2, 1, o2);
        }
        tensor_set(dg1, grads[t], 0, sizeof(grads[t]));
        if (t >= 3) {
            tensor_set(dg2, grads[t], 0, sizeof(grads[t]));
        }
        optim_step(dyn_f.opt);
    }
    const std::vector<float> dyn_w1 = read_f32(dw1);
    const std::vector<float> dyn_w2 = read_f32(dw2);

    // 参照1：第一组 = 单组优化器 5 步（标准路径）
    OptimFixture ref1;
    Tensor*      rw1 = new_tensor_1d(ref1.ctx, TYPE_F32, N);
    Tensor*      rg1 = make_grad_tensor(ref1.ctx, rw1);
    rw1->grad        = rg1;
    ref1.alloc();
    tensor_set(rw1, init, 0, sizeof(init));
    ref1.opt = optim_sgd_new(ref1.ctx, &rw1, 1, o1);
    for (int t = 0; t < 5; ++t) {
        tensor_set(rg1, grads[t], 0, sizeof(grads[t]));
        optim_step(ref1.opt);
    }
    const std::vector<float> w1_ref = read_f32(rw1);

    // 参照2：第二组 = 全新优化器从 t=3 起跑 2 步
    OptimFixture ref2;
    Tensor*      rw2 = new_tensor_1d(ref2.ctx, TYPE_F32, N);
    Tensor*      rg2 = make_grad_tensor(ref2.ctx, rw2);
    rw2->grad        = rg2;
    ref2.alloc();
    tensor_set(rw2, init, 0, sizeof(init));
    ref2.opt = optim_sgd_new(ref2.ctx, &rw2, 1, o2);
    for (int t = 3; t < 5; ++t) {
        tensor_set(rg2, grads[t], 0, sizeof(grads[t]));
        optim_step(ref2.opt);
    }
    const std::vector<float> w2_ref = read_f32(rw2);

    TRC_EXPECT(dyn_w1.size() == N && dyn_w2.size() == N);
    double worst = 0.0;
    for (int i = 0; i < N; ++i) {
        worst = std::max(worst, std::fabs((double) dyn_w1[(size_t) i] - (double) w1_ref[(size_t) i]));
        worst = std::max(worst, std::fabs((double) dyn_w2[(size_t) i] - (double) w2_ref[(size_t) i]));
    }
    TRC_EXPECT_NEAR(worst, 0.0, 1e-6);
    std::printf("    动态加组轨迹与独立参照一致（最大差 %.3g）\n", worst);
}
