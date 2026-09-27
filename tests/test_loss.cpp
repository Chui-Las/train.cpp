// train.cpp - 损失函数测试（M1.5c）
//
// 覆盖：
//   1. PyTorch 黄金对拍：tests/golden/loss_{mse,l1}_*.npy（缺失时自动跳过）
//   2. mse_loss / l1_loss 有限差分 gradcheck（走 autograd）
//   3. 端到端：线性回归 + mse_loss + SGD 收敛（验证损失可直接用于训练）
#include "npy.h"
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
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

// npy（C 序）形状 -> 本库 ne（反转）；仅创建张量，数据由调用方 tensor_set
Tensor* load_tensor(Context* ctx, const std::string& path, Type type, NpyArray& arr) {
    if (!npy_load(path, arr)) {
        return nullptr;
    }
    int64_t ne[MAX_DIMS] = {1, 1, 1, 1};
    const int n_dims = (int) arr.shape.size();
    for (int i = 0; i < n_dims; ++i) {
        ne[i] = arr.shape[(size_t) (n_dims - 1 - i)];
    }
    return new_tensor_nd(ctx, type, n_dims > 0 ? n_dims : 1, ne);
}

// ---------------- 训练 fixture（gradcheck/端到端用） ----------------

struct LossFixture {
    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Graph*   g   = nullptr;
    Buffer*  buf = nullptr;

    explicit LossFixture(size_t mem = 1 << 20) {
        ctx = context_new(mem);
        dev = trc_test::test_device();
        g   = graph_new(ctx);
    }

    ~LossFixture() {
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

// 中心差分校验（同 test_grad.cpp）
void check_gradients(LossFixture& f, const std::vector<Tensor*>& xs, Tensor* loss, double eps = 1e-2,
                     double tol = 5e-3) {
    if (const Tensor* bad = trc_test::first_unsupported_node(f.g, f.dev)) {
        std::printf("    跳过：设备不支持算子 %s\n", op_name(bad->op));
        return;
    }
    f.step();
    for (Tensor* x : xs) {
        const int64_t n = tensor_nelements(x);
        const std::vector<float> base = read_f32(x);
        Tensor* gx = graph_get_grad(f.g, x);
        TRC_EXPECT(gx != nullptr);
        if (gx == nullptr) {
            continue;
        }
        const std::vector<float> analytic = read_f32(gx);
        for (int64_t i = 0; i < n; ++i) {
            const float v  = base[(size_t) i];
            const float vp = v + (float) eps;
            const float vm = v - (float) eps;
            tensor_set(x, &vp, (size_t) i * sizeof(float), sizeof(float));
            f.step();
            const float lp = read_scalar(loss);
            tensor_set(x, &vm, (size_t) i * sizeof(float), sizeof(float));
            f.step();
            const float lm = read_scalar(loss);
            const double numeric = ((double) lp - (double) lm) / (2.0 * eps);
            const double ana     = analytic[(size_t) i];
            TRC_EXPECT_NEAR(numeric, ana, tol * (1.0 + std::fabs(ana)));
            tensor_set(x, &v, (size_t) i * sizeof(float), sizeof(float));
        }
        write_f32(x, base);
        f.step();
    }
}

void sgd_update(Tensor* p, Tensor* grad, float lr) {
    const int64_t n = tensor_nelements(p);
    std::vector<float> pv = read_f32(p);
    const std::vector<float> gv = read_f32(grad);
    for (int64_t i = 0; i < n; ++i) {
        pv[(size_t) i] -= lr * gv[(size_t) i];
    }
    write_f32(p, pv);
}

} // namespace

// ---------------------------------------------------------------- 黄金对拍

TRC_TEST(loss_golden_mse_l1) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    const std::string dir = TRC_GOLDEN_DIR;
    const char* kinds[] = {"mse", "l1"};

    int n_ok = 0;
    int n_skip = 0;
    for (const char* kind : kinds) {
        Context* ctx = context_new(1 << 20);
        Graph*   g   = graph_new(ctx);

        NpyArray p_npy, t_npy, o_npy;
        Tensor* pred   = load_tensor(ctx, dir + "/loss_" + kind + "_pred.npy", TYPE_F32, p_npy);
        Tensor* target = load_tensor(ctx, dir + "/loss_" + kind + "_target.npy", TYPE_F32, t_npy);
        if (pred == nullptr || target == nullptr ||
            !npy_load(dir + "/loss_" + kind + "_out.npy", o_npy)) {
            ++n_skip;
            graph_free(g);
            context_free(ctx);
            continue;
        }

        Tensor* loss = (std::string(kind) == "mse") ? mse_loss(ctx, pred, target)
                                                    : l1_loss(ctx, pred, target);
        Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
        tensor_set(pred, p_npy.raw.data(), 0, p_npy.raw.size());
        tensor_set(target, t_npy.raw.data(), 0, t_npy.raw.size());
        graph_build_forward_expand(ctx, g, loss);

        if (const Tensor* bad = trc_test::first_unsupported_node(g, dev)) {
            std::printf("    [跳过] loss_%s 后端不支持 %s\n", kind, op_name(bad->op));
            ++n_skip;
            buffer_free(buf);
            graph_free(g);
            context_free(ctx);
            continue;
        }

        dev->graph_compute(g);
        const float got = read_scalar(loss);
        const float exp = ((const float*) o_npy.raw.data())[0];
        TRC_EXPECT_NEAR(got, exp, 1e-5 + 1e-4 * std::fabs((double) exp));
        std::printf("    [ OK ] loss_%-4s 实际 %.6g 期望 %.6g（差 %.3g）\n", kind, got, exp,
                    std::fabs((double) got - (double) exp));
        ++n_ok;

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }

    if (n_ok == 0 && n_skip > 0) {
        std::printf("    跳过：未找到 loss 黄金数据（请先运行 python scripts/gen_golden.py）\n");
    } else {
        std::printf("    loss 黄金对拍：%d 个用例通过（%d 个缺失/跳过）\n", n_ok, n_skip);
    }
}

// ---------------------------------------------------------------- gradcheck

TRC_TEST(loss_gradcheck_mse) {
    LossFixture f;
    if (!f.device_ok()) {
        return;
    }
    Tensor* pred   = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    tensor_set_param(pred);
    Tensor* loss = mse_loss(f.ctx, pred, target);
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(pred, {0.5f, -0.3f, 0.8f, 1.2f, -0.9f, 0.1f});
    write_f32(target, {0.1f, 0.2f, -0.4f, 0.7f, 0.3f, -0.2f});
    check_gradients(f, {pred}, loss);
}

TRC_TEST(loss_gradcheck_l1) {
    LossFixture f;
    if (!f.device_ok()) {
        return;
    }
    // 取值远离 0（L1 在 0 处不可导）
    Tensor* pred   = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    tensor_set_param(pred);
    Tensor* loss = l1_loss(f.ctx, pred, target);
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(pred, {0.5f, -0.3f, 0.8f, 1.2f, -0.9f, 0.1f});
    write_f32(target, {-0.6f, 0.9f, -0.2f, -0.8f, 0.4f, 1.1f});
    check_gradients(f, {pred}, loss);
}

// ---------------------------------------------------------------- 端到端

// 线性回归 y = 2*x0 - 3*x1 + 1，mse_loss + SGD：loss 应显著下降且参数有限
TRC_TEST(loss_mse_end_to_end) {
    LossFixture f(1 << 22);
    if (!f.device_ok()) {
        return;
    }

    constexpr int64_t IN = 2;
    constexpr int64_t OUT = 1;
    constexpr int64_t BATCH = 16;

    Tensor* w = new_tensor_2d(f.ctx, TYPE_F32, IN, OUT);  // [in, out]
    Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, OUT);
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, IN, BATCH);
    Tensor* t = new_tensor_2d(f.ctx, TYPE_F32, OUT, BATCH);
    tensor_set_param(w);
    tensor_set_param(b);

    Tensor* pred = add(f.ctx, mul_mat(f.ctx, w, x), b);
    Tensor* loss = mse_loss(f.ctx, pred, t);
    tensor_set_loss(loss);
    f.build(loss);

    uint32_t seed = 12345u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> vw((size_t) (IN * OUT));
    std::vector<float> vx((size_t) (IN * BATCH));
    std::vector<float> vt((size_t) BATCH);
    for (float& v : vw) {
        v = rnd() * 0.1f;
    }
    for (int64_t i = 0; i < BATCH; ++i) {
        const float x0 = rnd();
        const float x1 = rnd();
        vx[(size_t) (i * IN + 0)] = x0;
        vx[(size_t) (i * IN + 1)] = x1;
        vt[(size_t) i] = 2.0f * x0 - 3.0f * x1 + 1.0f;
    }
    write_f32(w, vw);
    write_f32(b, {0.0f});
    write_f32(x, vx);
    write_f32(t, vt);

    f.step();
    const float loss0 = read_scalar(loss);
    TRC_EXPECT(std::isfinite(loss0));
    TRC_EXPECT(loss0 > 0.0f);

    const float lr = 0.05f;
    const int   steps = 500;
    for (int it = 0; it < steps; ++it) {
        f.step();
        sgd_update(w, graph_get_grad(f.g, w), lr);
        sgd_update(b, graph_get_grad(f.g, b), lr);
    }
    f.step();
    const float loss1 = read_scalar(loss);
    std::printf("    线性回归训练: loss %.6g -> %.6g (%d 步, lr=%g)\n", loss0, loss1, steps, lr);
    TRC_EXPECT(std::isfinite(loss1));
    TRC_EXPECT(loss1 < 0.05f * loss0);

    for (Tensor* p : {w, b}) {
        for (float v : read_f32(p)) {
            TRC_EXPECT(std::isfinite(v));
        }
    }
}
