// train.cpp - autograd 测试：有限差分 gradcheck + 梯度累加
// 约定：对每个参数张量 x，数值梯度用中心差分 (L(x+eps)-L(x-eps))/(2eps)，
//       与反向图给出的解析梯度逐元素比较。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
#include <vector>

using namespace traincpp;

namespace {

struct GradFixture {
    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Graph*   g   = nullptr;
    Buffer*  buf = nullptr;

    explicit GradFixture(size_t mem = 1 << 20) {
        ctx = context_new(mem);
        dev = trc_test::test_device();  // TRC_TEST_DEVICE=cpu（默认）/ vulkan
        g   = graph_new(ctx);
    }

    ~GradFixture() {
        graph_free(g);
        if (buf != nullptr) {
            buffer_free(buf);
        }
        context_free(ctx);
    }

    // 前向展开 + 反向展开 + 分配内存（顺序必须如此）
    void build(Tensor* loss) {
        graph_build_forward_expand(ctx, g, loss);
        graph_build_backward_expand(g);
        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    }

    // 一个训练步：清零梯度累加器（LOSS 播种 1.0）→ 前向 + 反向
    void step() {
        graph_reset(g);
        dev->graph_compute(g);
    }
};

std::vector<float> read_f32(const Tensor* t) {
    std::vector<float> v((size_t) tensor_nelements(t));
    tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// 设备不可用（TRC_TEST_DEVICE=vulkan 但无 Vulkan 设备）时跳过当前用例
bool grad_device_ok(const GradFixture& f) {
    if (f.dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return false;
    }
    return true;
}

void write_f32(Tensor* t, const std::vector<float>& v) {
    tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

void write_i32(Tensor* t, const std::vector<int32_t>& v) {
    tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
}

float read_scalar(const Tensor* t) {
    float v = 0.0f;
    tensor_get(t, &v, 0, sizeof(float));
    return v;
}

// 中心差分校验：对 xs 中每个张量的每个元素比较解析梯度与数值梯度
void check_gradients(GradFixture& f, const std::vector<Tensor*>& xs, Tensor* loss, double eps = 1e-2,
                     double tol = 5e-3) {
    if (!grad_device_ok(f)) {
        return;
    }
    // 整图含当前设备不支持的算子时跳过（与 test_golden 行为一致，打印原因不静默）
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
        TRC_EXPECT(tensor_is_contiguous(gx));
        if (gx == nullptr) {
            continue;
        }
        const std::vector<float> analytic = read_f32(gx);

        for (int64_t i = 0; i < n; ++i) {
            const float v = base[(size_t) i];
            float vp = v + (float) eps;
            float vm = v - (float) eps;

            tensor_set(x, &vp, (size_t) i * sizeof(float), sizeof(float));
            f.step();
            const float lp = read_scalar(loss);

            tensor_set(x, &vm, (size_t) i * sizeof(float), sizeof(float));
            f.step();
            const float lm = read_scalar(loss);

            const double numeric = ((double) lp - (double) lm) / (2.0 * eps);
            const double ana     = analytic[(size_t) i];
            TRC_EXPECT_NEAR(numeric, ana, tol * (1.0 + std::fabs(ana)));

            // 还原当前元素，避免影响下一个元素的差分
            tensor_set(x, &v, (size_t) i * sizeof(float), sizeof(float));
        }

        write_f32(x, base);
        f.step();
    }
}

} // namespace

// ---------------------------------------------------------------- 二元

TRC_TEST(grad_add_sub_mul_sum) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* y = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    tensor_set_param(x);
    tensor_set_param(y);
    // L = sum( (x+y) * (x-y) ) = sum(x² - y²)
    Tensor* loss = sum(f.ctx, mul(f.ctx, add(f.ctx, x, y), sub(f.ctx, x, y)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    write_f32(y, {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f});
    check_gradients(f, {x, y}, loss);
}

TRC_TEST(grad_div) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* y = new_tensor_1d(f.ctx, TYPE_F32, 4);
    tensor_set_param(x);
    tensor_set_param(y);
    Tensor* loss = sum(f.ctx, div(f.ctx, x, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(y, {0.5f, 1.5f, 2.5f, 3.5f});
    check_gradients(f, {x, y}, loss);
}

TRC_TEST(grad_broadcast) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 3);
    tensor_set_param(x);
    tensor_set_param(b);
    // L = sum( x * (x + b) )：b 沿 ne1 广播，反向需 repeat_back
    Tensor* loss = sum(f.ctx, mul(f.ctx, x, add(f.ctx, x, b)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    write_f32(b, {0.5f, 1.5f, 2.5f});
    check_gradients(f, {x, b}, loss);
}

TRC_TEST(grad_add1) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* s = new_tensor_1d(f.ctx, TYPE_F32, 1);
    tensor_set_param(x);
    tensor_set_param(s);
    Tensor* loss = sum(f.ctx, add1(f.ctx, x, s));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(s, {0.5f});
    check_gradients(f, {x, s}, loss);
}

// ---------------------------------------------------------------- 一元

TRC_TEST(grad_unary_basic) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 6);
    tensor_set_param(x);
    // 覆盖 neg / abs / sqr / exp，值均远离 0
    Tensor* t1 = add(f.ctx, neg(f.ctx, x), abs_op(f.ctx, x));
    Tensor* t2 = add(f.ctx, sqr(f.ctx, x), exp_op(f.ctx, x));
    Tensor* loss = sum(f.ctx, add(f.ctx, t1, t2));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {-1.2f, -0.7f, -0.3f, 0.4f, 0.9f, 1.6f});
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_unary_sqrt_log_sin_cos) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 5);
    tensor_set_param(x);
    Tensor* t1 = add(f.ctx, sqrt_op(f.ctx, x), log_op(f.ctx, x));
    Tensor* t2 = add(f.ctx, sin_op(f.ctx, x), cos_op(f.ctx, x));
    Tensor* loss = sum(f.ctx, add(f.ctx, t1, t2));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {0.3f, 0.8f, 1.5f, 2.2f, 3.1f});
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_activations) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 6);
    tensor_set_param(x);
    // relu / leaky_relu / sigmoid / tanh 一组；silu / softplus 一组
    Tensor* t1 = add(f.ctx,
                     add(f.ctx, relu(f.ctx, x), leaky_relu(f.ctx, x, 0.1f)),
                     add(f.ctx, sigmoid(f.ctx, x), tanh_op(f.ctx, x)));
    Tensor* t2 = add(f.ctx, silu(f.ctx, x), softplus(f.ctx, x));
    Tensor* loss = sum(f.ctx, add(f.ctx, t1, t2));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {-1.2f, -0.7f, -0.3f, 0.4f, 0.9f, 1.6f});
    check_gradients(f, {x}, loss);
}

// LeakyReLU 反向在 ±0 的导数约定：
// PyTorch：x > 0 → 1；x <= 0（含 ±0）→ slope。kink 点中心差分不可用，此处用解析断言。
TRC_TEST(grad_leaky_relu_zero_convention) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 6);
    tensor_set_param(x);
    Tensor* loss = sum(f.ctx, leaky_relu(f.ctx, x, 0.1f));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {0.0f, -0.0f, 1e-30f, -1e-30f, 1.0f, -1.0f});
    f.step();

    Tensor* gx = graph_get_grad(f.g, x);
    TRC_EXPECT(gx != nullptr);
    if (gx == nullptr) {
        return;
    }
    const std::vector<float> got    = read_f32(gx);
    const std::vector<float> expect = {0.1f, 0.1f, 1.0f, 0.1f, 1.0f, 0.1f};
    for (size_t i = 0; i < expect.size(); ++i) {
        TRC_EXPECT_NEAR(got[i], expect[i], 1e-6);
    }
}

TRC_TEST(grad_clamp_scale) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 4);
    tensor_set_param(x);
    // clamp 微分点避开边界：-0.8/-0.2/0.1/0.7，范围 [-0.5, 0.5] → 梯度 0/1/1/0
    Tensor* loss = sum(f.ctx, scale(f.ctx, clamp(f.ctx, x, -0.5f, 0.5f), 2.0f));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {-0.8f, -0.2f, 0.1f, 0.7f});
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_hardswish_gelu) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 6);
    tensor_set_param(x);
    // hardswish 的两个折点 ±3 要避开
    Tensor* loss = sum(f.ctx, add(f.ctx, hardswish(f.ctx, x), gelu(f.ctx, x)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {-2.5f, -1.1f, -0.4f, 0.6f, 1.8f, 2.9f});
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 归约

TRC_TEST(grad_reductions) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    tensor_set_param(x);
    // L = sum(mean(x)) + sum(sum_rows(x))：mean 反向按 1/ne0 缩放，sum_rows 反向 repeat
    Tensor* t1 = sum(f.ctx, mean(f.ctx, x));
    Tensor* t2 = sum(f.ctx, sum_rows(f.ctx, x));
    Tensor* loss = add(f.ctx, t1, t2);
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 线代

TRC_TEST(grad_mul_mat) {
    GradFixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);  // [k=3, m=2]
    Tensor* b = new_tensor_2d(f.ctx, TYPE_F32, 3, 4);  // [k=3, n=4]
    tensor_set_param(a);
    tensor_set_param(b);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, mul_mat(f.ctx, a, b)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    write_f32(b, {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.2f, 0.4f, 0.6f, 0.8f});
    check_gradients(f, {a, b}, loss);
}

namespace {

// 生成小幅确定值，便于有限差分（避开 0 与溢出的组合）
std::vector<float> small_ramp(int64_t n) {
    std::vector<float> v((size_t) n);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.25f + 0.03f * (float) (i % 17);
    }
    return v;
}

} // namespace

// mul_mat batch：a 单平面广播到 b 的 2x2 平面（dA 必须沿 b 的平面归约回 a）
TRC_TEST(grad_mul_mat_batch) {
    GradFixture f;
    Tensor* a = new_tensor_4d(f.ctx, TYPE_F32, 3, 2, 1, 1);  // [k=3, m=2, 1, 1]
    Tensor* b = new_tensor_4d(f.ctx, TYPE_F32, 3, 4, 2, 2);  // [k=3, n=4, b2=2, b3=2]
    tensor_set_param(a);
    tensor_set_param(b);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, mul_mat(f.ctx, a, b)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(a, small_ramp(6));
    write_f32(b, small_ramp(48));
    check_gradients(f, {a, b}, loss);
}

// mul_mat batch 整除广播：a2=2、b2=4（dA 归约需 repeat_back 支持整除）
TRC_TEST(grad_mul_mat_batch_div) {
    GradFixture f;
    Tensor* a = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 2, 1);  // [k=2, m=2, a2=2, 1]
    Tensor* b = new_tensor_4d(f.ctx, TYPE_F32, 2, 3, 4, 1);  // [k=2, n=3, b2=4, 1]
    tensor_set_param(a);
    tensor_set_param(b);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, mul_mat(f.ctx, a, b)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(a, small_ramp(8));
    write_f32(b, small_ramp(24));
    check_gradients(f, {a, b}, loss);
}

// conv_transpose_1d 反向（M2.3b）：dW 经 im2col+mul_mat、dX 经 mul_mat（stride=2 含取整）
TRC_TEST(grad_conv_transpose_1d) {
    GradFixture f;
    Tensor* w = new_tensor_3d(f.ctx, TYPE_F32, 3, 2, 2);  // [K=3, Cout=2, Cin=2]
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 5, 2);     // [T_in=5, Cin=2]
    tensor_set_param(w);
    tensor_set_param(x);
    Tensor* y = conv_transpose_1d(f.ctx, w, x, 2, 0, 1);  // [T_out=(5-1)*2+3=11, Cout=2]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(w, small_ramp(12));
    write_f32(x, small_ramp(10));
    check_gradients(f, {w, x}, loss);
}

// pool_2d 反向（M2.3g）：AVG 无填充（窗口不重叠）
TRC_TEST(grad_pool_2d_avg) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);  // [W=4, H=3]
    tensor_set_param(x);
    Tensor* y = pool_2d(f.ctx, x, POOL_AVG, 2, 2, 2, 2, 0, 0);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, small_ramp(12));
    check_gradients(f, {x}, loss);
}

// pool_2d 反向（M2.3g）：AVG 重叠窗口 + 填充（分母恒为 k0*k1）
TRC_TEST(grad_pool_2d_avg_pad) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 3);
    tensor_set_param(x);
    Tensor* y = pool_2d(f.ctx, x, POOL_AVG, 2, 2, 1, 1, 1, 1);  // [4,4]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, small_ramp(9));
    check_gradients(f, {x}, loss);
}

// pool_2d 反向（M2.3g）：MAX 重叠窗口（梯度归到各窗口首个严格最大值）
TRC_TEST(grad_pool_2d_max) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 3);
    tensor_set_param(x);
    Tensor* y = pool_2d(f.ctx, x, POOL_MAX, 2, 2, 1, 1, 1, 1);  // [4,4]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, small_ramp(9));
    check_gradients(f, {x}, loss);
}

// conv_transpose_2d 反向（M2.3g）：stride=1 组合反向（im2col + mul_mat）
TRC_TEST(grad_conv_transpose_2d) {
    GradFixture f;
    Tensor* w = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 2, 2);  // [KW=2, KH=2, Cout=2, Cin=2]
    Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 3, 2, 2, 1);  // [W=3, H=2, Cin=2, N=1]
    tensor_set_param(w);
    tensor_set_param(x);
    Tensor* y = conv_transpose_2d(f.ctx, w, x, 1);  // [4,3,2,1]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(w, small_ramp(16));
    write_f32(x, small_ramp(12));
    check_gradients(f, {w, x}, loss);
}

// conv_transpose_2d 反向（M2.3g）：stride=2（含 (ow-kw)%stride 过滤）
TRC_TEST(grad_conv_transpose_2d_stride2) {
    GradFixture f;
    Tensor* w = new_tensor_4d(f.ctx, TYPE_F32, 3, 2, 2, 2);  // [KW=3, KH=2, Cout=2, Cin=2]
    Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 3, 2, 2, 1);
    tensor_set_param(w);
    tensor_set_param(x);
    Tensor* y = conv_transpose_2d(f.ctx, w, x, 2);  // [7,4,2,1]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(w, small_ramp(24));
    write_f32(x, small_ramp(12));
    check_gradients(f, {w, x}, loss);
}

// scale_bias：标量仿射 y = s*x + b（组合实现）
TRC_TEST(grad_scale_bias) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    tensor_set_param(x);
    Tensor* y = scale_bias(f.ctx, x, 1.7f, -0.3f);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, small_ramp(12));
    check_gradients(f, {x}, loss);
}

// pad 补零左填充反向（M2.3c）：梯度应取中心区域（含左填充偏移）
TRC_TEST(grad_pad_zero_left) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    tensor_set_param(x);
    Tensor* y = pad_ext(f.ctx, x, 1, 2, 2, 0, 0, 0, 0, 0, PAD_ZERO);  // [6,4]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, small_ramp(6));
    check_gradients(f, {x}, loss);
}

// pad reflect 反向（M2.3c）：左右/上下非对称，覆盖折叠区叠加
TRC_TEST(grad_pad_reflect) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    tensor_set_param(x);
    Tensor* y = pad_ext(f.ctx, x, 2, 1, 1, 1, 0, 0, 0, 0, PAD_REFLECT);  // [7,5]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, small_ramp(12));
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 注意力

TRC_TEST(grad_soft_max) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    tensor_set_param(x);
    Tensor* y = soft_max_ext(f.ctx, x, nullptr, 0.5f, 0.0f);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, y));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {0.1f, -0.4f, 0.7f, 1.2f, -1.0f, 0.3f, 0.6f, -0.2f, 0.9f, -0.5f, 1.5f, 0.0f});
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 损失（交叉熵，M1.5c）

TRC_TEST(grad_cross_entropy) {
    GradFixture f;
    // logits [nc=4, nr=2]，target 为软标签（与 ggml 语义一致：同形状概率分布）
    Tensor* logits = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    tensor_set_param(logits);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    Tensor* loss = cross_entropy_loss(f.ctx, logits, target);
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(target, {0.1f, 0.2f, 0.3f, 0.4f, 0.25f, 0.25f, 0.25f, 0.25f});
    write_f32(logits, {0.1f, -0.4f, 0.7f, 1.2f, -1.0f, 0.3f, 0.6f, -0.2f});
    check_gradients(f, {logits}, loss);
}

TRC_TEST(grad_cross_entropy_mlp) {
    GradFixture f;
    // 两层分类器：mul_mat + 广播 add + relu + 交叉熵（覆盖 CE 反向与上游链式传导）
    Tensor* x  = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);   // [in=3, batch=2]
    Tensor* w1 = new_tensor_2d(f.ctx, TYPE_F32, 3, 4);   // [in, hidden]
    Tensor* b1 = new_tensor_2d(f.ctx, TYPE_F32, 4, 1);   // [hidden, 1]
    Tensor* w2 = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);   // [hidden, classes]
    Tensor* tgt = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);  // [classes, batch]
    tensor_set_param(w1);
    tensor_set_param(b1);
    tensor_set_param(w2);

    Tensor* h = relu(f.ctx, add(f.ctx, mul_mat(f.ctx, w1, x), b1));
    Tensor* logits = mul_mat(f.ctx, w2, h);
    Tensor* loss = cross_entropy_loss(f.ctx, logits, tgt);
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {0.5f, -0.3f, 0.8f, -0.6f, 0.2f, 0.9f});
    write_f32(w1, {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.7f, 0.2f, -0.1f, 0.6f, -0.3f, 0.4f, 0.8f});
    write_f32(b1, {0.1f, -0.2f, 0.3f, 0.05f});
    write_f32(w2, {0.2f, -0.5f, 0.4f, 0.1f, 0.3f, -0.2f, 0.6f, -0.1f, 0.2f, 0.4f, -0.3f, 0.5f});
    write_f32(tgt, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    check_gradients(f, {w1, b1, w2}, loss);
}

// ---------------------------------------------------------------- 形状/视图

TRC_TEST(grad_repeat) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 1);
    tensor_set_param(x);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 2, 4);
    Tensor* loss = sum(f.ctx, repeat(f.ctx, x, target));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f});
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_reshape_transpose_permute) {
    GradFixture f;
    Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 2, 3, 4);
    tensor_set_param(x);
    Tensor* r = reshape_2d(f.ctx, x, 6, 4);
    Tensor* t = cont(f.ctx, transpose(f.ctx, r));           // [4,6] 连续
    Tensor* p = permute(f.ctx, t, 1, 0, 2, 3);               // 再置换回来 [6,4]
    Tensor* loss = sum(f.ctx, sqr(f.ctx, p));
    tensor_set_loss(loss);
    f.build(loss);

    std::vector<float> vx(24);
    for (int i = 0; i < 24; ++i) {
        vx[(size_t) i] = (float) (i + 1) * 0.3f;
    }
    write_f32(x, vx);
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_view_acc) {
    GradFixture f;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 4);  // [3,4]
    tensor_set_param(x);
    // 取第 1..2 行构成的视图 [3,2]（offset = 1 行 = 12 字节）
    Tensor* v = view_2d(f.ctx, x, 3, 2, 12);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, v));
    tensor_set_loss(loss);
    f.build(loss);

    std::vector<float> vx(12);
    for (int i = 0; i < 12; ++i) {
        vx[(size_t) i] = (float) (i + 1) * 0.5f;
    }
    write_f32(x, vx);
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 累加

TRC_TEST(grad_accumulation_shared_input) {
    GradFixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 4);
    tensor_set_param(x);
    // x 被 sqr 与 mul 同时使用：dL/dx = 2x + 2x = 4x
    Tensor* loss = add(f.ctx, sum(f.ctx, sqr(f.ctx, x)), sum(f.ctx, mul(f.ctx, x, x)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {-1.0f, 0.5f, 2.0f, -3.0f});
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 梯度累积（M1.5d）

// 累积等价性：先算 A 再算 B 的累积梯度 = gA + gB
TRC_TEST(grad_accumulate_equivalence) {
    GradFixture f(1 << 22);
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* W = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    tensor_set_param(W);
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
    Tensor* t = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
    Tensor* loss = mse_loss(f.ctx, mul_mat(f.ctx, W, x), t);
    tensor_set_loss(loss);
    f.build(loss);

    if (const Tensor* bad = trc_test::first_unsupported_node(f.g, f.dev)) {
        std::printf("    跳过：设备不支持算子 %s\n", op_name(bad->op));
        return;
    }

    const std::vector<float> xa = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f};
    const std::vector<float> ta = {0.5f, -0.5f, 0.0f, 1.0f, -1.0f, 0.25f};
    const std::vector<float> xb = {-0.5f, 1.0f, 2.0f, -3.0f, 0.75f, -0.25f};
    const std::vector<float> tb = {0.0f, 0.5f, -1.0f, 1.5f, -0.5f, 0.0f};

    // gA / gB（各自独立的一步）
    graph_reset(f.g);
    write_f32(x, xa);
    write_f32(t, ta);
    f.dev->graph_compute(f.g);
    const std::vector<float> gA = read_f32(graph_get_grad(f.g, W));

    graph_reset(f.g);
    write_f32(x, xb);
    write_f32(t, tb);
    f.dev->graph_compute(f.g);
    const std::vector<float> gB = read_f32(graph_get_grad(f.g, W));

    // 累积：一次 graph_reset 清零后连续两次 compute，中间只播种
    graph_reset(f.g);
    write_f32(x, xa);
    write_f32(t, ta);
    f.dev->graph_compute(f.g);
    graph_reset_accumulate(f.g);
    write_f32(x, xb);
    write_f32(t, tb);
    f.dev->graph_compute(f.g);
    const std::vector<float> gAB = read_f32(graph_get_grad(f.g, W));

    TRC_EXPECT(gA.size() == 4 && gB.size() == 4 && gAB.size() == 4);
    for (size_t i = 0; i < gAB.size(); ++i) {
        TRC_EXPECT_NEAR(gAB[i], (double) gA[i] + (double) gB[i], 1e-6);
    }
}

// 损失缩放 1/K 的 K 个 micro-batch 累积梯度 = 全批量均值梯度
TRC_TEST(grad_accumulate_scaled_matches_full) {
    GradFixture f(1 << 22);
    if (!grad_device_ok(f)) {
        return;
    }
    // 全批量：4 个样本
    Tensor* W1 = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    tensor_set_param(W1);
    Tensor* x1 = new_tensor_2d(f.ctx, TYPE_F32, 2, 4);
    Tensor* t1 = new_tensor_2d(f.ctx, TYPE_F32, 2, 4);
    Tensor* loss_full = mse_loss(f.ctx, mul_mat(f.ctx, W1, x1), t1);
    tensor_set_loss(loss_full);

    // micro-batch：2 个样本，损失缩放 0.5（K=2）
    Tensor* W2 = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    tensor_set_param(W2);
    Tensor* x2 = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    Tensor* t2 = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    Tensor* loss_micro = scale(f.ctx, mse_loss(f.ctx, mul_mat(f.ctx, W2, x2), t2), 0.5f);
    tensor_set_loss(loss_micro);

    Graph* g2 = graph_new(f.ctx);
    graph_build_forward_expand(f.ctx, f.g, loss_full);
    graph_build_forward_expand(f.ctx, g2, loss_micro);
    graph_build_backward_expand(f.g);
    graph_build_backward_expand(g2);
    f.buf = buffer_alloc_ctx_tensors(f.ctx, f.dev->default_buffer_type());

    for (Graph* g : {f.g, g2}) {
        if (const Tensor* bad = trc_test::first_unsupported_node(g, f.dev)) {
            std::printf("    跳过：设备不支持算子 %s\n", op_name(bad->op));
            graph_free(g2);
            return;
        }
    }

    const std::vector<float> w0 = {0.1f, -0.2f, 0.3f, 0.4f};
    write_f32(W1, w0);
    write_f32(W2, w0);

    // ne=[2, N]：元素 (i0,i1) 的偏移 = i0 + 2*i1
    const std::vector<float> xf = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> tf = {0.5f, -0.5f, 1.0f, 0.0f, -1.0f, 0.5f, 0.25f, -0.25f};
    const std::vector<float> xa(xf.begin(), xf.begin() + 4);
    const std::vector<float> ta(tf.begin(), tf.begin() + 4);
    const std::vector<float> xb(xf.begin() + 4, xf.end());
    const std::vector<float> tb(tf.begin() + 4, tf.end());

    graph_reset(f.g);
    write_f32(x1, xf);
    write_f32(t1, tf);
    f.dev->graph_compute(f.g);
    const std::vector<float> g_full = read_f32(graph_get_grad(f.g, W1));

    graph_reset(g2);
    write_f32(x2, xa);
    write_f32(t2, ta);
    f.dev->graph_compute(g2);
    graph_reset_accumulate(g2);
    write_f32(x2, xb);
    write_f32(t2, tb);
    f.dev->graph_compute(g2);
    const std::vector<float> g_acc = read_f32(graph_get_grad(g2, W2));

    TRC_EXPECT(g_full.size() == 4 && g_acc.size() == 4);
    for (size_t i = 0; i < g_full.size(); ++i) {
        TRC_EXPECT_NEAR(g_acc[i], g_full[i], 1e-6);
    }

    graph_free(g2);
}

// 多图共享守卫（M2.1e / R4）不得误伤重建流程：graph_free/graph_clear 后同一参数可再次构建反向
TRC_TEST(grad_param_reuse_after_graph_release) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }

    Tensor* w = new_tensor_1d(f.ctx, TYPE_F32, 3);
    tensor_set_param(w);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, w));
    tensor_set_loss(loss);

    // 第一张图：构建反向后释放
    Graph* g1 = graph_new(f.ctx);
    graph_build_forward_expand(f.ctx, g1, loss);
    graph_build_backward_expand(g1);
    graph_free(g1);

    // 第二张图：同一参数再次构建反向（应成功）
    Graph* g2 = graph_new(f.ctx);
    graph_build_forward_expand(f.ctx, g2, loss);
    graph_build_backward_expand(g2);
    TRC_EXPECT(graph_get_grad(g2, w) != nullptr);

    // graph_clear 后同一 Graph 对象可重新构建（应成功）
    graph_clear(g2);
    graph_build_forward_expand(f.ctx, g2, loss);
    graph_build_backward_expand(g2);
    TRC_EXPECT(graph_get_grad(g2, w) != nullptr);

    graph_free(g2);
}

// ---------------------------------------------------------------- 索引/形状/卷积（M1.3b）

TRC_TEST(grad_get_rows) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* table = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);  // [ne0=4, n_vocab=3]
    Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
    tensor_set_param(table);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, get_rows(f.ctx, table, idx)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(table, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    write_i32(idx, {2, 0});
    check_gradients(f, {table}, loss);
}

TRC_TEST(grad_concat_pad) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
    Tensor* y = new_tensor_2d(f.ctx, TYPE_F32, 1, 3);
    tensor_set_param(x);
    tensor_set_param(y);
    Tensor* c = concat(f.ctx, x, y, 0);       // [3,3]
    Tensor* p = pad(f.ctx, x, 1, 0, 0, 0);    // [3,3]
    Tensor* loss = add(f.ctx, sum(f.ctx, sqr(f.ctx, c)), sum(f.ctx, sqr(f.ctx, p)));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    write_f32(y, {0.5f, 1.5f, 2.5f});
    check_gradients(f, {x, y}, loss);
}

TRC_TEST(grad_conv_1d) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    // IC=2、N=2：覆盖 im2col_back 1D 的 N/IC 步长写回（IC=1 曾掩盖该缺陷，M1.4d 修复）
    Tensor* kernel = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 2);  // [KW=2, IC=2, OC=2]
    Tensor* image = new_tensor_3d(f.ctx, TYPE_F32, 4, 2, 2);   // [W=4, IC=2, N=2]
    tensor_set_param(kernel);
    tensor_set_param(image);
    Tensor* r = conv_1d(f.ctx, kernel, image, 1, 0, 1);
    TRC_EXPECT_EQ(r->ne[0], 3);  // OW
    TRC_EXPECT_EQ(r->ne[1], 2);  // OC
    TRC_EXPECT_EQ(r->ne[2], 2);  // N
    Tensor* loss = sum(f.ctx, sqr(f.ctx, r));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(kernel, {0.5f, -0.3f, 0.8f, 0.2f, 0.1f, -0.6f, 0.4f, 0.7f});
    write_f32(image, {1.0f, 2.0f, -1.0f, 0.5f, 0.3f, -0.2f, 0.9f, -0.4f,
                      0.6f, -0.8f, 0.25f, 1.5f, -0.7f, 0.45f, -1.2f, 0.35f});
    check_gradients(f, {kernel, image}, loss);
}

TRC_TEST(grad_conv_2d) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    // IC=2：多通道卷积反向（im2col_back 2D + mul_mat 反向组合）
    Tensor* kernel = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 2, 2);  // [KW, KH, IC, OC]
    Tensor* image = new_tensor_4d(f.ctx, TYPE_F32, 3, 3, 2, 1);   // [W, H, IC, N]
    tensor_set_param(kernel);
    tensor_set_param(image);
    Tensor* r = conv_2d(f.ctx, kernel, image, 1, 1, 0, 0, 1, 1);
    TRC_EXPECT_EQ(r->ne[0], 2);  // OW
    TRC_EXPECT_EQ(r->ne[1], 2);  // OH
    Tensor* loss = sum(f.ctx, sqr(f.ctx, r));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(kernel, {0.5f, -0.3f, 0.8f, 0.2f, 0.1f, -0.6f, 0.4f, 0.7f,
                       -0.2f, 0.9f, 0.3f, -0.5f, 0.6f, -0.1f, -0.7f, 0.45f});
    write_f32(image, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                      0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f});
    check_gradients(f, {kernel, image}, loss);
}

TRC_TEST(grad_norm_rms_norm) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    tensor_set_param(x);
    Tensor* loss = add(f.ctx, sum(f.ctx, sqr(f.ctx, norm(f.ctx, x, 1e-5f))),
                       sum(f.ctx, sqr(f.ctx, rms_norm(f.ctx, x, 1e-5f))));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {0.3f, -0.7f, 1.2f, 0.8f, -1.1f, 0.4f, 1.6f, -0.2f, 0.9f, -1.4f, 0.5f, 1.0f});
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_group_norm) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 3, 2, 4);  // [ne0=3, ne1=2, C=4]
    tensor_set_param(x);
    Tensor* loss = sum(f.ctx, sqr(f.ctx, group_norm(f.ctx, x, 2, 1e-5f)));
    tensor_set_loss(loss);
    f.build(loss);

    std::vector<float> vx(24);
    for (int i = 0; i < 24; ++i) {
        vx[(size_t) i] = 0.3f * (float) (i % 7) - 0.8f;
    }
    write_f32(x, vx);
    check_gradients(f, {x}, loss);
}

TRC_TEST(grad_gelu_erf) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 6);
    tensor_set_param(x);
    Tensor* loss = sum(f.ctx, gelu_erf(f.ctx, x));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {-1.5f, -0.6f, -0.1f, 0.4f, 1.1f, 2.0f});
    check_gradients(f, {x}, loss);
}

// ---------------------------------------------------------------- 端到端训练（M1.3c）

// 主机端手写一步 SGD：p -= lr * grad（正式优化器 API 属于 M1.5）
void sgd_update(Tensor* p, Tensor* grad, float lr) {
    const size_t n = (size_t) tensor_nelements(p);
    std::vector<float> v(n);
    tensor_get(p, v.data(), 0, n * sizeof(float));
    std::vector<float> g(n);
    tensor_get(grad, g.data(), 0, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        v[i] -= lr * g[i];
    }
    tensor_set(p, v.data(), 0, n * sizeof(float));
}

// 两层 MLP（linear+relu+linear）+ MSE 组合损失，SGD 走若干步 loss 应显著下降
TRC_TEST(end_to_end_mlp_sgd) {
    GradFixture f(1 << 22);
    if (!grad_device_ok(f)) {
        return;
    }

    constexpr int64_t IN = 2;
    constexpr int64_t HID = 8;
    constexpr int64_t OUT = 1;
    constexpr int64_t BATCH = 16;

    Tensor* W1 = new_tensor_2d(f.ctx, TYPE_F32, IN, HID);     // [in, hidden]
    Tensor* b1 = new_tensor_2d(f.ctx, TYPE_F32, HID, 1);      // 沿 batch 广播的偏置
    Tensor* W2 = new_tensor_2d(f.ctx, TYPE_F32, HID, OUT);    // [hidden, out]
    Tensor* b2 = new_tensor_2d(f.ctx, TYPE_F32, OUT, 1);
    Tensor* x  = new_tensor_2d(f.ctx, TYPE_F32, IN, BATCH);   // [in, batch]
    Tensor* y  = new_tensor_2d(f.ctx, TYPE_F32, OUT, BATCH);  // 目标 [out, batch]

    tensor_set_param(W1);
    tensor_set_param(b1);
    tensor_set_param(W2);
    tensor_set_param(b2);

    // 前向：h = relu(W1·x + b1)；pred = W2·h + b2；loss = MSE(pred, y)
    Tensor* h    = relu(f.ctx, add(f.ctx, mul_mat(f.ctx, W1, x), b1));
    Tensor* pred = add(f.ctx, mul_mat(f.ctx, W2, h), b2);
    Tensor* loss = scale(f.ctx, sum(f.ctx, sqr(f.ctx, sub(f.ctx, pred, y))),
                         1.0f / (float) (OUT * BATCH));
    tensor_set_loss(loss);
    f.build(loss);

    // 确定性伪随机数据（LCG），teacher: y = tanh(2*x0 + x1)
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

    f.step();
    const float loss0 = read_scalar(loss);
    TRC_EXPECT(std::isfinite(loss0));
    TRC_EXPECT(loss0 > 0.0f);

    // 第一步梯度必须有限且非全零
    {
        const std::vector<float> g = read_f32(graph_get_grad(f.g, W1));
        bool nonzero = false;
        for (float v : g) {
            TRC_EXPECT(std::isfinite(v));
            nonzero = nonzero || (std::fabs(v) > 1e-8f);
        }
        TRC_EXPECT(nonzero);
    }

    const float lr = 0.05f;
    const int   steps = 400;
    for (int it = 0; it < steps; ++it) {
        f.step();
        sgd_update(W1, graph_get_grad(f.g, W1), lr);
        sgd_update(b1, graph_get_grad(f.g, b1), lr);
        sgd_update(W2, graph_get_grad(f.g, W2), lr);
        sgd_update(b2, graph_get_grad(f.g, b2), lr);
    }
    f.step();
    const float loss1 = read_scalar(loss);
    std::printf("    MLP 训练: loss %.6g -> %.6g (%d 步, lr=%g)\n", loss0, loss1, steps, lr);
    TRC_EXPECT(std::isfinite(loss1));

    // 参数不得出现 NaN/Inf
    for (Tensor* p : {W1, b1, W2, b2}) {
        const std::vector<float> v = read_f32(p);
        for (float val : v) {
            TRC_EXPECT(std::isfinite(val));
        }
    }

    TRC_EXPECT(loss1 < 0.2f * loss0);
}

// cast 反向：经过 F16 往返的梯度应为全 1（f16 量化噪声不适合中心差分）
TRC_TEST(grad_cast_identity) {
    GradFixture f;
    if (!grad_device_ok(f)) {
        return;
    }
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 4);
    tensor_set_param(x);
    Tensor* loss = sum(f.ctx, cast(f.ctx, cast(f.ctx, x, TYPE_F16), TYPE_F32));
    tensor_set_loss(loss);
    f.build(loss);

    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    f.step();

    Tensor* gx = graph_get_grad(f.g, x);
    TRC_EXPECT(gx != nullptr);
    const std::vector<float> grad = read_f32(gx);
    for (float v : grad) {
        TRC_EXPECT_NEAR(v, 1.0f, 1e-6);
    }
}

