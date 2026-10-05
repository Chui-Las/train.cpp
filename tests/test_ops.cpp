// train.cpp - CPU 后端算子测试（M1.1 基础算子）
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>

using namespace traincpp;

namespace {

// 测试夹具：创建上下文、CPU buffer、计算图
struct Fixture {
    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Buffer*  buf = nullptr;
    Graph*   graph = nullptr;

    explicit Fixture(size_t mem = 1 << 20) {
        ctx = context_new(mem);
        dev = device_cpu();
        graph = graph_new(ctx);
    }

    ~Fixture() {
        graph_free(graph);
        if (buf != nullptr) {
            buffer_free(buf);
        }
        context_free(ctx);
    }

    // 为已有张量分配后端内存（必须在所有张量创建之后调用）
    void alloc() {
        buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    }

    void compute(Tensor* out) {
        graph_build_forward_expand(ctx, graph, out);
        dev->graph_compute(graph);
    }
};

} // namespace

TRC_TEST(cpu_device_registered) {
    Device* dev = device_cpu();
    TRC_EXPECT(dev != nullptr);
    TRC_EXPECT(dev->props().type == DeviceType::CPU);
    TRC_EXPECT(device_count() >= 1);
}

TRC_TEST(cpu_add_same_shape) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* r = add(f.ctx, a, b);
    f.alloc();

    const float va[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float vb[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], va[i] + vb[i], 1e-6);
    }
}

TRC_TEST(cpu_add_broadcast) {
    Fixture f;
    // a: [3, 2]，b: [3]（沿第 1 维广播）
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* r = add(f.ctx, a, b);
    TRC_EXPECT_EQ(r->ne[0], 3);
    TRC_EXPECT_EQ(r->ne[1], 2);
    f.alloc();

    const float va[6] = {1, 2, 3, 4, 5, 6};
    const float vb[3] = {10, 20, 30};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    const float expected[6] = {11, 22, 33, 14, 25, 36};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_mul_sub_div_chain) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* r = div(f.ctx, mul(f.ctx, sub(f.ctx, a, b), a), b);
    f.alloc();

    const float va[3] = {4.0f, 6.0f, 8.0f};
    const float vb[3] = {2.0f, 3.0f, 4.0f};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    float out[3] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 3; ++i) {
        const float expected = (va[i] - vb[i]) * va[i] / vb[i];
        TRC_EXPECT_NEAR(out[i], expected, 1e-5);
    }
}

TRC_TEST(cpu_unary_ops) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* y_sig = sigmoid(f.ctx, a);
    Tensor* y_relu = relu(f.ctx, a);
    Tensor* y_silu = silu(f.ctx, a);
    Tensor* y_tanh = tanh_op(f.ctx, a);
    Tensor* y_neg = neg(f.ctx, a);
    f.alloc();

    const float va[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(y_sig);
    f.compute(y_relu);
    f.compute(y_silu);
    f.compute(y_tanh);
    f.compute(y_neg);

    float out[4] = {0};
    tensor_get(y_sig, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], 1.0 / (1.0 + std::exp(-va[i])), 1e-6);
    }

    tensor_get(y_relu, out, 0, sizeof(out));
    const float exp_relu[4] = {0, 0, 1, 2};
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], exp_relu[i], 1e-6);
    }

    tensor_get(y_silu, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], va[i] / (1.0 + std::exp(-va[i])), 1e-6);
    }

    tensor_get(y_tanh, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], std::tanh(va[i]), 1e-6);
    }

    tensor_get(y_neg, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], -va[i], 1e-6);
    }
}

TRC_TEST(cpu_scale) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* r = scale(f.ctx, a, 2.5f);
    f.alloc();

    const float va[3] = {1.0f, -2.0f, 4.0f};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(r);

    float out[3] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 3; ++i) {
        TRC_EXPECT_NEAR(out[i], va[i] * 2.5f, 1e-6);
    }
}

TRC_TEST(cpu_mul_mat) {
    Fixture f;
    // a [k=2, m=2]：列主序（ne0 最内层）
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    // b [k=2, n=3]
    Tensor* b = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
    Tensor* r = mul_mat(f.ctx, a, b);
    TRC_EXPECT_EQ(r->ne[0], 2);
    TRC_EXPECT_EQ(r->ne[1], 3);
    f.alloc();

    // a(k,i)：a = [[1,3],[2,4]] 转置后存储布局 [k + i*2]
    const float va[4] = {1, 2, 3, 4};
    // b(k,j)：b = [[1,3,5],[2,4,6]]
    const float vb[6] = {1, 2, 3, 4, 5, 6};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    // r(i,j) = sum_k a(k,i)*b(k,j)
    const float expected[6] = {
        1 * 1 + 2 * 2,  // r(0,0)=5
        3 * 1 + 4 * 2,  // r(1,0)=11
        1 * 3 + 2 * 4,  // r(0,1)=11
        3 * 3 + 4 * 4,  // r(1,1)=25
        1 * 5 + 2 * 6,  // r(0,2)=17
        3 * 5 + 4 * 6,  // r(1,2)=39
    };
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

// mul_mat batch：a 的平面广播到 b 的平面（i2 % a2）
TRC_TEST(cpu_mul_mat_batch) {
    Fixture f;
    Tensor* a = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 1);  // [k=2, m=2, 1, 1]
    Tensor* b = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 2, 1);  // [k=2, n=2, b2=2, 1]
    Tensor* r = mul_mat(f.ctx, a, b);
    TRC_EXPECT_EQ(r->ne[0], 2);
    TRC_EXPECT_EQ(r->ne[1], 2);
    TRC_EXPECT_EQ(r->ne[2], 2);
    TRC_EXPECT_EQ(r->ne[3], 1);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};              // a(k,i)
    const float vb[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // b(k,j,i2)：平面 0 = [1,2,3,4]，平面 1 = [5,6,7,8]
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    // 平面 0 与 2D 基础用例一致；平面 1 使用 b 的第二平面（a 广播）
    const float expected[8] = {5, 11, 11, 25, 17, 39, 23, 53};
    float out[8] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 8; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

// mul_mat batch 整除广播：a2=2、b2=4 → 平面 i2 用 a 的平面 (i2 % 2)
TRC_TEST(cpu_mul_mat_batch_broadcast_div) {
    Fixture f;
    Tensor* a = new_tensor_4d(f.ctx, TYPE_F32, 1, 1, 2, 1);  // [k=1, m=1, a2=2, 1]
    Tensor* b = new_tensor_4d(f.ctx, TYPE_F32, 1, 1, 2, 2);  // [k=1, n=1, b2=2, b3=2]
    Tensor* r = mul_mat(f.ctx, a, b);
    TRC_EXPECT_EQ(r->ne[0], 1);
    TRC_EXPECT_EQ(r->ne[1], 1);
    TRC_EXPECT_EQ(r->ne[2], 2);
    TRC_EXPECT_EQ(r->ne[3], 2);
    f.alloc();

    const float va[2] = {1, 2};        // a(k,i,i2)
    const float vb[4] = {1, 2, 3, 4};  // b(k,j,i2,i3)
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    // 平面 (i2,i3)：a 取 (i2%2, i3%1)=i2%2，b 取 (i2,i3)
    const float expected[4] = {1 * 1, 2 * 2, 1 * 3, 2 * 4};
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

TRC_TEST(cpu_recompute) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* r = add(f.ctx, a, b);
    f.alloc();

    const float v1[2] = {1.0f, 2.0f};
    const float v2[2] = {3.0f, 4.0f};
    tensor_set(a, v1, 0, sizeof(v1));
    tensor_set(b, v2, 0, sizeof(v2));

    f.compute(r);
    float out[2] = {0};
    tensor_get(r, out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out[0], 4.0, 1e-6);
    TRC_EXPECT_NEAR(out[1], 6.0, 1e-6);

    // 修改输入后重新计算同一图
    const float v3[2] = {10.0f, 20.0f};
    tensor_set(b, v3, 0, sizeof(v3));
    f.dev->graph_compute(f.graph);
    tensor_get(r, out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out[0], 11.0, 1e-6);
    TRC_EXPECT_NEAR(out[1], 22.0, 1e-6);
}

// ---------------- 视图与形状 ----------------

TRC_TEST(cpu_view_2d_and_cont) {
    Fixture f;
    // a: [4, 3]，数据 0..11（行主序，ne0 最内层）
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    // 子矩阵：列 2..3（k=2,3），行 0..1（j=0,1）；沿用源步长 nb=[4,16]
    Tensor* v = view_2d(f.ctx, a, 2, 2, 2 * sizeof(float));
    // 物化连续拷贝
    Tensor* c = cont(f.ctx, v);
    TRC_EXPECT(c != v);  // v 非连续，必须生成 cont 节点

    f.alloc();

    float va[12];
    for (int i = 0; i < 12; ++i) {
        va[i] = (float) i;
    }
    tensor_set(a, va, 0, sizeof(va));

    // 视图张量不是连续张量（nb[1]=16 != 8）
    TRC_EXPECT(!tensor_is_contiguous(v));
    TRC_EXPECT_EQ(v->nb[0], 4);
    TRC_EXPECT_EQ(v->nb[1], 16);

    // 说明：tensor_get 与 ggml 一致是“原始缓冲区字节”语义，非连续视图需先 cont 物化
    f.compute(c);
    float cout[4] = {0};
    tensor_get(c, cout, 0, sizeof(cout));
    // 视图元素 (k=0,1; j=0,1)：源下标 (2+k) + 4*j -> 2,3,6,7
    const float v_expected[4] = {2, 3, 6, 7};
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(cout[i], v_expected[i], 1e-6);
    }
    TRC_EXPECT(tensor_is_contiguous(c));
}

TRC_TEST(cpu_reshape_and_transpose) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);  // 数据 1..6
    Tensor* rs = reshape_2d(f.ctx, a, 2, 3);           // 连续，可直接共享数据
    Tensor* tr = transpose(f.ctx, a);                  // ne=[2,3]，非连续
    Tensor* trc = cont(f.ctx, tr);

    TRC_EXPECT(tensor_is_contiguous(rs));
    TRC_EXPECT_EQ(rs->ne[0], 2);
    TRC_EXPECT_EQ(rs->ne[1], 3);
    TRC_EXPECT_EQ(rs->nb[1], 8);
    TRC_EXPECT_EQ(tr->ne[0], 2);
    TRC_EXPECT_EQ(tr->ne[1], 3);
    TRC_EXPECT_EQ(tr->nb[0], a->nb[1]);  // 4*3=12
    TRC_EXPECT_EQ(tr->nb[1], a->nb[0]);  // 4

    f.alloc();

    const float va[6] = {1, 2, 3, 4, 5, 6};
    tensor_set(a, va, 0, sizeof(va));

    // 说明：转置视图是非连续的，用 cont 物化后验证：
    // 源下标 (j) + 3*(i) -> [1,4,2,5,3,6]
    f.compute(trc);
    float cout[6] = {0};
    tensor_get(trc, cout, 0, sizeof(cout));
    const float t_expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(cout[i], t_expected[i], 1e-6);
    }

    // reshape 是连续视图，可直接按字节读取
    float rout[6] = {0};
    tensor_get(rs, rout, 0, sizeof(rout));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(rout[i], va[i], 1e-6);
    }
}

TRC_TEST(cpu_view_arithmetic) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    Tensor* v0 = view_2d(f.ctx, a, 2, 2, 0);                    // 行 0..1 的前 2 列
    Tensor* v1 = view_2d(f.ctx, a, 2, 2, 2 * sizeof(float));    // 行 0..1 的后 2 列
    Tensor* r = add(f.ctx, v0, v1);

    f.alloc();

    const float va[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(r);
    // v0 = [1,2,5,6]，v1 = [3,4,7,8] -> r = [4,6,12,14]
    const float expected[4] = {4, 6, 12, 14};
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_cast) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* h = cast(f.ctx, a, TYPE_F16);
    Tensor* back = cast(f.ctx, h, TYPE_F32);
    TRC_EXPECT_EQ(h->type, TYPE_F16);
    TRC_EXPECT_EQ(h->nb[0], 2);
    f.alloc();

    // 这些值在 f16 中可精确表示
    const float va[4] = {1.5f, -2.25f, 0.5f, 3.0f};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(h);
    f.compute(back);

    float out[4] = {0};
    tensor_get(back, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], va[i], 1e-6);
    }
}

// ---------------- 逐元素补充算子 ----------------

TRC_TEST(cpu_add1_and_clamp) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* s = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* r = add1(f.ctx, a, s);
    Tensor* c = clamp(f.ctx, r, -0.5f, 1.5f);
    f.alloc();

    const float va[4] = {-2.0f, -1.0f, 0.0f, 1.0f};
    const float vs[1] = {0.5f};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(s, vs, 0, sizeof(vs));

    f.compute(c);
    const float expected[4] = {-0.5f, -0.5f, 0.5f, 1.5f};
    float out[4] = {0};
    tensor_get(c, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_leaky_relu) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 5);
    Tensor* r = leaky_relu(f.ctx, a, 0.1f);
    f.alloc();

    const float va[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(r);
    const float expected[5] = {-0.2f, -0.1f, 0.0f, 1.0f, 2.0f};
    float out[5] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 5; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_activation_extra) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* g = gelu(f.ctx, a);
    Tensor* ge = gelu_erf(f.ctx, a);
    Tensor* sp = softplus(f.ctx, a);
    Tensor* hs = hardswish(f.ctx, a);
    Tensor* si = sin_op(f.ctx, a);
    Tensor* co = cos_op(f.ctx, a);
    f.alloc();

    const float one = 1.0f;
    tensor_set(a, &one, 0, sizeof(one));

    const float eps = 1e-5f;
    float out = 0;
    f.compute(g);
    tensor_get(g, &out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out, 0.841192f, eps);

    f.compute(ge);
    tensor_get(ge, &out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out, 0.8413447f, eps);

    f.compute(sp);
    tensor_get(sp, &out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out, 1.3132617f, eps);

    f.compute(hs);
    tensor_get(hs, &out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out, 0.6666667f, eps);

    f.compute(si);
    tensor_get(si, &out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out, 0.8414710f, eps);

    f.compute(co);
    tensor_get(co, &out, 0, sizeof(out));
    TRC_EXPECT_NEAR(out, 0.5403023f, eps);
}

// ---------------- 归约 ----------------

TRC_TEST(cpu_reductions) {
    Fixture f;
    // a: [3, 2] -> 数据 [1..6]
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* s = sum(f.ctx, a);
    Tensor* sr = sum_rows(f.ctx, a);
    Tensor* m = mean(f.ctx, a);
    TRC_EXPECT_EQ(s->ne[0], 1);
    TRC_EXPECT_EQ(s->ne[1], 1);
    TRC_EXPECT_EQ(sr->ne[0], 1);
    TRC_EXPECT_EQ(sr->ne[1], 2);
    TRC_EXPECT_EQ(m->ne[0], 1);
    TRC_EXPECT_EQ(m->ne[1], 2);
    f.alloc();

    const float va[6] = {1, 2, 3, 4, 5, 6};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(s);
    f.compute(sr);
    f.compute(m);

    float sv = 0;
    tensor_get(s, &sv, 0, sizeof(sv));
    TRC_EXPECT_NEAR(sv, 21.0, 1e-5);

    float srv[2] = {0};
    tensor_get(sr, srv, 0, sizeof(srv));
    TRC_EXPECT_NEAR(srv[0], 6.0, 1e-5);
    TRC_EXPECT_NEAR(srv[1], 15.0, 1e-5);

    float mv[2] = {0};
    tensor_get(m, mv, 0, sizeof(mv));
    TRC_EXPECT_NEAR(mv[0], 2.0, 1e-5);
    TRC_EXPECT_NEAR(mv[1], 5.0, 1e-5);
}

// ---------------- 归一化 ----------------

TRC_TEST(cpu_norm_and_rms_norm) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* n = norm(f.ctx, a, 1e-5f);
    Tensor* r = rms_norm(f.ctx, a, 1e-5f);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(n);
    f.compute(r);

    // LayerNorm: mean=2.5, var=1.25
    const float mean = 2.5f;
    const float var = 1.25f;
    const float scale = 1.0f / std::sqrt(var + 1e-5f);
    float nout[4] = {0};
    tensor_get(n, nout, 0, sizeof(nout));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(nout[i], (va[i] - mean) * scale, 1e-5);
    }

    // RMSNorm: ms=(1+4+9+16)/4=7.5
    const float rms_scale = 1.0f / std::sqrt(7.5f + 1e-5f);
    float rout[4] = {0};
    tensor_get(r, rout, 0, sizeof(rout));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(rout[i], va[i] * rms_scale, 1e-5);
    }
}

TRC_TEST(cpu_group_norm) {
    Fixture f;
    // [ne0=2, ne1=2, ne2=2]：两个通道，每组一个通道
    Tensor* a = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 2);
    Tensor* g = group_norm(f.ctx, a, 2, 1e-5f);
    f.alloc();

    // 通道 0：1,2,3,4；通道 1：10,12,14,16（数据布局 i0 + 2*i1 + 4*i2）
    const float va[8] = {1, 2, 3, 4, 10, 12, 14, 16};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(g);

    const float m0 = 2.5f;
    const float v0 = 1.25f;
    const float s0 = 1.0f / std::sqrt(v0 + 1e-5f);
    const float m1 = 13.0f;
    const float v1 = 5.0f;
    const float s1 = 1.0f / std::sqrt(v1 + 1e-5f);

    float out[8] = {0};
    tensor_get(g, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], (va[i] - m0) * s0, 1e-5);
    }
    for (int i = 4; i < 8; ++i) {
        TRC_EXPECT_NEAR(out[i], (va[i] - m1) * s1, 1e-5);
    }
}

// ---------------- soft_max ----------------

TRC_TEST(cpu_soft_max_basic) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* y = soft_max(f.ctx, a);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};
    tensor_set(a, va, 0, sizeof(va));
    f.compute(y);

    double sum = 0;
    for (int i = 0; i < 4; ++i) {
        sum += std::exp((double) va[i] - 4.0);
    }
    float out[4] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], std::exp((double) va[i] - 4.0) / sum, 1e-5);
    }
}

TRC_TEST(cpu_soft_max_scale_and_mask) {
    Fixture f;
    // a: [4, 2]；mask: [4, 2]（行级 mask）
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    Tensor* mask = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    Tensor* y = soft_max_ext(f.ctx, a, mask, 1.0f, 0.0f);
    f.alloc();

    const float va[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const float vm[8] = {0, 10, 0, 10, 10, 0, 10, 0};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(mask, vm, 0, sizeof(vm));

    f.compute(y);

    // 行 0：{0,10,0,10} -> softmax
    const float row0[4] = {0, 10, 0, 10};
    double sum0 = 0;
    for (int i = 0; i < 4; ++i) {
        sum0 += std::exp((double) row0[i] - 10.0);
    }
    // 行 1：a=1 时 {11,1,11,1} -> softmax
    const float row1[4] = {11, 1, 11, 1};
    double sum1 = 0;
    for (int i = 0; i < 4; ++i) {
        sum1 += std::exp((double) row1[i] - 11.0);
    }

    float out[8] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], std::exp((double) row0[i] - 10.0) / sum0, 1e-5);
    }
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[4 + i], std::exp((double) row1[i] - 11.0) / sum1, 1e-5);
    }
}

// ---------------- 索引 ----------------

TRC_TEST(cpu_get_rows) {
    Fixture f;
    // 表 [ne0=3, ne1=4]（4 行，每行 3 个元素）
    Tensor* table = new_tensor_2d(f.ctx, TYPE_F32, 3, 4);
    Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
    Tensor* r = get_rows(f.ctx, table, idx);
    TRC_EXPECT_EQ(r->ne[0], 3);
    TRC_EXPECT_EQ(r->ne[1], 2);
    f.alloc();

    float vt[12];
    for (int i = 0; i < 12; ++i) {
        vt[i] = (float) (i + 1);
    }
    const int32_t vi[2] = {3, 0};
    tensor_set(table, vt, 0, sizeof(vt));
    tensor_set(idx, vi, 0, sizeof(vi));

    f.compute(r);

    // 第 3 行 {10,11,12} + 第 0 行 {1,2,3}
    const float expected[6] = {10, 11, 12, 1, 2, 3};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_get_rows_3d) {
    Fixture f;
    // 表 [2, 3, 2]（词汇 3，第二组维 2）；索引 [2, 2]（b.ne1 == table.ne2）
    Tensor* table = new_tensor_3d(f.ctx, TYPE_F32, 2, 3, 2);
    Tensor* idx = new_tensor_2d(f.ctx, TYPE_I32, 2, 2);
    Tensor* r = get_rows(f.ctx, table, idx);
    TRC_EXPECT_EQ(r->ne[0], 2);
    TRC_EXPECT_EQ(r->ne[1], 2);
    TRC_EXPECT_EQ(r->ne[2], 2);
    f.alloc();

    // table(i0,i1,i2) = 1 + i0 + 2*i1 + 6*i2
    float vt[12];
    for (int i = 0; i < 12; ++i) {
        vt[i] = (float) (i + 1);
    }
    const int32_t vi[4] = {1, 2, 0, 1};  // b(i10,i11)
    tensor_set(table, vt, 0, sizeof(vt));
    tensor_set(idx, vi, 0, sizeof(vi));

    f.compute(r);

    // r(i0,i10,i11) = table(i0, b(i10,i11), i11)
    const float expected[8] = {3, 4, 5, 6, 7, 8, 9, 10};
    float out[8] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 8; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_get_rows_transposed_table) {
    Fixture f;
    // 连续存储 base [4,3]（ne=[4,3]），转置视图 table=[3,4]（dim=3, vocab=4）：
    // table(d, r) = base(r, d) = r + 4*d（行内步长 table->nb[0] = 4 个 float）
    Tensor* base  = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    Tensor* table = transpose(f.ctx, base);
    Tensor* idx   = new_tensor_1d(f.ctx, TYPE_I32, 2);
    Tensor* r     = get_rows(f.ctx, table, idx);
    TRC_EXPECT_EQ(r->ne[0], 3);
    TRC_EXPECT_EQ(r->ne[1], 2);
    f.alloc();

    float vb[12];
    for (int i = 0; i < 12; ++i) {
        vb[i] = (float) i;
    }
    const int32_t vi[2] = {2, 0};
    tensor_set(base, vb, 0, sizeof(vb));
    tensor_set(idx, vi, 0, sizeof(vi));

    f.compute(r);

    // 行 2 -> [2,6,10]，行 0 -> [0,4,8]（修复前 CPU 会沿连续内存读成 [2,3,4]/[0,1,2]）
    const float expected[6] = {2, 6, 10, 0, 4, 8};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_set_rows) {
    Fixture f;
    // 目标 [3, 4]：4 行、每行 3 个元素
    Tensor* dest = new_tensor_2d(f.ctx, TYPE_F32, 3, 4);
    Tensor* values = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
    Tensor* r = set_rows(f.ctx, dest, values, idx);
    TRC_EXPECT(r->view_src == dest);
    f.alloc();

    const float vd[12] = {0};
    const float vv[6] = {1, 2, 3, 4, 5, 6};
    const int32_t vi[2] = {3, 0};
    tensor_set(dest, vd, 0, sizeof(vd));
    tensor_set(values, vv, 0, sizeof(vv));
    tensor_set(idx, vi, 0, sizeof(vi));

    f.compute(r);

    // 行 3 <- {1,2,3}；行 0 <- {4,5,6}
    const float expected[12] = {4, 5, 6, 0, 0, 0, 0, 0, 0, 1, 2, 3};
    float out[12] = {0};
    tensor_get(dest, out, 0, sizeof(out));
    for (int i = 0; i < 12; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

// ---------------- 形状扩展 ----------------

TRC_TEST(cpu_repeat) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
    Tensor* r = repeat(f.ctx, a, target);
    TRC_EXPECT_EQ(r->ne[0], 2);
    TRC_EXPECT_EQ(r->ne[1], 3);
    f.alloc();

    const float va[2] = {1, 2};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(r);

    const float expected[6] = {1, 2, 1, 2, 1, 2};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_concat) {
    Fixture f;
    // dim=1 拼接：[2,2] + [2,1] -> [2,3]
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    Tensor* b = new_tensor_2d(f.ctx, TYPE_F32, 2, 1);
    Tensor* r = concat(f.ctx, a, b, 1);
    TRC_EXPECT_EQ(r->ne[0], 2);
    TRC_EXPECT_EQ(r->ne[1], 3);
    // dim=0 拼接：[2] + [3] -> [5]
    Tensor* a1 = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* b1 = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* r1 = concat(f.ctx, a1, b1, 0);
    TRC_EXPECT_EQ(r1->ne[0], 5);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};
    const float vb[2] = {5, 6};
    const float va1[2] = {7, 8};
    const float vb1[3] = {9, 10, 11};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));
    tensor_set(a1, va1, 0, sizeof(va1));
    tensor_set(b1, vb1, 0, sizeof(vb1));

    f.compute(r);
    f.compute(r1);

    const float expected[6] = {1, 2, 3, 4, 5, 6};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }

    const float expected1[5] = {7, 8, 9, 10, 11};
    float out1[5] = {0};
    tensor_get(r1, out1, 0, sizeof(out1));
    for (int i = 0; i < 5; ++i) {
        TRC_EXPECT_NEAR(out1[i], expected1[i], 1e-6);
    }
}

TRC_TEST(cpu_pad) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    Tensor* r = pad(f.ctx, a, 1, 1, 0, 0);  // ne0: +1, ne1: +1
    TRC_EXPECT_EQ(r->ne[0], 3);
    TRC_EXPECT_EQ(r->ne[1], 3);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(r);

    // 数据布局 i0 + 3*i1：[1,2,0, 3,4,0, 0,0,0]
    const float expected[9] = {1, 2, 0, 3, 4, 0, 0, 0, 0};
    float out[9] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 9; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

// pad_ext 左填充补零（M2.3c）：x=[1,2,3] -> lp=1, rp=2 -> [0,1,2,3,0,0]
TRC_TEST(cpu_pad_left) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* r = pad_ext(f.ctx, a, 1, 2, 0, 0, 0, 0, 0, 0, PAD_ZERO);
    TRC_EXPECT_EQ(r->ne[0], 6);
    f.alloc();

    const float va[3] = {1, 2, 3};
    tensor_set(a, va, 0, sizeof(va));
    f.compute(r);

    const float expected[6] = {0, 1, 2, 3, 0, 0};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

// pad reflect 1D（M2.3c）：x=[1,2,3,4] -> lp=2, rp=1 -> [3,2,1,2,3,4,3]
TRC_TEST(cpu_pad_reflect_1d) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* r = pad_ext(f.ctx, a, 2, 1, 0, 0, 0, 0, 0, 0, PAD_REFLECT);
    TRC_EXPECT_EQ(r->ne[0], 7);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};
    tensor_set(a, va, 0, sizeof(va));
    f.compute(r);

    const float expected[7] = {3, 2, 1, 2, 3, 4, 3};
    float out[7] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 7; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

// pad reflect 2D 左右/上下非对称：x [W=3,H=2] -> lp0=1,rp0=1,lp1=1,rp1=0
TRC_TEST(cpu_pad_reflect_2d) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* r = pad_ext(f.ctx, a, 1, 1, 1, 0, 0, 0, 0, 0, PAD_REFLECT);
    TRC_EXPECT_EQ(r->ne[0], 5);
    TRC_EXPECT_EQ(r->ne[1], 3);
    f.alloc();

    const float va[6] = {1, 2, 3, 4, 5, 6};  // [W,H]：h=0 行 {1,2,3}，h=1 行 {4,5,6}
    tensor_set(a, va, 0, sizeof(va));
    f.compute(r);

    // fw: 0->1,1->0,2->1,3->2,4->1；fh: 0->1,1->0,2->1
    const float expected[15] = {5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5};
    float out[15] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 15; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

// ---------------- 卷积 ----------------

TRC_TEST(cpu_im2col_1d) {
    Fixture f;
    // 核 [KW=2, IC=1, OC=1]（只用到形状）；输入 [W=4, IC=1, N=1]
    Tensor* k = new_tensor_3d(f.ctx, TYPE_F32, 2, 1, 1);
    Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 4, 1, 1);
    Tensor* r = im2col(f.ctx, k, x, 1, 0, 0, 0, 1, 0, false, TYPE_F32);
    TRC_EXPECT_EQ(r->ne[0], 2);   // IC*KW
    TRC_EXPECT_EQ(r->ne[1], 3);   // OW
    TRC_EXPECT_EQ(r->ne[2], 1);   // N
    f.alloc();

    const float vx[4] = {1, 2, 3, 4};
    const float vk[2] = {0, 0};
    tensor_set(x, vx, 0, sizeof(vx));
    tensor_set(k, vk, 0, sizeof(vk));

    f.compute(r);

    // 每个输出位置一列 [x[i], x[i+1]]
    const float expected[6] = {1, 2, 2, 3, 3, 4};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_conv_1d) {
    Fixture f;
    // 核 [KW=2, IC=1, OC=1] = {2,3}；输入 [W=4, IC=1, N=1] = {1,2,3,4}
    Tensor* k = new_tensor_3d(f.ctx, TYPE_F32, 2, 1, 1);
    Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 4, 1, 1);
    Tensor* r = conv_1d(f.ctx, k, x, 1, 0, 1);
    TRC_EXPECT_EQ(r->ne[0], 3);  // OW
    TRC_EXPECT_EQ(r->ne[1], 1);  // OC
    TRC_EXPECT_EQ(r->ne[2], 1);  // N
    f.alloc();

    const float vk[2] = {2, 3};
    const float vx[4] = {1, 2, 3, 4};
    tensor_set(k, vk, 0, sizeof(vk));
    tensor_set(x, vx, 0, sizeof(vx));

    f.compute(r);

    const float expected[3] = {8, 13, 18};
    float out[3] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 3; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

// conv_1d 批量回归（N=2）：ggml 组合的 reshape 顺序在 N>1 时结果错误（本库已修正）
TRC_TEST(cpu_conv_1d_batch) {
    Fixture f;
    Tensor* k = new_tensor_3d(f.ctx, TYPE_F32, 2, 1, 1);
    Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 4, 1, 2);
    Tensor* r = conv_1d(f.ctx, k, x, 1, 0, 1);
    TRC_EXPECT_EQ(r->ne[0], 3);  // OW
    TRC_EXPECT_EQ(r->ne[1], 1);  // OC
    TRC_EXPECT_EQ(r->ne[2], 2);  // N
    f.alloc();

    const float vk[2] = {2, 3};
    const float vx[8] = {1, 2, 3, 4, 10, 20, 30, 40};
    tensor_set(k, vk, 0, sizeof(vk));
    tensor_set(x, vx, 0, sizeof(vx));

    f.compute(r);

    // batch0: {8,13,18}；batch1: {80,130,180}
    const float expected[6] = {8, 13, 18, 80, 130, 180};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

TRC_TEST(cpu_conv_2d) {
    Fixture f;
    // 核 [KW=2, KH=2, IC=1, OC=1] = {1,2,3,4}；输入 [W=3, H=3, IC=1, N=1] = 1..9
    Tensor* k = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 1);
    Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 3, 3, 1, 1);
    Tensor* r = conv_2d(f.ctx, k, x, 1, 1, 0, 0, 1, 1);
    TRC_EXPECT_EQ(r->ne[0], 2);  // OW
    TRC_EXPECT_EQ(r->ne[1], 2);  // OH
    TRC_EXPECT_EQ(r->ne[2], 1);  // OC
    TRC_EXPECT_EQ(r->ne[3], 1);  // N
    f.alloc();

    const float vk[4] = {1, 2, 3, 4};  // k(kw,kh)：k(0,0)=1,k(1,0)=2,k(0,1)=3,k(1,1)=4
    float vx[9];
    for (int i = 0; i < 9; ++i) {
        vx[i] = (float) (i + 1);
    }
    tensor_set(k, vk, 0, sizeof(vk));
    tensor_set(x, vx, 0, sizeof(vx));

    f.compute(r);

    // out(iow,ioh) = sum_{kw,kh} x(iow+kw, ioh+kh) * k(kw,kh)
    const float expected[4] = {37, 47, 67, 77};
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-4);
    }
}

TRC_TEST(cpu_col2im_1d) {
    Fixture f;
    // cols [K*OC=2, T_in=3]，OC=1，s=1，p=0 -> T_out=4
    Tensor* cols = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
    Tensor* r = col2im_1d(f.ctx, cols, 1, 1, 0);
    TRC_EXPECT_EQ(r->ne[0], 4);
    TRC_EXPECT_EQ(r->ne[1], 1);
    f.alloc();

    const float vc[6] = {1, 2, 3, 4, 5, 6};  // 列 t: {c0,c1}
    tensor_set(cols, vc, 0, sizeof(vc));

    f.compute(r);

    const float expected[4] = {1, 5, 9, 6};
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

TRC_TEST(cpu_conv_transpose_1d) {
    Fixture f;
    // 单通道：核 [K=2, Cout=1, Cin=1] = {2,3}；输入 [T_in=3, Cin=1] = {1,2,3}；s=2
    Tensor* k1 = new_tensor_3d(f.ctx, TYPE_F32, 2, 1, 1);
    Tensor* x1 = new_tensor_2d(f.ctx, TYPE_F32, 3, 1);
    Tensor* r1 = conv_transpose_1d(f.ctx, k1, x1, 2, 0, 1);
    TRC_EXPECT_EQ(r1->ne[0], 6);  // (3-1)*2 + 2
    TRC_EXPECT_EQ(r1->ne[1], 1);
    f.alloc();

    const float vk1[2] = {2, 3};
    const float vx1[3] = {1, 2, 3};
    tensor_set(k1, vk1, 0, sizeof(vk1));
    tensor_set(x1, vx1, 0, sizeof(vx1));

    f.compute(r1);

    const float expected1[6] = {2, 3, 4, 6, 6, 9};
    float out1[6] = {0};
    tensor_get(r1, out1, 0, sizeof(out1));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out1[i], expected1[i], 1e-5);
    }
}

TRC_TEST(cpu_sgn_step) {
    Fixture f;
    Tensor* a = new_tensor_1d(f.ctx, TYPE_F32, 5);
    Tensor* s = sgn(f.ctx, a);
    Tensor* t = step(f.ctx, a);
    f.alloc();

    const float va[5] = {-2.0f, -0.0f, 0.0f, 1.5f, -3.0f};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(s);
    f.compute(t);

    const float es[5] = {-1.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    const float et[5] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float os[5] = {0};
    float ot[5] = {0};
    tensor_get(s, os, 0, sizeof(os));
    tensor_get(t, ot, 0, sizeof(ot));
    for (int i = 0; i < 5; ++i) {
        TRC_EXPECT_NEAR(os[i], es[i], 1e-6);
        TRC_EXPECT_NEAR(ot[i], et[i], 1e-6);
    }
}

TRC_TEST(cpu_repeat_back) {
    Fixture f;
    // a [3,1] 先 repeat 到 [3,4]，再 repeat_back 回 [3,1]：结果应为 4 * a
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 1);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 3, 4);
    Tensor* rep = repeat(f.ctx, a, target);
    Tensor* rb = repeat_back(f.ctx, rep, a);
    TRC_EXPECT_EQ(rb->ne[0], 3);
    TRC_EXPECT_EQ(rb->ne[1], 1);
    f.alloc();

    const float va[3] = {1, 2, 3};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(rb);

    const float expected[3] = {4, 8, 12};
    float out[3] = {0};
    tensor_get(rb, out, 0, sizeof(out));
    for (int i = 0; i < 3; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_acc) {
    Fixture f;
    // target [4,2] = 1..8；把 b [2,2] 累加到 offset=4 字节、行步长 nb1=16 字节的区域
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    Tensor* b = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    Tensor* r = acc(f.ctx, target, b, 16, 0, 0, 4, false);
    TRC_EXPECT(tensor_are_same_shape(r, target));
    f.alloc();

    const float vt[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float vb[4] = {10, 20, 30, 40};
    tensor_set(target, vt, 0, sizeof(vt));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    // 行 i1=0 的列 i0=1,2 加 b 的第 0 行；行 i1=1 的列 i0=1,2 加 b 的第 1 行
    const float expected[8] = {1, 12, 23, 4, 5, 36, 47, 8};
    float out[8] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 8; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }

    // 非 inplace：target 本身不被修改
    float tgt[8] = {0};
    tensor_get(target, tgt, 0, sizeof(tgt));
    for (int i = 0; i < 8; ++i) {
        TRC_EXPECT_NEAR(tgt[i], vt[i], 1e-6);
    }
}

TRC_TEST(cpu_acc_inplace_3d) {
    Fixture f;
    // M1.5d 回归：inplace acc 的 op_params 布局（int32[5]）与区域偏移、
    // 3D 张量的 nb2（旧实现把 inplace 标志写在字节 8，会覆盖 nb2 低字节）
    Tensor* target = new_tensor_3d(f.ctx, TYPE_F32, 2, 3, 4);  // nb1=8、nb2=24 字节
    Tensor* b = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 2);
    const size_t region_off = (size_t) 1 * target->nb[1] + (size_t) 1 * target->nb[2];
    Tensor* r = acc(f.ctx, target, b, target->nb[1], target->nb[2], target->nb[3], region_off, true);
    TRC_EXPECT(tensor_are_same_shape(r, target));  // 与 ggml 一致：结果为 a 的完整视图
    f.alloc();

    std::vector<float> vt(24);
    for (int i = 0; i < 24; ++i) {
        vt[(size_t) i] = (float) i;
    }
    const float vb[8] = {100, 101, 102, 103, 104, 105, 106, 107};
    tensor_set(target, vt.data(), 0, vt.size() * sizeof(float));
    tensor_set(b, vb, 0, sizeof(vb));

    f.compute(r);

    // 期望：region 起点 (i0=0,i1=1,i2=1)，把 b 的 [2,2,2] 累加到 target 对应位置
    std::vector<float> expected = vt;
    for (int i2 = 0; i2 < 2; ++i2) {
        for (int i1 = 0; i1 < 2; ++i1) {
            for (int i0 = 0; i0 < 2; ++i0) {
                const size_t dst_i = (size_t) ((i2 + 1) * 3 + (i1 + 1)) * 2 + (size_t) i0;
                const size_t src_i = (size_t) (i2 * 2 + i1) * 2 + (size_t) i0;
                expected[dst_i] += vb[src_i];
            }
        }
    }

    float out[24] = {0};
    tensor_get(r, out, 0, sizeof(out));  // r 是 target 的完整视图，同时验证原地修改
    for (int i = 0; i < 24; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[(size_t) i], 1e-6);
    }
}

TRC_TEST(cpu_get_rows_back) {
    Fixture f;
    // 表 [4,3]，索引 {2,0}；梯度 [4,2] scatter-add 回表
    Tensor* grad = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
    Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
    Tensor* table = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
    Tensor* r = get_rows_back(f.ctx, grad, idx, table);
    TRC_EXPECT(tensor_are_same_shape(r, table));
    f.alloc();

    const float vg[8] = {1, 2, 3, 4, 10, 20, 30, 40};  // 列 0 = 1..4，列 1 = 10..40
    const int32_t vi[2] = {2, 0};
    tensor_set(grad, vg, 0, sizeof(vg));
    tensor_set(idx, vi, 0, sizeof(vi));

    f.compute(r);

    // 行 2 += 列 0；行 0 += 列 1
    const float expected[12] = {10, 20, 30, 40, 0, 0, 0, 0, 1, 2, 3, 4};
    float out[12] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 12; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_im2col_back_1d) {
    Fixture f;
    // 1D：K=2、IC=1、W=4、s=1、p=0 -> OW=3；梯度 [2,3,1,1] 反传到输入 [4,1,1,1]
    Tensor* grad = new_tensor_4d(f.ctx, TYPE_F32, 2, 3, 1, 1);
    Tensor* kernel = new_tensor_3d(f.ctx, TYPE_F32, 2, 1, 1);
    const int64_t ne[4] = {4, 1, 1, 1};
    Tensor* r = im2col_back(f.ctx, grad, kernel, ne, 1, 0, 0, 0, 1, 0, false);
    TRC_EXPECT_EQ(r->ne[0], 4);
    f.alloc();

    const float vk[2] = {1, 1};
    // 列布局 [ik, iow]：iow=0: (1,2)；iow=1: (3,4)；iow=2: (5,6)
    const float vg[6] = {1, 2, 3, 4, 5, 6};
    tensor_set(kernel, vk, 0, sizeof(vk));
    tensor_set(grad, vg, 0, sizeof(vg));

    f.compute(r);

    const float expected[4] = {1, 5, 9, 6};
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_im2col_back_1d_multichannel) {
    Fixture f;
    // 回归（M1.4d Vulkan conformance 暴露）：1D 且 IC>1/N>1 时，N/IC 的写回步长必须是 nb2/nb1
    // （同 ggml 的 ofs0/ofs1）；此前误用 elem_offset(dst, iiw, iih, iic, in) 会写到 dim2/dim3，
    // 既结果错误又越界写。K=2、IC=2、W=4、s=1、p=0 -> OW=3、N=2
    Tensor* grad = new_tensor_4d(f.ctx, TYPE_F32, 4, 3, 2, 1);  // [feat=IC*KW=4, OW=3, N=2, 1]
    Tensor* kernel = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 1);  // [KW=2, IC=2, OC=1]（仅取形状）
    const int64_t ne[4] = {4, 2, 2, 1};                        // 图像 [W=4, IC=2, N=2, 1]
    Tensor* r = im2col_back(f.ctx, grad, kernel, ne, 1, 0, 0, 0, 1, 0, false);
    TRC_EXPECT_EQ(r->ne[0], 4);
    TRC_EXPECT_EQ(r->ne[1], 2);
    TRC_EXPECT_EQ(r->ne[2], 2);
    f.alloc();

    // grad[f, iow, n] = f*100 + iow*10 + n
    float vg[24] = {0};
    for (int fq = 0; fq < 4; ++fq) {
        for (int ow = 0; ow < 3; ++ow) {
            for (int n = 0; n < 2; ++n) {
                vg[fq + ow * 4 + n * 12] = (float) (fq * 100 + ow * 10 + n);
            }
        }
    }
    tensor_set(grad, vg, 0, sizeof(vg));

    f.compute(r);

    const float expected[16] = {0,   110, 130, 120, 200, 510, 530, 320,
                                1,   112, 132, 121, 201, 512, 532, 321};
    float       out[16]      = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 16; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_cast_roundtrip_2d) {
    Fixture f;
    // [7,3] 上的 F32→F16→F32 往返（M1.4e Vulkan conformance 暴露：ramp 含 0.0，
    // 触发 fp16_to_fp32 对零/次正规数的解码缺陷，见 trc_float.h）
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 7, 3);
    Tensor* h = cast(f.ctx, x, TYPE_F16);
    Tensor* y = cast(f.ctx, h, TYPE_F32);
    TRC_EXPECT_EQ(y->ne[0], 7);
    TRC_EXPECT_EQ(y->ne[1], 3);
    f.alloc();

    float in[21];
    for (int i = 0; i < 21; ++i) {
        in[i] = (float) ((i % 17) - 8) * 0.25f;
    }
    tensor_set(x, in, 0, sizeof(in));

    f.compute(y);

    float out[21] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 21; ++i) {
        TRC_EXPECT_NEAR(out[i], in[i], 1e-3);
    }
}

TRC_TEST(cpu_cast_f16_subnormal) {
    Fixture f;
    // F16 零/次正规数/最小正规数/最大值的 F32→F16→F32 解码回归：
    // 曾因 magic_mask 移位写错（<<22 而非 <<23）使 0x0000 解码为 -0.5
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 8);
    Tensor* y = cast(f.ctx, cast(f.ctx, x, TYPE_F16), TYPE_F32);
    f.alloc();

    const float in[8] = {0.0f, -0.0f, 5.9604645e-08f, 1.0e-05f, 6.1035156e-05f,
                         1.0f, -1.0f, 65504.0f};
    tensor_set(x, in, 0, sizeof(in));

    f.compute(y);

    float out[8] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 8; ++i) {
        // F16 精度：次正规数绝对误差 ≤ 2^-25（≈3e-8），其余相对误差 ≤ 2^-11
        const double tol = 1e-7 + 5e-4 * std::fabs((double) in[i]);
        TRC_EXPECT_NEAR(out[i], in[i], tol);
    }
}

TRC_TEST(cpu_transpose_outer_singleton) {
    Fixture f;
    // 回归：源张量外层维为 1（[4,1]）时，转置后的 ne[1]=4 不能被截断（曾因 n_dims 用源维度而丢维）
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 4, 1);
    Tensor* t = transpose(f.ctx, a);
    TRC_EXPECT_EQ(t->ne[0], 1);
    TRC_EXPECT_EQ(t->ne[1], 4);
    Tensor* c = cont(f.ctx, t);
    f.alloc();

    const float va[4] = {1, 2, 3, 4};
    tensor_set(a, va, 0, sizeof(va));

    f.compute(c);

    float out[4] = {0};
    tensor_get(c, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], va[i], 1e-6);
    }
}

TRC_TEST(cpu_erf) {
    Fixture f;
    Tensor* x = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* r = erf_op(f.ctx, x);
    f.alloc();

    const float vx[3] = {-1.0f, 0.0f, 1.0f};
    tensor_set(x, vx, 0, sizeof(vx));
    f.compute(r);

    const float expected[3] = {-0.8427008f, 0.0f, 0.8427008f};
    float out[3] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 3; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

TRC_TEST(cpu_norm_back) {
    Fixture f;
    Tensor* dy = new_tensor_2d(f.ctx, TYPE_F32, 4, 1);
    Tensor* x  = new_tensor_2d(f.ctx, TYPE_F32, 4, 1);
    Tensor* r  = norm_back(f.ctx, dy, x, 1e-5f);
    f.alloc();

    const float vx[4]  = {-1.0f, 0.0f, 1.0f, 2.0f};
    const float vdy[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_set(x, vx, 0, sizeof(vx));
    tensor_set(dy, vdy, 0, sizeof(vdy));

    f.compute(r);

    // 测试端用 double 复算：mean=0.5，var=1.25，rstd=1/sqrt(var+eps)
    const double eps = 1e-5;
    const double mean = (-1.0 + 0.0 + 1.0 + 2.0) / 4.0;
    const double rstd = 1.0 / std::sqrt(1.25 + eps);
    const double xc[4] = {vx[0] - mean, vx[1] - mean, vx[2] - mean, vx[3] - mean};
    const double sum_dy = 10.0;
    const double sum_dyxc = 1.0 * -1.5 + 2.0 * -0.5 + 3.0 * 0.5 + 4.0 * 1.5;  // 5
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        const double dx = rstd * (vdy[i] - sum_dy / 4.0 - xc[i] * rstd * rstd * sum_dyxc / 4.0);
        TRC_EXPECT_NEAR(out[i], dx, 1e-5);
    }
}

TRC_TEST(cpu_rms_norm_back) {
    Fixture f;
    Tensor* dz = new_tensor_2d(f.ctx, TYPE_F32, 4, 1);
    Tensor* x  = new_tensor_2d(f.ctx, TYPE_F32, 4, 1);
    Tensor* r  = rms_norm_back(f.ctx, dz, x, 1e-5f);
    f.alloc();

    const float vx[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    const float vdz[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_set(x, vx, 0, sizeof(vx));
    tensor_set(dz, vdz, 0, sizeof(vdz));

    f.compute(r);

    // 与 ggml 公式一致：rrms=1/sqrt(sum_xx/n+eps)，scale_x=-sum_xdz/(sum_xx+eps*n)
    const double eps = 1e-5;
    const double sum_xx = 30.0;
    const double sum_xdz = 30.0;
    const double rrms = 1.0 / std::sqrt(sum_xx / 4.0 + eps);
    const double scale_x = -sum_xdz / (sum_xx + eps * 4.0);
    float out[4] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        const double dx = (vdz[i] + vx[i] * scale_x) * rrms;
        TRC_EXPECT_NEAR(out[i], dx, 1e-5);
    }
}

TRC_TEST(cpu_group_norm_back) {
    Fixture f;
    // x [2,2,2]（通道维 ne2=2，2 组），分组反向等价于对每组 4 个元素做 LayerNorm 反向
    Tensor* dy = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 2);
    Tensor* x  = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 2);
    Tensor* r  = group_norm_back(f.ctx, dy, x, 2, 1e-5f);
    f.alloc();

    const float vx[8]  = {1, 2, 3, 4, 5, 6, 7, 8};  // 通道 0 = 1..4，通道 1 = 5..8
    const float vdy[8] = {1, 2, 3, 4, 4, 3, 2, 1};
    tensor_set(x, vx, 0, sizeof(vx));
    tensor_set(dy, vdy, 0, sizeof(vdy));

    f.compute(r);

    const double eps = 1e-5;
    float out[8] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int c = 0; c < 2; ++c) {
        double mean = 0.0;
        for (int i = 0; i < 4; ++i) {
            mean += vx[c * 4 + i];
        }
        mean /= 4.0;
        double var = 0.0;
        for (int i = 0; i < 4; ++i) {
            const double d = vx[c * 4 + i] - mean;
            var += d * d;
        }
        var /= 4.0;
        const double rstd = 1.0 / std::sqrt(var + eps);
        double sum_dy = 0.0;
        double sum_dyxc = 0.0;
        for (int i = 0; i < 4; ++i) {
            const double d = vx[c * 4 + i] - mean;
            sum_dy += vdy[c * 4 + i];
            sum_dyxc += vdy[c * 4 + i] * d;
        }
        for (int i = 0; i < 4; ++i) {
            const double d = vx[c * 4 + i] - mean;
            const double dx = rstd * (vdy[c * 4 + i] - sum_dy / 4.0 - d * rstd * rstd * sum_dyxc / 4.0);
            TRC_EXPECT_NEAR(out[c * 4 + i], dx, 1e-5);
        }
    }
}

TRC_TEST(cpu_conv_transpose_1d_multichannel) {
    Fixture f;
    // 核 [K=2, Cout=2, Cin=2]（布局 k + 2*co + 4*ci）
    Tensor* k = new_tensor_3d(f.ctx, TYPE_F32, 2, 2, 2);
    // 输入 [T_in=2, Cin=2]（布局 t + 2*ci）
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
    Tensor* r = conv_transpose_1d(f.ctx, k, x, 1, 0, 1);
    TRC_EXPECT_EQ(r->ne[0], 3);  // (2-1)*1 + 2
    TRC_EXPECT_EQ(r->ne[1], 2);  // Cout
    f.alloc();

    // W[k,co,ci]: k=0: ci=(1,2) co0, (3,4) co1；k=1: ci=(5,6) co0, (7,8) co1
    const float vk[8] = {1, 5, 3, 7, 2, 6, 4, 8};
    // x[t,ci]: t0: (1,3)；t1: (2,4)
    const float vx[4] = {1, 2, 3, 4};
    tensor_set(k, vk, 0, sizeof(vk));
    tensor_set(x, vx, 0, sizeof(vx));

    f.compute(r);

    // out(t,co) = sum_{i,ci} x(i,ci) * W(t-i, co, ci)
    const float expected[6] = {7, 33, 34, 15, 53, 46};
    float out[6] = {0};
    tensor_get(r, out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-4);
    }
}

TRC_TEST(cpu_cross_entropy_loss) {
    Fixture f;
    // logits [nc=3, nr=2]；target 为软标签（每行和为 1）
    Tensor* logits = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* loss   = cross_entropy_loss(f.ctx, logits, target);
    TRC_EXPECT_EQ(loss->ne[0], 1);
    TRC_EXPECT_EQ(loss->type, TYPE_F32);
    f.alloc();

    const float vl[6] = {0, 1, 2, 1, 0, -1};
    const float vt[6] = {0.2f, 0.3f, 0.5f, 1.0f, 0.0f, 0.0f};
    tensor_set(logits, vl, 0, sizeof(vl));
    tensor_set(target, vt, 0, sizeof(vt));

    f.compute(loss);

    // 手算：-1/nr * Σ_row Σ_c t*(logits - lse)，lse = max + log(Σexp(x-max))
    double row0 = 0.0;
    {
        const double lse = 2.0 + std::log(std::exp(-2.0) + std::exp(-1.0) + 1.0);
        row0 = 0.2 * (0.0 - lse) + 0.3 * (1.0 - lse) + 0.5 * (2.0 - lse);
    }
    double row1 = 0.0;
    {
        const double lse = 1.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
        row1 = 1.0 * (1.0 - lse);
    }
    const double expected_loss = -(row0 + row1) / 2.0;

    float got = 0.0f;
    tensor_get(loss, &got, 0, sizeof(got));
    TRC_EXPECT_NEAR(got, expected_loss, 1e-5);
}

TRC_TEST(cpu_cross_entropy_loss_back) {
    Fixture f;
    // grad 标量 = 2.0，检查 dA = (softmax(logits) - target) * grad / nr
    Tensor* grad   = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* logits = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* target = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* dA     = cross_entropy_loss_back(f.ctx, grad, logits, target);
    TRC_EXPECT_EQ(dA->ne[0], 3);
    TRC_EXPECT_EQ(dA->ne[1], 2);
    f.alloc();

    const float vg[1] = {2.0f};
    const float vl[6] = {0, 1, 2, 1, 0, -1};
    const float vt[6] = {0.2f, 0.3f, 0.5f, 1.0f, 0.0f, 0.0f};
    tensor_set(grad, vg, 0, sizeof(vg));
    tensor_set(logits, vl, 0, sizeof(vl));
    tensor_set(target, vt, 0, sizeof(vt));

    f.compute(dA);

    float out[6] = {0};
    tensor_get(dA, out, 0, sizeof(out));
    const double lse[2] = {
        2.0 + std::log(std::exp(-2.0) + std::exp(-1.0) + 1.0),
        1.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0)),
    };
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            const double soft = std::exp(vl[r * 3 + c] - lse[r]);
            const double expected = (soft - vt[r * 3 + c]) * 2.0 / 2.0;  // grad / nr
            TRC_EXPECT_NEAR(out[r * 3 + c], expected, 1e-6);
        }
    }
}

// ---------------------------------------------------------------- supports_op 如实化（M2.1f / R9）

// CPU 仅支持 F32 计算的算子必须如实声明；纯字节/桥接类算子按真实类型集合声明
TRC_TEST(cpu_supports_op_type_constraints) {
    Device*  dev = device_cpu();
    Context* ctx = context_new(1 << 16);

    // F32 逐元素：支持
    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* r = add(ctx, a, b);
    TRC_EXPECT(dev->supports_op(r));

    // F16 逐元素：CPU kernel 仅支持 F32 → 不支持
    Tensor* ah = new_tensor_1d(ctx, TYPE_F16, 4);
    Tensor* bh = new_tensor_1d(ctx, TYPE_F16, 4);
    Tensor* rh = add(ctx, ah, bh);
    TRC_EXPECT(!dev->supports_op(rh));

    // cast：CPU 经 double 桥接支持基本类型互转 → F32→I32 支持
    Tensor* ci = cast(ctx, a, TYPE_I32);
    TRC_EXPECT(dev->supports_op(ci));

    // 整图 pre-flight：包含 F16 节点的图必须报告该节点
    Graph* g = graph_new(ctx);
    graph_build_forward_expand(ctx, g, rh);
    TRC_EXPECT(trc_test::first_unsupported_node(g, dev) == rh);

    // 纯 F32 图：无不支持节点
    Graph* g2 = graph_new(ctx);
    graph_build_forward_expand(ctx, g2, r);
    TRC_EXPECT(trc_test::first_unsupported_node(g2, dev) == nullptr);

    graph_free(g2);
    graph_free(g);
    context_free(ctx);
}

// ---------------------------------------------------------------- 池化 / 2D 转置卷积（M2.3g）

// pool_2d 前向（AVG/MAX）：4x4 输入、2x2 核、步长 2、无填充
TRC_TEST(cpu_pool_2d_avg_max) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 4, 4);  // [W=4, H=4, C=1, N=1]
    Tensor* ya = pool_2d(f.ctx, a, POOL_AVG, 2, 2, 2, 2, 0, 0);
    Tensor* ym = pool_2d(f.ctx, a, POOL_MAX, 2, 2, 2, 2, 0, 0);
    TRC_EXPECT_EQ(ya->ne[0], 2);
    TRC_EXPECT_EQ(ya->ne[1], 2);
    TRC_EXPECT_EQ(ya->ne[2], 1);
    f.alloc();

    float v[16];
    for (int i = 0; i < 16; ++i) {
        v[i] = (float) (i + 1);  // 布局 w + 4*h
    }
    tensor_set(a, v, 0, sizeof(v));

    f.compute(ya);
    f.compute(ym);

    // 窗口：{1,2,5,6} {3,4,7,8} {9,10,13,14} {11,12,15,16}（ow 最快）
    const float exp_avg[4] = {3.5f, 5.5f, 11.5f, 13.5f};
    const float exp_max[4] = {6.0f, 8.0f, 14.0f, 16.0f};
    float out[4] = {0};
    tensor_get(ya, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], exp_avg[i], 1e-6);
    }
    tensor_get(ym, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], exp_max[i], 1e-6);
    }
}

// pool_2d AVG + padding：分母恒为 k0*k1（越界按 0 计入，与 ggml 一致）
TRC_TEST(cpu_pool_2d_avg_pad) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 3, 3);  // [W=3, H=3]
    Tensor* y = pool_2d(f.ctx, a, POOL_AVG, 2, 2, 1, 1, 1, 1);
    TRC_EXPECT_EQ(y->ne[0], 4);  // (3+2-2)/1+1
    TRC_EXPECT_EQ(y->ne[1], 4);
    f.alloc();

    float v[9];
    for (int i = 0; i < 9; ++i) {
        v[i] = (float) (i + 1);  // 布局 w + 3*h
    }
    tensor_set(a, v, 0, sizeof(v));
    f.compute(y);

    // 独立参照（含越界按 0 计入分母）
    float out[16] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int oh = 0; oh < 4; ++oh) {
        for (int ow = 0; ow < 4; ++ow) {
            double sum = 0.0;
            for (int kh = 0; kh < 2; ++kh) {
                for (int kw = 0; kw < 2; ++kw) {
                    const int ih = oh + kh - 1;
                    const int iw = ow + kw - 1;
                    if (ih >= 0 && ih < 3 && iw >= 0 && iw < 3) {
                        sum += (double) v[iw + 3 * ih];
                    }
                }
            }
            TRC_EXPECT_NEAR(out[ow + 4 * oh], sum / 4.0, 1e-6);
        }
    }
}

// pool_2d 反向（专用 kernel）：AVG 均分核面积；MAX 归到首个严格最大值
TRC_TEST(cpu_pool_2d_back_hand) {
    Fixture f;
    Tensor* a = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);  // [2,2]
    Tensor* dy = new_tensor_4d(f.ctx, TYPE_F32, 1, 1, 1, 1);
    Tensor* d_avg = pool_2d_back(f.ctx, dy, a, POOL_AVG, 2, 2, 2, 2, 0, 0);
    Tensor* d_max = pool_2d_back(f.ctx, dy, a, POOL_MAX, 2, 2, 2, 2, 0, 0);
    f.alloc();

    const float va[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float vdy[1] = {1.0f};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(dy, vdy, 0, sizeof(vdy));

    f.compute(d_avg);
    f.compute(d_max);

    const float exp_avg[4] = {0.25f, 0.25f, 0.25f, 0.25f};  // 1/(2*2)
    const float exp_max[4] = {0.0f, 0.0f, 0.0f, 1.0f};       // 最大值在第 4 个（w=1,h=1）
    float out[4] = {0};
    tensor_get(d_avg, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], exp_avg[i], 1e-6);
    }
    tensor_get(d_max, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(out[i], exp_max[i], 1e-6);
    }
}

// conv_transpose_2d 前向：stride=1 手算（W=H=2、K=K=2、单通道）
TRC_TEST(cpu_conv_transpose_2d) {
    Fixture f;
    Tensor* w = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 1);  // [KW,KH,Cout,Cin]
    Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 1);  // [W,H,Cin,N]
    Tensor* y = conv_transpose_2d(f.ctx, w, x, 1);
    TRC_EXPECT_EQ(y->ne[0], 3);  // (2-1)*1+2
    TRC_EXPECT_EQ(y->ne[1], 3);
    TRC_EXPECT_EQ(y->ne[2], 1);
    TRC_EXPECT_EQ(y->ne[3], 1);
    f.alloc();

    const float vw[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float vx[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // 布局 iw + 2*ih
    tensor_set(w, vw, 0, sizeof(vw));
    tensor_set(x, vx, 0, sizeof(vx));
    f.compute(y);

    // out[ow,oh] = Σ_{kw,kh} x[ow-kw, oh-kh]（布局 ow + 3*oh）
    const float expected[9] = {1, 3, 2, 4, 10, 6, 3, 7, 4};
    float out[9] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 9; ++i) {
        TRC_EXPECT_NEAR(out[i], expected[i], 1e-6);
    }
}

// conv_transpose_2d 前向：stride=2（全 1 输入 => 输出 4x4 全 1）
TRC_TEST(cpu_conv_transpose_2d_stride2) {
    Fixture f;
    Tensor* w = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 1);
    Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 1);
    Tensor* y = conv_transpose_2d(f.ctx, w, x, 2);
    TRC_EXPECT_EQ(y->ne[0], 4);  // (2-1)*2+2
    TRC_EXPECT_EQ(y->ne[1], 4);
    f.alloc();

    const float vw[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float vx[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    tensor_set(w, vw, 0, sizeof(vw));
    tensor_set(x, vx, 0, sizeof(vx));
    f.compute(y);

    float out[16] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 16; ++i) {
        TRC_EXPECT_NEAR(out[i], 1.0f, 1e-6);
    }
}

// ---------------- 设备端优化器步（M4.1）----------------

// AdamW 单步（首步）：参数/m/v 与手算一致
TRC_TEST(cpu_opt_step_adamw) {
    Fixture f;
    Tensor* p = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* g = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* m = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* v = new_tensor_1d(f.ctx, TYPE_F32, 4);

    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, lr = 0.01f, wd = 0.0f;
    const double bc1 = 1.0 - std::pow((double) beta1, 1.0);
    const double bc2 = 1.0 - std::pow((double) beta2, 1.0);
    Tensor*      node = opt_step_adamw(f.ctx, p, g, m, v, (float) ((double) lr / bc1),
                                       (float) std::sqrt(bc2), 1.0f - lr * wd, beta1, beta2, eps);
    f.alloc();

    const float vp[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float vg[4] = {0.1f, -0.2f, 0.3f, -0.4f};
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    tensor_set(p, vp, 0, sizeof(vp));
    tensor_set(g, vg, 0, sizeof(vg));
    tensor_set(m, zeros, 0, sizeof(zeros));
    tensor_set(v, zeros, 0, sizeof(zeros));

    f.compute(node);

    float out[4] = {0}, mo[4] = {0}, vo[4] = {0};
    tensor_get(p, out, 0, sizeof(out));
    tensor_get(m, mo, 0, sizeof(mo));
    tensor_get(v, vo, 0, sizeof(vo));
    for (int i = 0; i < 4; ++i) {
        const double gi   = vg[i];
        const double mexp = (gi - 0.0) * (1.0 - beta1);
        const double vexp = 0.0 * beta2 + gi * gi * (1.0 - beta2);
        const double denom = std::sqrt(vexp) / std::sqrt(bc2) + eps;
        const double pexp  = vp[i] * (1.0 - lr * wd) - ((double) lr / bc1) * (mexp / denom);
        TRC_EXPECT_NEAR(mo[i], mexp, 1e-6);
        TRC_EXPECT_NEAR(vo[i], vexp, 1e-6);
        TRC_EXPECT_NEAR(out[i], pexp, 1e-6);
    }
}

// SGD 单步（首步，动量 + Nesterov + 权重衰减）：与手算一致
TRC_TEST(cpu_opt_step_sgd_momentum) {
    Fixture f;
    Tensor* p = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* g = new_tensor_1d(f.ctx, TYPE_F32, 4);
    Tensor* m = new_tensor_1d(f.ctx, TYPE_F32, 4);

    const float lr = 0.1f, mom = 0.9f, damp = 0.0f, wd = 0.5f;
    Tensor*     node = opt_step_sgd(f.ctx, p, g, m, lr, mom, damp, wd, true /*nesterov*/,
                                    true /*is_first*/);
    f.alloc();

    const float vp[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float vg[4] = {0.1f, -0.2f, 0.3f, -0.4f};
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    tensor_set(p, vp, 0, sizeof(vp));
    tensor_set(g, vg, 0, sizeof(vg));
    tensor_set(m, zeros, 0, sizeof(zeros));

    f.compute(node);

    float out[4] = {0}, mo[4] = {0};
    tensor_get(p, out, 0, sizeof(out));
    tensor_get(m, mo, 0, sizeof(mo));
    for (int i = 0; i < 4; ++i) {
        const double d    = vg[i] + wd * vp[i];
        const double mexp = d;                        // 首步直接取 d
        const double dex  = d + mom * mexp;           // Nesterov
        const double pexp = vp[i] - lr * dex;
        TRC_EXPECT_NEAR(mo[i], mexp, 1e-6);
        TRC_EXPECT_NEAR(out[i], pexp, 1e-6);
    }
}

// M4.5：梯度平方和累积（sum_sqr_acc）——acc[0] += Σ a²，跨多个张量顺序累积
TRC_TEST(cpu_sum_sqr_acc) {
    Fixture f;
    Tensor* acc = new_tensor_1d(f.ctx, TYPE_F32, 1);
    Tensor* a1  = new_tensor_1d(f.ctx, TYPE_F32, 3);
    Tensor* a2  = new_tensor_1d(f.ctx, TYPE_F32, 2);
    Tensor* n1  = sum_sqr_acc(f.ctx, acc, a1);
    Tensor* n2  = sum_sqr_acc(f.ctx, acc, a2);
    f.alloc();

    const float init  = 1.0f;                      // 累积初值
    const float v1[3] = {1.0f, 2.0f, 3.0f};        // Σ = 14
    const float v2[2] = {4.0f, 5.0f};              // Σ = 41
    tensor_set(acc, &init, 0, sizeof(init));
    tensor_set(a1, v1, 0, sizeof(v1));
    tensor_set(a2, v2, 0, sizeof(v2));

    f.graph->nodes.push_back(n1);
    f.graph->nodes.push_back(n2);
    f.dev->graph_compute(f.graph);

    float got = 0.0f;
    tensor_get(acc, &got, 0, sizeof(got));
    TRC_EXPECT_NEAR(got, 1.0f + 14.0f + 41.0f, 1e-5);
}

// M4.5：梯度就地裁剪缩放（clip_scale_inplace）：超门限缩放、未超门限不变
TRC_TEST(cpu_clip_scale_inplace) {
    const float va[2] = {3.0f, 4.0f};

    {
        Fixture f;
        Tensor* a    = new_tensor_1d(f.ctx, TYPE_F32, 2);
        Tensor* norm = new_tensor_1d(f.ctx, TYPE_F32, 1);
        Tensor* node = clip_scale_inplace(f.ctx, a, norm, 1.0f, 1e-6f);
        f.alloc();
        const float vnorm = 25.0f;  // total = 5 > max_norm = 1
        tensor_set(a, va, 0, sizeof(va));
        tensor_set(norm, &vnorm, 0, sizeof(vnorm));
        f.compute(node);

        float got[2] = {0};
        tensor_get(a, got, 0, sizeof(got));
        const float scale = 1.0f / (5.0f + 1e-6f);
        TRC_EXPECT_NEAR(got[0], 3.0f * scale, 1e-6);
        TRC_EXPECT_NEAR(got[1], 4.0f * scale, 1e-6);
    }
    {
        Fixture f;
        Tensor* a    = new_tensor_1d(f.ctx, TYPE_F32, 2);
        Tensor* norm = new_tensor_1d(f.ctx, TYPE_F32, 1);
        Tensor* node = clip_scale_inplace(f.ctx, a, norm, 10.0f, 1e-6f);
        f.alloc();
        const float vnorm = 25.0f;  // total = 5 <= max_norm = 10 → scale clamp 到 1
        tensor_set(a, va, 0, sizeof(va));
        tensor_set(norm, &vnorm, 0, sizeof(vnorm));
        f.compute(node);

        float got[2] = {0};
        tensor_get(a, got, 0, sizeof(got));
        TRC_EXPECT_NEAR(got[0], 3.0f, 1e-6);
        TRC_EXPECT_NEAR(got[1], 4.0f, 1e-6);
    }
}

// M4.5：WeightNorm 同步（weightnorm_sync）——g[oc] = sqrt(Σ_rest v²)
TRC_TEST(cpu_weightnorm_sync) {
    Fixture f;
    Tensor* v    = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);  // [n_rest=2, oc=3]
    Tensor* g    = new_tensor_2d(f.ctx, TYPE_F32, 1, 3);
    Tensor* node = weightnorm_sync(f.ctx, v, g);
    f.alloc();

    const float vv[6] = {3.0f, 4.0f, 0.0f, 5.0f, 1.0f, 2.0f};
    tensor_set(v, vv, 0, sizeof(vv));
    f.compute(node);

    float got[3] = {0};
    tensor_get(g, got, 0, sizeof(got));
    TRC_EXPECT_NEAR(got[0], 5.0f, 1e-6);          // sqrt(9+16)
    TRC_EXPECT_NEAR(got[1], 5.0f, 1e-6);          // sqrt(0+25)
    TRC_EXPECT_NEAR(got[2], std::sqrt(5.0f), 1e-6);  // sqrt(1+4)
}
