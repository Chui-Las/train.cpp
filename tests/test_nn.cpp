// train.cpp - nn 层系统测试（M1.5b）
//
// 覆盖：
//   1. 黄金数据对拍：tests/golden/nn_*.npy（scripts/gen_golden.py 生成；缺失时自动跳过）
//      Linear / Embedding / Conv1d / Conv2d / LayerNorm / RMSNorm / GroupNorm
//   2. 有限差分 gradcheck（每层参数梯度，走 autograd）
//   3. 初始化（fan_in/fan_out、kaiming bound、延迟填充、确定性、参数命名）
//   4. Dropout（train/eval、mask 一致性、反向 = mask、确定性）
//   5. 端到端：两层 MLP（Linear+ReLU）+ MSE + AdamW 收敛
#include "npy.h"
#include "test_util.h"
#include "traincpp/traincpp.h"

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

// ---------------- golden meta ----------------

struct NnMeta {
    std::string kind;
    int64_t has_bias = 0;
    int64_t in_features = 0;
    int64_t out_features = 0;
    int64_t num_embeddings = 0;
    int64_t embedding_dim = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t kernel_w = 0;
    int64_t kernel_h = 0;
    int64_t stride0 = 1;
    int64_t stride1 = 1;
    int64_t padding0 = 0;
    int64_t padding1 = 0;
    int64_t dilation0 = 1;
    int64_t dilation1 = 1;
    int64_t num_groups = 1;
    int64_t num_channels = 0;
    int64_t normalized_size = 0;
    float   eps = 1e-5f;
};

bool load_nn_meta(const std::string& path, NnMeta& m) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string key;
    while (f >> key) {
        if (key == "kind") {
            f >> m.kind;
        } else if (key == "has_bias") {
            f >> m.has_bias;
        } else if (key == "in_features") {
            f >> m.in_features;
        } else if (key == "out_features") {
            f >> m.out_features;
        } else if (key == "num_embeddings") {
            f >> m.num_embeddings;
        } else if (key == "embedding_dim") {
            f >> m.embedding_dim;
        } else if (key == "in_channels") {
            f >> m.in_channels;
        } else if (key == "out_channels") {
            f >> m.out_channels;
        } else if (key == "kernel") {
            f >> m.kernel;
        } else if (key == "kernel_w") {
            f >> m.kernel_w;
        } else if (key == "kernel_h") {
            f >> m.kernel_h;
        } else if (key == "stride0") {
            f >> m.stride0;
        } else if (key == "stride1") {
            f >> m.stride1;
        } else if (key == "padding0") {
            f >> m.padding0;
        } else if (key == "padding1") {
            f >> m.padding1;
        } else if (key == "dilation0") {
            f >> m.dilation0;
        } else if (key == "dilation1") {
            f >> m.dilation1;
        } else if (key == "num_groups") {
            f >> m.num_groups;
        } else if (key == "num_channels") {
            f >> m.num_channels;
        } else if (key == "normalized_size") {
            f >> m.normalized_size;
        } else if (key == "eps") {
            f >> m.eps;
        } else {
            std::string rest;
            std::getline(f, rest);
        }
    }
    return true;
}

// npy（C 序）形状 -> 本库 ne（反转）；加载原始字节
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

double compare_f32_tensor(const Tensor* out, const NpyArray& expect, double atol, double rtol) {
    const int64_t n = tensor_nelements(out);
    TRC_EXPECT((int64_t) (expect.raw.size() / sizeof(float)) == n);
    const std::vector<float> got = read_f32(out);
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float exp = ((const float*) expect.raw.data())[(size_t) i];
        const double diff = std::fabs((double) got[(size_t) i] - (double) exp);
        const double tol  = atol + rtol * std::fabs((double) exp);
        if (diff > tol) {
            trc_test::add_failure("nn 对拍: 元素 " + std::to_string(i) + " 实际 " +
                                  std::to_string(got[(size_t) i]) + " 期望 " +
                                  std::to_string(exp));
        }
        worst = std::max(worst, diff);
    }
    return worst;
}

// ---------------- 训练 fixture（gradcheck/端到端用） ----------------

struct NnFixture {
    Context* ctx = nullptr;
    Device*  dev = nullptr;
    Graph*   g   = nullptr;
    Buffer*  buf = nullptr;

    explicit NnFixture(size_t mem = 1 << 22) {
        ctx = context_new(mem);
        dev = trc_test::test_device();
        g   = graph_new(ctx);
    }

    ~NnFixture() {
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
void check_gradients(NnFixture& f, const std::vector<Tensor*>& xs, Tensor* loss, double eps = 1e-2,
                     double tol = 5e-3) {
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
            const float v = base[(size_t) i];
            const float vp = v + (float) eps;
            const float vm = v - (float) eps;
            tensor_set(x, &vp, (size_t) i * sizeof(float), sizeof(float));
            f.step();
            const float lp = read_scalar(loss);
            tensor_set(x, &vm, (size_t) i * sizeof(float), sizeof(float));
            f.step();
            const float lm = read_scalar(loss);
            const double numeric = ((double) lp - (double) lm) / (2.0 * eps);
            const double ana = analytic[(size_t) i];
            TRC_EXPECT_NEAR(numeric, ana, tol * (1.0 + std::fabs(ana)));
            tensor_set(x, &v, (size_t) i * sizeof(float), sizeof(float));
        }
        write_f32(x, base);
        f.step();
    }
}

} // namespace

// ---------------------------------------------------------------- golden 对拍

TRC_TEST(nn_golden_layers) {
    const std::string dir = TRC_GOLDEN_DIR;
    const char* names[] = {"linear_bias", "linear_nobias", "embedding", "conv1d_bias",
                           "conv2d_bias", "layernorm", "rmsnorm", "groupnorm",
                           "convtranspose1d_bias", "convtranspose1d_stride2_pad",
                           "conv1d_grouped2", "conv1d_grouped3",
                           "weightnorm_linear", "weightnorm_conv1d",
                           "weightnorm_conv1d_grouped2"};

    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    int n_ok = 0;
    int n_skip = 0;
    for (const char* name : names) {
        NnMeta meta;
        if (!load_nn_meta(dir + "/nn_" + name + "_meta.txt", meta)) {
            ++n_skip;
            continue;
        }

        Context* ctx = context_new(1 << 22);
        Graph*   g   = graph_new(ctx);

        NpyArray w_npy, b_npy, x_npy, out_npy;
        // weightnorm 用例约定：w 槽位存 v、b 槽位存 g（即使 has_bias=0 也必须加载 b）
        const bool is_weightnorm = meta.kind.rfind("weightnorm_", 0) == 0;
        const bool need_bias = meta.has_bias != 0 || is_weightnorm;
        Tensor* w = load_tensor(ctx, dir + "/nn_" + name + "_w.npy", TYPE_F32, w_npy);
        Tensor* x = load_tensor(ctx, dir + "/nn_" + name + "_x.npy",
                                meta.kind == "embedding" ? TYPE_I32 : TYPE_F32, x_npy);
        Tensor* b = need_bias ? load_tensor(ctx, dir + "/nn_" + name + "_b.npy", TYPE_F32, b_npy)
                              : nullptr;
        if (w == nullptr || x == nullptr || !npy_load(dir + "/nn_" + name + "_out.npy", out_npy) ||
            (need_bias && b == nullptr)) {
            trc_test::add_failure(std::string("nn 对拍: 黄金数据缺失 ") + name);
            graph_free(g);
            context_free(ctx);
            continue;
        }

        Linear lin{};
        Embedding emb{};
        Conv1d c1{};
        Conv2d c2{};
        ConvTranspose1d ct{};
        LayerNorm ln{};
        RmsNorm rn{};
        GroupNorm gn{};
        Tensor* out = nullptr;

        if (meta.kind == "linear") {
            linear_init_weight(ctx, &lin, w, b);
            out = linear_forward(ctx, &lin, x);
        } else if (meta.kind == "embedding") {
            embedding_init_weight(ctx, &emb, w);
            out = embedding_forward(ctx, &emb, x);
        } else if (meta.kind == "conv1d") {
            conv1d_init_weight(ctx, &c1, w, b);
            c1.stride   = meta.stride0;
            c1.padding  = meta.padding0;
            c1.dilation = meta.dilation0;
            out = conv1d_forward(ctx, &c1, x);
        } else if (meta.kind == "conv1d_grouped") {
            c1.groups = meta.num_groups;  // 先设 groups：init_weight 据此推导 in_channels
            conv1d_init_weight(ctx, &c1, w, b);
            c1.stride   = meta.stride0;
            c1.padding  = meta.padding0;
            c1.dilation = meta.dilation0;
            out = conv1d_forward(ctx, &c1, x);
        } else if (meta.kind == "conv2d") {
            conv2d_init_weight(ctx, &c2, w, b);
            c2.stride_w   = meta.stride0;
            c2.stride_h   = meta.stride1;
            c2.padding_w  = meta.padding0;
            c2.padding_h  = meta.padding1;
            c2.dilation_w = meta.dilation0;
            c2.dilation_h = meta.dilation1;
            out = conv2d_forward(ctx, &c2, x);
        } else if (meta.kind == "convtranspose1d") {
            convtranspose1d_init_weight(ctx, &ct, w, b);
            ct.stride  = meta.stride0;
            ct.padding = meta.padding0;
            out = convtranspose1d_forward(ctx, &ct, x);
        } else if (meta.kind == "weightnorm_linear") {
            WeightNorm wn{};
            weightnorm_init(ctx, &wn, w, b);
            Linear lw{};
            lw.has_bias     = false;
            lw.in_features  = w->ne[0];
            lw.out_features = w->ne[1];
            out = linear_forward_weight(ctx, &lw, weightnorm_forward(ctx, &wn), x);
        } else if (meta.kind == "weightnorm_conv1d") {
            WeightNorm wn{};
            weightnorm_init(ctx, &wn, w, b);
            Conv1d cw{};
            cw.has_bias     = false;
            cw.in_channels  = w->ne[1] * meta.num_groups;
            cw.out_channels = w->ne[2];
            cw.groups       = meta.num_groups;  // M3.1：分组 weightnorm 用例
            cw.stride       = meta.stride0;
            cw.padding      = meta.padding0;
            cw.dilation     = meta.dilation0;
            out = conv1d_forward_weight(ctx, &cw, weightnorm_forward(ctx, &wn), x);
        } else if (meta.kind == "layernorm") {
            layernorm_init_weight(ctx, &ln, w, b);
            ln.eps = meta.eps;
            out = layernorm_forward(ctx, &ln, x);
        } else if (meta.kind == "rmsnorm") {
            rmsnorm_init_weight(ctx, &rn, w);
            rn.eps = meta.eps;
            out = rmsnorm_forward(ctx, &rn, x);
        } else if (meta.kind == "groupnorm") {
            groupnorm_init_weight(ctx, &gn, w, b);
            gn.num_groups = meta.num_groups;
            gn.eps        = meta.eps;
            out = groupnorm_forward(ctx, &gn, x);
        } else {
            trc_test::add_failure(std::string("nn 对拍: 未知 kind ") + meta.kind);
            graph_free(g);
            context_free(ctx);
            continue;
        }

        Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
        tensor_set(w, w_npy.raw.data(), 0, w_npy.raw.size());
        tensor_set(x, x_npy.raw.data(), 0, x_npy.raw.size());
        if (b != nullptr) {
            tensor_set(b, b_npy.raw.data(), 0, b_npy.raw.size());
        }
        graph_build_forward_expand(ctx, g, out);

        if (const Tensor* bad = trc_test::first_unsupported_node(g, dev)) {
            std::printf("    [跳过] %-16s 后端不支持 %s\n", name, op_name(bad->op));
            ++n_skip;
            buffer_free(buf);
            graph_free(g);
            context_free(ctx);
            continue;
        }

        dev->graph_compute(g);
        const double worst = compare_f32_tensor(out, out_npy, 1e-4, 1e-3);
        std::printf("    [%s] max_abs %.3g\n", name, worst);
        ++n_ok;

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }

    if (n_ok == 0 && n_skip > 0) {
        std::printf("    跳过：未找到 nn 黄金数据（请先运行 python scripts/gen_golden.py）\n");
    } else {
        std::printf("    nn 黄金对拍：%d 个用例通过（%d 个缺失/跳过）\n", n_ok, n_skip);
    }
}

// ---------------------------------------------------------------- gradcheck

TRC_TEST(nn_gradcheck_linear_embedding) {
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Linear l{};
        Rng rng;
        rng_seed(&rng, 11);
        linear_init(f.ctx, &l, 3, 4, true, &rng, "lin");
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
        Tensor* loss = sum(f.ctx, sqr(f.ctx, linear_forward(f.ctx, &l, x)));
        tensor_set_param(l.weight);
        tensor_set_param(l.bias);
        tensor_set_loss(loss);
        f.build(loss);
        write_f32(x, {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f});
        check_gradients(f, {l.weight, l.bias}, loss);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Embedding e{};
        Rng rng;
        rng_seed(&rng, 12);
        embedding_init(f.ctx, &e, 5, 3, &rng, "emb");
        Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 4);
        Tensor* loss = sum(f.ctx, sqr(f.ctx, embedding_forward(f.ctx, &e, idx)));
        tensor_set_param(e.weight);
        tensor_set_loss(loss);
        f.build(loss);
        const int32_t iv[4] = {0, 2, 1, 4};
        tensor_set(idx, iv, 0, sizeof(iv));
        check_gradients(f, {e.weight}, loss);
    }
}

TRC_TEST(nn_gradcheck_conv) {
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Conv1d c{};
        Rng rng;
        rng_seed(&rng, 21);
        conv1d_init(f.ctx, &c, 2, 2, 3, 1, 1, 1, true, &rng, "c1");
        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 5, 2, 2);  // [W,IC,N]
        Tensor* loss = sum(f.ctx, sqr(f.ctx, conv1d_forward(f.ctx, &c, x)));
        tensor_set_param(c.weight);
        tensor_set_param(c.bias);
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.1f * (float) ((int) i - 7);
        }
        write_f32(x, xv);
        check_gradients(f, {c.weight, c.bias}, loss);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Conv2d c{};
        Rng rng;
        rng_seed(&rng, 22);
        conv2d_init(f.ctx, &c, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1, true, &rng, "c2");
        Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 4, 3, 2, 1);  // [W,H,IC,N]
        Tensor* loss = sum(f.ctx, sqr(f.ctx, conv2d_forward(f.ctx, &c, x)));
        tensor_set_param(c.weight);
        tensor_set_param(c.bias);
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.05f * (float) ((int) i - 10);
        }
        write_f32(x, xv);
        check_gradients(f, {c.weight, c.bias}, loss);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        ConvTranspose1d ct{};
        Rng rng;
        rng_seed(&rng, 23);
        // stride=2 + padding=1（中心裁剪 view）+ N=2（层内切片/concat 批次路径）
        convtranspose1d_init(f.ctx, &ct, 2, 2, 3, 2, 1, 1, true, &rng, "ct1");
        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 5, 2, 2);  // [T,IC,N]
        Tensor* loss = sum(f.ctx, sqr(f.ctx, convtranspose1d_forward(f.ctx, &ct, x)));
        tensor_set_param(x);  // 输入也参与 gradcheck（内部张量梯度）
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.1f * (float) ((int) i - 9);
        }
        write_f32(x, xv);
        check_gradients(f, {ct.weight, ct.bias, x}, loss);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Conv1d c{};
        Rng rng;
        rng_seed(&rng, 24);
        conv1d_init_groups(f.ctx, &c, 4, 4, 3, 1, 1, 1, 2, true, &rng, "cg");
        TRC_EXPECT_EQ(c.groups, 2);
        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 5, 4, 2);  // [W,IC,N]
        Tensor* loss = sum(f.ctx, sqr(f.ctx, conv1d_forward(f.ctx, &c, x)));
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.08f * (float) ((int) i - 11);
        }
        write_f32(x, xv);
        check_gradients(f, {c.weight, c.bias}, loss);
    }
}

// N=1 走核心直接调用、N>1 走层内切片+concat；两条路径结果必须逐位一致
TRC_TEST(nn_convtranspose1d_batch_paths) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }
    ConvTranspose1d ct{};
    ConvTranspose1d ct_nb{};  // 无 bias：用于显式权重前向的等价性/生效性断言
    Rng rng;
    rng_seed(&rng, 41);
    convtranspose1d_init(f.ctx, &ct, 2, 2, 3, 2, 1, 1, true, &rng, "ct");
    convtranspose1d_init(f.ctx, &ct_nb, 2, 2, 3, 2, 1, 1, false, &rng, "ct_nb");
    Tensor* x2 = new_tensor_3d(f.ctx, TYPE_F32, 4, 2, 2);  // [T,IC,N=2]
    Tensor* x1 = new_tensor_3d(f.ctx, TYPE_F32, 4, 2, 1);  // [T,IC,1]：核心直调

    Tensor* y2 = convtranspose1d_forward(f.ctx, &ct, x2);
    Tensor* y1 = convtranspose1d_forward(f.ctx, &ct, x1);
    TRC_EXPECT_EQ(y1->ne[2], 1);
    TRC_EXPECT_EQ(y2->ne[2], 2);
    Tensor* y2_0 = view_4d(f.ctx, y2, y1->ne[0], y1->ne[1], 1, 1, 0);  // 第 0 个样本
    Tensor* d    = sum(f.ctx, sqr(f.ctx, sub(f.ctx, y1, y2_0)));

    // convtranspose1d_forward_weight（M2.4 / §7.8 D.2）：
    //   1) 显式传入层自身权重 → 与 convtranspose1d_forward 逐位一致；
    //   2) 显式传入 2×权重 → 输出恰为 2×（无 bias 时线性；2 为 2 的幂，浮点逐位成立）
    Tensor* y_nb   = convtranspose1d_forward(f.ctx, &ct_nb, x1);
    Tensor* y_nb_w = convtranspose1d_forward_weight(f.ctx, &ct_nb, ct_nb.weight, x1);
    Tensor* y_nb_2x =
        convtranspose1d_forward_weight(f.ctx, &ct_nb, scale(f.ctx, ct_nb.weight, 2.0f), x1);
    Tensor* d_same = sum(f.ctx, sqr(f.ctx, sub(f.ctx, y_nb, y_nb_w)));
    Tensor* d_2x   = sum(f.ctx, sqr(f.ctx, sub(f.ctx, y_nb_2x, add(f.ctx, y_nb, y_nb))));

    graph_build_forward_expand(f.ctx, f.g, d);
    graph_build_forward_expand(f.ctx, f.g, d_same);
    graph_build_forward_expand(f.ctx, f.g, d_2x);
    f.buf = buffer_alloc_ctx_tensors(f.ctx, f.dev->default_buffer_type());

    std::vector<float> vx2((size_t) tensor_nelements(x2));
    for (size_t i = 0; i < vx2.size(); ++i) {
        vx2[i] = 0.1f * (float) ((int) i - 7);
    }
    tensor_set(x2, vx2.data(), 0, vx2.size() * sizeof(float));
    tensor_set(x1, vx2.data(), 0, (size_t) tensor_nelements(x1) * sizeof(float));
    f.dev->graph_compute(f.g);
    const float dval       = read_scalar(d);
    const float dval_same  = read_scalar(d_same);
    const float dval_2x    = read_scalar(d_2x);
    std::printf("    ConvTranspose1d N=1/N=2 路径差 = %.3g；显式权重等价差 = %.3g；2×权重差 = %.3g\n",
                (double) dval, (double) dval_same, (double) dval_2x);
    TRC_EXPECT(dval == 0.0f);
    TRC_EXPECT(dval_same == 0.0f);
    TRC_EXPECT(dval_2x == 0.0f);
}

// 尾部 1 维权重布局（M2.4 验收发现）：tensor_n_dims 为 ggml 折叠语义，最后一维为 1 的合法
// 权重（如 ConvTranspose1d 的 Cin=1、Conv1d/Conv2d 的 OC=1）曾被误拒或误推 fan。
// 本用例覆盖各层"最后一维为 1"的初始化与前向（含显式权重入口）。
TRC_TEST(nn_weight_trailing_one_dims) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }
    Rng rng;
    rng_seed(&rng, 4242);

    // ConvTranspose1d：Cin=1（尾部 1 维形态）；fan_in = K*Cout = 12
    ConvTranspose1d ct{};
    convtranspose1d_init(f.ctx, &ct, 1, 3, 4, 2, 1, 1, true, &rng, "ct1");
    Tensor* x_ct = new_tensor_3d(f.ctx, TYPE_F32, 5, 1, 2);
    Tensor* y_ct = convtranspose1d_forward(f.ctx, &ct, x_ct);
    TRC_EXPECT_EQ(y_ct->ne[0], (5 - 1) * 2 - 2 + 4);
    TRC_EXPECT_EQ(y_ct->ne[1], 3);
    TRC_EXPECT_EQ(y_ct->ne[2], 2);

    // 显式权重入口接受 [K,Cout,Cin=1]（convtranspose1d_init_weight / _forward_weight）
    ConvTranspose1d ct2{};
    ct2.stride   = 2;  // 显式权重入口不设置卷积参数，与 ct 对齐
    ct2.padding  = 1;
    ct2.dilation = 1;
    Tensor* w_ct = new_tensor_3d(f.ctx, TYPE_F32, 4, 3, 1);
    nn_init_uniform(f.ctx, w_ct, -0.1f, 0.1f, &rng);
    convtranspose1d_init_weight(f.ctx, &ct2, w_ct, nullptr);
    Tensor* y_ct2 = convtranspose1d_forward_weight(f.ctx, &ct2, w_ct, x_ct);
    TRC_EXPECT_EQ(y_ct2->ne[0], y_ct->ne[0]);

    // Conv1d：OC=1
    Conv1d c1{};
    conv1d_init(f.ctx, &c1, 2, 1, 3, 1, 1, 1, true, &rng, "c1");
    Tensor* x_c = new_tensor_3d(f.ctx, TYPE_F32, 6, 2, 2);
    Tensor* y_c = conv1d_forward(f.ctx, &c1, x_c);
    TRC_EXPECT_EQ(y_c->ne[1], 1);

    // Conv2d：OC=1
    Conv2d c2{};
    conv2d_init(f.ctx, &c2, 2, 1, 3, 3, 1, 1, 0, 0, 1, 1, true, &rng, "c2");
    Tensor* x_c2 = new_tensor_4d(f.ctx, TYPE_F32, 5, 5, 2, 1);
    Tensor* y_c2 = conv2d_forward(f.ctx, &c2, x_c2);
    TRC_EXPECT_EQ(y_c2->ne[3], 1);

    // Linear：out=1（linear_init_weight 显式入口 [in,1]）
    Linear l{};
    Tensor* w_l = new_tensor_2d(f.ctx, TYPE_F32, 3, 1);
    Tensor* b_l = new_tensor_1d(f.ctx, TYPE_F32, 1);
    nn_init_uniform(f.ctx, w_l, -0.5f, 0.5f, &rng);
    nn_init_constant(f.ctx, b_l, 0.1f);
    linear_init_weight(f.ctx, &l, w_l, b_l);
    Tensor* x_l = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* y_l = linear_forward(f.ctx, &l, x_l);
    TRC_EXPECT_EQ(y_l->ne[0], 1);
    TRC_EXPECT_EQ(y_l->ne[1], 2);

    // Embedding：dim=1（[1,n_vocab]）
    Embedding e{};
    Tensor* w_e = new_tensor_2d(f.ctx, TYPE_F32, 1, 3);
    nn_init_uniform(f.ctx, w_e, -0.5f, 0.5f, &rng);
    embedding_init_weight(f.ctx, &e, w_e);
    Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
    Tensor* y_e = embedding_forward(f.ctx, &e, idx);
    TRC_EXPECT_EQ(y_e->ne[0], 1);
    TRC_EXPECT_EQ(y_e->ne[1], 2);

    graph_build_forward_expand(f.ctx, f.g, y_ct);
    graph_build_forward_expand(f.ctx, f.g, y_ct2);
    graph_build_forward_expand(f.ctx, f.g, y_c);
    graph_build_forward_expand(f.ctx, f.g, y_c2);
    graph_build_forward_expand(f.ctx, f.g, y_l);
    graph_build_forward_expand(f.ctx, f.g, y_e);
    f.buf = buffer_alloc_ctx_tensors(f.ctx, f.dev->default_buffer_type());

    // kaiming bound 回归：fan_in 必须是 K*Cout=12（旧折叠语义会误算为 K=4，界放宽 1.7×）
    {
        const std::vector<float> w     = read_f32(ct.weight);
        const float              bound = 1.0f / std::sqrt(12.0f);
        float                    max_abs = 0.0f;
        bool                     all_zero = true;
        for (float v : w) {
            max_abs  = std::max(max_abs, std::fabs(v));
            all_zero = all_zero && (v == 0.0f);
        }
        TRC_EXPECT(!all_zero);
        TRC_EXPECT(max_abs <= bound + 1e-6f);
    }

    write_f32(x_ct, std::vector<float>((size_t) tensor_nelements(x_ct), 0.1f));
    write_f32(x_c, std::vector<float>((size_t) tensor_nelements(x_c), 0.2f));
    write_f32(x_c2, std::vector<float>((size_t) tensor_nelements(x_c2), 0.3f));
    write_f32(x_l, std::vector<float>((size_t) tensor_nelements(x_l), 0.4f));
    {
        std::vector<int32_t> vi((size_t) tensor_nelements(idx), 0);
        tensor_set(idx, vi.data(), 0, vi.size() * sizeof(int32_t));
    }
    f.dev->graph_compute(f.g);

    for (const Tensor* t : {y_ct, y_ct2, y_c, y_c2, y_l, y_e}) {
        const std::vector<float> v = read_f32(t);
        bool                     ok = true;
        for (float x : v) {
            ok = ok && std::isfinite(x);
        }
        TRC_EXPECT(ok);
    }
    std::printf("    尾部 1 维布局：ConvTranspose1d(Cin=1)/Conv1d(OC=1)/Conv2d(OC=1)/Linear(out=1)/"
                "Embedding(dim=1) 通过\n");
}

// WeightNorm 重参数化对 v/g 的有限差分（Linear 2D 与 Conv1d 3D 两种输出通道布局）
TRC_TEST(nn_gradcheck_weightnorm) {
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Rng rng;
        rng_seed(&rng, 61);
        Tensor* v = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);  // [in,out]
        Tensor* g = new_tensor_2d(f.ctx, TYPE_F32, 1, 3);  // [1,out]
        WeightNorm wn{};
        weightnorm_init(f.ctx, &wn, v, g, "wn");
        nn_init_kaiming_uniform(f.ctx, v, std::sqrt(5.0f), &rng);
        nn_init_constant(f.ctx, g, 0.5f);

        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 2);
        Linear lw{};
        lw.has_bias = false;
        Tensor* loss = sum(f.ctx, sqr(f.ctx, linear_forward_weight(
                                              f.ctx, &lw, weightnorm_forward(f.ctx, &wn), x)));
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.1f * (float) ((int) i - 3);
        }
        write_f32(x, xv);
        check_gradients(f, {v, g}, loss, 0.01, 5e-3);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Rng rng;
        rng_seed(&rng, 62);
        Tensor* v = new_tensor_3d(f.ctx, TYPE_F32, 3, 2, 2);  // [K,IC,OC]
        Tensor* g = new_tensor_3d(f.ctx, TYPE_F32, 1, 1, 2);  // [1,1,OC]
        WeightNorm wn{};
        weightnorm_init(f.ctx, &wn, v, g, "wnc");
        nn_init_kaiming_uniform(f.ctx, v, std::sqrt(5.0f), &rng);
        nn_init_constant(f.ctx, g, 0.5f);

        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 5, 2, 2);  // [W,IC,N]
        Conv1d cw{};
        cw.has_bias = false;
        cw.stride   = 1;
        cw.padding  = 1;
        cw.dilation = 1;
        Tensor* loss = sum(f.ctx, sqr(f.ctx, conv1d_forward_weight(
                                              f.ctx, &cw, weightnorm_forward(f.ctx, &wn), x)));
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.1f * (float) ((int) i - 9);
        }
        write_f32(x, xv);
        check_gradients(f, {v, g}, loss, 0.01, 5e-3);
    }
    {
        // WeightNorm × 分组卷积（M3.1）：v=[K,IC/g,OC]、g=[1,1,OC]，groups=2
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        Rng rng;
        rng_seed(&rng, 63);
        Tensor* v = new_tensor_3d(f.ctx, TYPE_F32, 3, 2, 4);  // [K,IC/g,OC]（IC=4）
        Tensor* g = new_tensor_3d(f.ctx, TYPE_F32, 1, 1, 4);  // [1,1,OC]
        WeightNorm wn{};
        weightnorm_init(f.ctx, &wn, v, g, "wng");
        nn_init_kaiming_uniform(f.ctx, v, std::sqrt(5.0f), &rng);
        nn_init_constant(f.ctx, g, 0.5f);

        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 5, 4, 2);  // [W,IC,N]
        Conv1d cw{};
        cw.has_bias = false;
        cw.groups   = 2;
        cw.stride   = 1;
        cw.padding  = 1;
        cw.dilation = 1;
        Tensor* loss = sum(f.ctx, sqr(f.ctx, conv1d_forward_weight(
                                              f.ctx, &cw, weightnorm_forward(f.ctx, &wn), x)));
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.1f * (float) ((int) i - 9);
        }
        write_f32(x, xv);
        check_gradients(f, {v, g}, loss, 0.01, 5e-3);
    }
}

// weightnorm_sync_g：分配后 g = ||v||（逐输出通道），且前向 w = g·v/||v||
TRC_TEST(nn_weightnorm_sync_g) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }
    constexpr int64_t IN  = 4;
    constexpr int64_t OUT = 3;
    Rng rng;
    rng_seed(&rng, 71);
    Tensor* v = new_tensor_2d(f.ctx, TYPE_F32, IN, OUT);
    Tensor* g = new_tensor_2d(f.ctx, TYPE_F32, 1, OUT);
    WeightNorm wn{};
    weightnorm_init(f.ctx, &wn, v, g, "wn");
    nn_init_kaiming_uniform(f.ctx, v, std::sqrt(5.0f), &rng);
    nn_init_constant(f.ctx, g, 1.0f);  // 先用 1；sync_g 后覆盖为 ||v||

    Tensor* w = weightnorm_forward(f.ctx, &wn);
    graph_build_forward_expand(f.ctx, f.g, w);
    f.buf = buffer_alloc_ctx_tensors(f.ctx, f.dev->default_buffer_type());

    weightnorm_sync_g(&wn);
    f.dev->graph_compute(f.g);

    const std::vector<float> vv = read_f32(v);
    const std::vector<float> gg = read_f32(g);
    const std::vector<float> wv = read_f32(w);
    TRC_EXPECT_EQ((int64_t) gg.size(), OUT);
    for (int64_t o = 0; o < OUT; ++o) {
        double acc = 0.0;
        for (int64_t i = 0; i < IN; ++i) {
            const double x = vv[(size_t) (i + o * IN)];
            acc += x * x;
        }
        const double norm = std::sqrt(acc);
        TRC_EXPECT_NEAR(gg[(size_t) o], norm, 1e-6);
        for (int64_t i = 0; i < IN; ++i) {
            const double expect = (double) gg[(size_t) o] * vv[(size_t) (i + o * IN)] / norm;
            TRC_EXPECT_NEAR(wv[(size_t) (i + o * IN)], expect, 1e-6);
        }
    }
}

// WeightNorm × 分组卷积（M3.1）：与主机端逐元素参照对比（含 stride/padding/多 batch/bias 只加一次），
// 验证 conv1d_forward_weight 的 groups>1 分组映射与 w = g·v/‖v‖ 组合；CPU 与 Vulkan 均执行。
TRC_TEST(nn_weightnorm_grouped_forward_ref) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }
    constexpr int64_t K = 3, IC = 4, OC = 4, GROUPS = 2, W = 6, N = 2, STRIDE = 1, PAD = 1;
    constexpr int64_t IC_G = IC / GROUPS;
    constexpr int64_t OC_G = OC / GROUPS;
    constexpr int64_t OW   = (W + 2 * PAD - K) / STRIDE + 1;

    Tensor* v = new_tensor_3d(f.ctx, TYPE_F32, K, IC_G, OC);  // [K,IC/g,OC]
    Tensor* g = new_tensor_3d(f.ctx, TYPE_F32, 1, 1, OC);     // [1,1,OC]
    WeightNorm wn{};
    weightnorm_init(f.ctx, &wn, v, g, "wng");

    Tensor* bias = new_tensor_1d(f.ctx, TYPE_F32, OC);
    Tensor* x    = new_tensor_3d(f.ctx, TYPE_F32, W, IC, N);
    Conv1d cw{};
    cw.bias         = bias;
    cw.has_bias     = true;
    cw.groups       = GROUPS;
    cw.out_channels = OC;
    cw.stride       = STRIDE;
    cw.padding      = PAD;
    cw.dilation     = 1;

    Tensor* y = conv1d_forward_weight(f.ctx, &cw, weightnorm_forward(f.ctx, &wn), x);
    graph_build_forward_expand(f.ctx, f.g, y);
    f.buf = buffer_alloc_ctx_tensors(f.ctx, f.dev->default_buffer_type());

    // 主机侧确定性数据（索引与 ne 布局一致：index = i0 + ne0*(i1 + ne1*i2)）
    std::vector<float> vv((size_t) (K * IC_G * OC));
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t i = 0; i < IC_G; ++i) {
            for (int64_t o = 0; o < OC; ++o) {
                vv[(size_t) (k + K * (i + IC_G * o))] =
                    0.2f * (float) ((k + 2 * i + 3 * o) % 7 - 3);
            }
        }
    }
    std::vector<float> gg((size_t) OC);
    std::vector<float> bv((size_t) OC);
    for (int64_t o = 0; o < OC; ++o) {
        gg[(size_t) o] = 0.5f + 0.25f * (float) o;
        bv[(size_t) o] = 0.1f * (float) o - 0.2f;
    }
    std::vector<float> xv((size_t) (W * IC * N));
    for (int64_t w = 0; w < W; ++w) {
        for (int64_t ic = 0; ic < IC; ++ic) {
            for (int64_t n = 0; n < N; ++n) {
                xv[(size_t) (w + W * (ic + IC * n))] =
                    0.1f * (float) ((w + 3 * ic + 5 * n) % 9 - 4);
            }
        }
    }
    write_f32(v, vv);
    write_f32(g, gg);
    write_f32(bias, bv);
    write_f32(x, xv);
    f.dev->graph_compute(f.g);

    // w[k,i,o] = g[o]·v[k,i,o]/sqrt(Σv²+eps)
    constexpr float eps = 1e-12f;
    std::vector<float> wv(vv.size());
    for (int64_t o = 0; o < OC; ++o) {
        double acc = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t i = 0; i < IC_G; ++i) {
                const double vx = (double) vv[(size_t) (k + K * (i + IC_G * o))];
                acc += vx * vx;
            }
        }
        const double norm = std::sqrt(acc + (double) eps);
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t i = 0; i < IC_G; ++i) {
                const size_t idx = (size_t) (k + K * (i + IC_G * o));
                wv[idx] = (float) ((double) gg[(size_t) o] * (double) vv[idx] / norm);
            }
        }
    }

    const std::vector<float> got = read_f32(y);
    TRC_EXPECT_EQ((int64_t) got.size(), OW * OC * N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            const int64_t group = oc / OC_G;
            for (int64_t ow = 0; ow < OW; ++ow) {
                double acc = (double) bv[(size_t) oc];
                for (int64_t k = 0; k < K; ++k) {
                    const int64_t iw = ow * STRIDE - PAD + k;
                    if (iw < 0 || iw >= W) {
                        continue;
                    }
                    for (int64_t ii = 0; ii < IC_G; ++ii) {
                        const int64_t ic = group * IC_G + ii;
                        acc += (double) wv[(size_t) (k + K * (ii + IC_G * oc))] *
                               (double) xv[(size_t) (iw + W * (ic + IC * n))];
                    }
                }
                const size_t idx = (size_t) (ow + OW * (oc + OC * n));
                TRC_EXPECT_NEAR(got[idx], (float) acc, 1e-5);
            }
        }
    }
}

TRC_TEST(nn_gradcheck_norms) {
    const float eps = 1e-5f;
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        LayerNorm ln{};
        layernorm_init(f.ctx, &ln, 4, eps, true, "ln");
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
        Tensor* loss = sum(f.ctx, sqr(f.ctx, layernorm_forward(f.ctx, &ln, x)));
        tensor_set_param(ln.weight);
        tensor_set_param(ln.bias);
        tensor_set_loss(loss);
        f.build(loss);
        write_f32(x, {0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f, -0.11f, 0.12f,
                      -0.13f});
        check_gradients(f, {ln.weight, ln.bias}, loss);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        RmsNorm rn{};
        rmsnorm_init(f.ctx, &rn, 4, eps, "rn");
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 4, 3);
        Tensor* loss = sum(f.ctx, sqr(f.ctx, rmsnorm_forward(f.ctx, &rn, x)));
        tensor_set_param(rn.weight);
        tensor_set_loss(loss);
        f.build(loss);
        write_f32(x, {0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f, -0.11f, 0.12f,
                      -0.13f});
        check_gradients(f, {rn.weight}, loss);
    }
    {
        NnFixture f;
        if (!f.device_ok()) {
            return;
        }
        GroupNorm gn{};
        groupnorm_init(f.ctx, &gn, 2, 4, eps, true, "gn");
        Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 3, 2, 4, 1);  // [W,H,C,N]
        Tensor* loss = sum(f.ctx, sqr(f.ctx, groupnorm_forward(f.ctx, &gn, x)));
        tensor_set_param(gn.weight);
        tensor_set_param(gn.bias);
        tensor_set_loss(loss);
        f.build(loss);
        std::vector<float> xv((size_t) tensor_nelements(x));
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = 0.1f * (float) ((int) i - 12);
        }
        write_f32(x, xv);
        check_gradients(f, {gn.weight, gn.bias}, loss);
    }
}

// ---------------------------------------------------------------- 初始化

TRC_TEST(nn_init_properties) {
    Device* dev = traincpp::device_cpu();
    if (dev == nullptr) {
        std::printf("    跳过：CPU 后端不可用\n");
        return;
    }

    // 延迟填充：分配前数据为空，分配后满足 kaiming bound
    Context* ctx = context_new(1 << 20);
    Rng rng;
    rng_seed(&rng, 123);
    Linear l{};
    linear_init(ctx, &l, 4, 3, true, &rng, "fc1");
    TRC_EXPECT(l.weight != nullptr && l.bias != nullptr);
    TRC_EXPECT_EQ(l.weight->ne[0], 4);
    TRC_EXPECT_EQ(l.weight->ne[1], 3);
    TRC_EXPECT(l.weight->data == nullptr);  // no_alloc 模式：延迟填充
    TRC_EXPECT(std::strcmp(l.weight->name, "fc1.weight") == 0);
    TRC_EXPECT(std::strcmp(l.bias->name, "fc1.bias") == 0);

    ParamList pl;
    linear_params(&l, &pl);
    TRC_EXPECT_EQ(param_list_count(&pl), 2);
    TRC_EXPECT(param_list_get(&pl, 0) == l.weight);

    TRC_EXPECT_EQ(nn_fan_in(l.weight), 4);
    TRC_EXPECT_EQ(nn_fan_out(l.weight), 3);

    buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    const float bound = 1.0f / std::sqrt(4.0f);
    for (float v : read_f32(l.weight)) {
        TRC_EXPECT(std::fabs(v) <= bound + 1e-6f);
    }
    for (float v : read_f32(l.bias)) {
        TRC_EXPECT(std::fabs(v) <= bound + 1e-6f);
    }
    // 归一化层初始化为 1/0
    LayerNorm ln{};
    layernorm_init(ctx, &ln, 4, 1e-5f, true, "ln");
    // 同 ctx 再次分配：新张量补齐（buffer_alloc 跳过已分配数据）
    buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    for (float v : read_f32(ln.weight)) {
        TRC_EXPECT_NEAR(v, 1.0, 1e-9);
    }
    for (float v : read_f32(ln.bias)) {
        TRC_EXPECT_NEAR(v, 0.0, 1e-9);
    }
    context_free(ctx);

    // conv fan_in/fan_out
    Context* ctx2 = context_new(1 << 20);
    Rng rng2;
    rng_seed(&rng2, 5);
    Conv1d c{};
    conv1d_init(ctx2, &c, 2, 4, 3, 1, 0, 1, true, &rng2, "c1");
    TRC_EXPECT_EQ(nn_fan_in(c.weight), 6);   // IC*KW
    TRC_EXPECT_EQ(nn_fan_out(c.weight), 12); // OC*KW
    buffer_alloc_ctx_tensors(ctx2, dev->default_buffer_type());
    context_free(ctx2);

    // 确定性：同种子两次初始化的值完全一致
    Context* ca = context_new(1 << 20);
    Context* cb = context_new(1 << 20);
    Rng ra, rb;
    rng_seed(&ra, 777);
    rng_seed(&rb, 777);
    Linear la{}, lb{};
    linear_init(ca, &la, 3, 5, true, &ra, "a");
    linear_init(cb, &lb, 3, 5, true, &rb, "b");
    buffer_alloc_ctx_tensors(ca, dev->default_buffer_type());
    buffer_alloc_ctx_tensors(cb, dev->default_buffer_type());
    const std::vector<float> va = read_f32(la.weight);
    const std::vector<float> vb = read_f32(lb.weight);
    TRC_EXPECT(va == vb);
    const std::vector<float> ba = read_f32(la.bias);
    const std::vector<float> bb = read_f32(lb.bias);
    TRC_EXPECT(ba == bb);
    context_free(ca);
    context_free(cb);
}

// ---------------------------------------------------------------- 参数标记与冻结（M2.3d）

// 每个 *_init / *_init_weight 都应自动标记参数：不手工 tensor_set_param 也能展开反向
TRC_TEST(nn_init_auto_tag_all_layers) {
    if (trc_test::test_device() == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    auto check_auto_tag = [](auto&& make_loss) {
        NnFixture f;
        ParamList params;
        Tensor* loss = make_loss(f, &params);
        TRC_EXPECT(loss != nullptr);
        TRC_EXPECT(param_list_count(&params) > 0);
        tensor_set_loss(loss);
        f.build(loss);  // 若未自动标参会中止"没有可训练参数"
        for (int64_t i = 0; i < param_list_count(&params); ++i) {
            Tensor* p = param_list_get(&params, i);
            TRC_EXPECT(graph_get_grad(f.g, p) != nullptr);
            TRC_EXPECT(p->grad_acc != nullptr);
        }
    };

    // ---- *_init（自动创建权重）----
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Linear l{};
        Rng rng;
        rng_seed(&rng, 101);
        linear_init(f.ctx, &l, 2, 3, true, &rng, "lin");
        linear_params(&l, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, linear_forward(f.ctx, &l, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Linear l{};
        Rng rng;
        rng_seed(&rng, 102);
        linear_init(f.ctx, &l, 2, 3, false, &rng, "lin_nobias");
        linear_params(&l, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, linear_forward(f.ctx, &l, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Embedding e{};
        Rng rng;
        rng_seed(&rng, 103);
        embedding_init(f.ctx, &e, 4, 2, &rng, "emb");
        embedding_params(&e, params);
        Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
        return sum(f.ctx, sqr(f.ctx, embedding_forward(f.ctx, &e, idx)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Conv1d c{};
        Rng rng;
        rng_seed(&rng, 104);
        conv1d_init(f.ctx, &c, 1, 2, 2, 1, 0, 1, true, &rng, "c1");
        conv1d_params(&c, params);
        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 4, 1, 1);
        return sum(f.ctx, sqr(f.ctx, conv1d_forward(f.ctx, &c, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Conv2d c{};
        Rng rng;
        rng_seed(&rng, 105);
        conv2d_init(f.ctx, &c, 1, 2, 2, 2, 1, 1, 0, 0, 1, 1, true, &rng, "c2");
        conv2d_params(&c, params);
        Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 3, 3, 1, 1);
        return sum(f.ctx, sqr(f.ctx, conv2d_forward(f.ctx, &c, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        LayerNorm ln{};
        layernorm_init(f.ctx, &ln, 2, 1e-5f, true, "ln");
        layernorm_params(&ln, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, layernorm_forward(f.ctx, &ln, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        RmsNorm rn{};
        rmsnorm_init(f.ctx, &rn, 2, 1e-5f, "rn");
        rmsnorm_params(&rn, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, rmsnorm_forward(f.ctx, &rn, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        GroupNorm gn{};
        groupnorm_init(f.ctx, &gn, 1, 2, 1e-5f, true, "gn");
        groupnorm_params(&gn, params);
        Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 2, 1);
        return sum(f.ctx, sqr(f.ctx, groupnorm_forward(f.ctx, &gn, x)));
    });

    // ---- *_init_weight（外部权重）----
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_2d(f.ctx, TYPE_F32, 2, 3);
        Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 3);
        Linear l{};
        linear_init_weight(f.ctx, &l, w, b);
        linear_params(&l, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, linear_forward(f.ctx, &l, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_2d(f.ctx, TYPE_F32, 2, 4);
        Embedding e{};
        embedding_init_weight(f.ctx, &e, w);
        embedding_params(&e, params);
        Tensor* idx = new_tensor_1d(f.ctx, TYPE_I32, 2);
        return sum(f.ctx, sqr(f.ctx, embedding_forward(f.ctx, &e, idx)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_3d(f.ctx, TYPE_F32, 2, 1, 2);
        Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 2);
        Conv1d c{};
        conv1d_init_weight(f.ctx, &c, w, b);
        conv1d_params(&c, params);
        Tensor* x = new_tensor_3d(f.ctx, TYPE_F32, 4, 1, 1);
        return sum(f.ctx, sqr(f.ctx, conv1d_forward(f.ctx, &c, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 1, 2);
        Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 2);
        Conv2d c{};
        conv2d_init_weight(f.ctx, &c, w, b);
        conv2d_params(&c, params);
        Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 3, 3, 1, 1);
        return sum(f.ctx, sqr(f.ctx, conv2d_forward(f.ctx, &c, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_1d(f.ctx, TYPE_F32, 2);
        Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 2);
        LayerNorm ln{};
        layernorm_init_weight(f.ctx, &ln, w, b);
        layernorm_params(&ln, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, layernorm_forward(f.ctx, &ln, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_1d(f.ctx, TYPE_F32, 2);
        RmsNorm rn{};
        rmsnorm_init_weight(f.ctx, &rn, w);
        rmsnorm_params(&rn, params);
        Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 2, 2);
        return sum(f.ctx, sqr(f.ctx, rmsnorm_forward(f.ctx, &rn, x)));
    });
    check_auto_tag([](NnFixture& f, ParamList* params) {
        Tensor* w = new_tensor_1d(f.ctx, TYPE_F32, 2);
        Tensor* b = new_tensor_1d(f.ctx, TYPE_F32, 2);
        GroupNorm gn{};
        groupnorm_init_weight(f.ctx, &gn, w, b);
        groupnorm_params(&gn, params);
        Tensor* x = new_tensor_4d(f.ctx, TYPE_F32, 2, 2, 2, 1);
        return sum(f.ctx, sqr(f.ctx, groupnorm_forward(f.ctx, &gn, x)));
    });
}

TRC_TEST(nn_param_freeze_unfreeze) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }

    Linear l1{}, l2{};
    Rng rng;
    rng_seed(&rng, 31);
    linear_init(f.ctx, &l1, 3, 4, true, &rng, "l1");
    linear_init(f.ctx, &l2, 4, 2, true, &rng, "l2");
    ParamList p1, p2, params;
    linear_params(&l1, &p1);
    linear_params(&l2, &p2);
    param_list_merge(&params, &p1);
    param_list_merge(&params, &p2);
    TRC_EXPECT_EQ(param_list_count(&params), 4);

    // 手工标参与自动标参叠加必须幂等（不中止）
    tensor_set_param(l1.weight);
    tensor_set_param(l1.bias);

    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 3, 2);
    Tensor* loss =
        sum(f.ctx, sqr(f.ctx, linear_forward(f.ctx, &l2, linear_forward(f.ctx, &l1, x))));
    tensor_set_loss(loss);

    // 1) 冻结第二层（只影响本次反向展开）：l1 有梯度、l2 无 grad/grad_acc
    graph_build_forward_expand(f.ctx, f.g, loss);
    param_list_set_param(&p2, false);
    graph_build_backward_expand(f.g);
    TRC_EXPECT(graph_get_grad(f.g, l1.weight) != nullptr);
    TRC_EXPECT(graph_get_grad_acc(f.g, l1.bias) != nullptr);
    TRC_EXPECT(graph_get_grad(f.g, l2.weight) == nullptr);
    TRC_EXPECT(graph_get_grad_acc(f.g, l2.weight) == nullptr);
    TRC_EXPECT(graph_get_grad(f.g, l2.bias) == nullptr);
    TRC_EXPECT(l2.weight->grad_acc == nullptr);

    // 2) 解冻并重建：graph_clear 释放归属后，全部参数恢复梯度
    graph_clear(f.g);
    param_list_set_param(&p2, true);
    graph_build_forward_expand(f.ctx, f.g, loss);
    graph_build_backward_expand(f.g);
    f.buf = buffer_alloc_ctx_tensors(f.ctx, f.dev->default_buffer_type());
    graph_reset(f.g);
    f.dev->graph_compute(f.g);
    for (int64_t i = 0; i < param_list_count(&params); ++i) {
        Tensor* p  = param_list_get(&params, i);
        Tensor* gp = graph_get_grad(f.g, p);
        TRC_EXPECT(gp != nullptr);
        TRC_EXPECT(p->grad_acc != nullptr);
        if (gp != nullptr) {
            const std::vector<float> g = read_f32(gp);
            for (float v : g) {
                TRC_EXPECT(std::isfinite(v));
            }
        }
    }
}

// ---------------------------------------------------------------- Dropout

TRC_TEST(nn_dropout_behavior) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }

    constexpr int64_t NELEM = 64 * 4;
    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, 64, 4);
    Dropout d{};
    dropout_init(&d, 0.25f, "drop");

    Tensor* y = dropout_forward(f.ctx, &d, x);
    TRC_EXPECT(y != x);  // 训练模式：创建 mask
    Tensor* loss = sum(f.ctx, y);
    tensor_set_param(x);
    tensor_set_loss(loss);
    f.build(loss);

    std::vector<float> xv((size_t) NELEM);
    for (size_t i = 0; i < xv.size(); ++i) {
        xv[i] = 0.5f + 0.01f * (float) i;
    }
    write_f32(x, xv);

    Rng rng;
    rng_seed(&rng, 42);
    dropout_refresh(&d, &rng);
    f.step();

    // y == x * mask（逐元素精确）；mask 只有 0 与 1/(1-p)
    const std::vector<float> mask = read_f32(d.mask);
    const std::vector<float> yv = read_f32(y);
    const float keep = 1.0f / (1.0f - 0.25f);
    int64_t n_zero = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
        TRC_EXPECT(mask[i] == 0.0f || std::fabs(mask[i] - keep) < 1e-6f);
        TRC_EXPECT(yv[i] == xv[i] * mask[i]);
        n_zero += (mask[i] == 0.0f) ? 1 : 0;
    }
    const double zero_ratio = (double) n_zero / (double) NELEM;
    TRC_EXPECT(zero_ratio > 0.15 && zero_ratio < 0.35);  // p=0.25，256 样本

    // 反向：dL/dx = mask（loss=sum(y)）
    const std::vector<float> gx = read_f32(graph_get_grad(f.g, x));
    for (size_t i = 0; i < gx.size(); ++i) {
        TRC_EXPECT_NEAR(gx[i], mask[i], 1e-6);
    }

    // eval：恒等（返回输入本身）
    dropout_set_training(&d, false);
    TRC_EXPECT(dropout_forward(f.ctx, &d, x) == x);

    // 确定性：同种子两次刷新结果一致
    dropout_set_training(&d, true);
    Rng r1, r2;
    rng_seed(&r1, 7);
    rng_seed(&r2, 7);
    dropout_refresh(&d, &r1);
    const std::vector<float> m1 = read_f32(d.mask);
    dropout_refresh(&d, &r2);
    const std::vector<float> m2 = read_f32(d.mask);
    TRC_EXPECT(m1 == m2);
}

// ---------------------------------------------------------------- 端到端

TRC_TEST(nn_end_to_end_mlp_layers) {
    NnFixture f;
    if (!f.device_ok()) {
        return;
    }

    constexpr int64_t IN = 2;
    constexpr int64_t HID = 8;
    constexpr int64_t OUT = 1;
    constexpr int64_t BATCH = 16;

    Rng rng;
    rng_seed(&rng, 2026);
    Linear l1{}, l2{};
    linear_init(f.ctx, &l1, IN, HID, true, &rng, "l1");
    linear_init(f.ctx, &l2, HID, OUT, true, &rng, "l2");

    Tensor* x = new_tensor_2d(f.ctx, TYPE_F32, IN, BATCH);
    Tensor* y = new_tensor_2d(f.ctx, TYPE_F32, OUT, BATCH);

    Tensor* h    = relu(f.ctx, linear_forward(f.ctx, &l1, x));
    Tensor* pred = linear_forward(f.ctx, &l2, h);
    Tensor* loss = scale(f.ctx, sum(f.ctx, sqr(f.ctx, sub(f.ctx, pred, y))),
                         1.0f / (float) (OUT * BATCH));
    tensor_set_loss(loss);

    ParamList params;
    linear_params(&l1, &params);
    linear_params(&l2, &params);
    for (Tensor* p : params.items) {
        tensor_set_param(p);
    }
    f.build(loss);

    uint32_t seed = 12345u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float) (seed >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> vx((size_t) (IN * BATCH));
    std::vector<float> vy((size_t) BATCH);
    for (int64_t b = 0; b < BATCH; ++b) {
        const float x0 = rnd();
        const float x1 = rnd();
        vx[(size_t) (b * IN + 0)] = x0;
        vx[(size_t) (b * IN + 1)] = x1;
        vy[(size_t) b] = std::tanh(2.0f * x0 + x1);
    }
    write_f32(x, vx);
    write_f32(y, vy);

    AdamwOptions o;
    o.lr = 0.05f;
    o.weight_decay = 0.0f;
    Optimizer* opt = optim_adamw_new(f.ctx, params.items.data(), param_list_count(&params), o);

    f.step();
    const float loss0 = read_scalar(loss);

    const int steps = 250;
    for (int i = 0; i < steps; ++i) {
        f.step();
        optim_step(opt);
    }
    f.step();
    const float loss1 = read_scalar(loss);
    std::printf("    MLP（层系统）AdamW: loss %.6g -> %.6g（%d 步）\n", loss0, loss1, steps);
    TRC_EXPECT(std::isfinite(loss1));
    TRC_EXPECT(loss1 < 0.2f * loss0);

    optim_free(opt);
}
