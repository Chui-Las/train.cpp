// train.cpp - 规模压力套件（M2.2d / R10 + R12）
//
// 目标：在真实规模下验证"不会静默算错/崩"——大张量逐元素/拷贝/索引/卷积、
//       batch≥2 + 长序列多步训练有限性、get_rows_back 词表×nr 阈值扫描、
//       图级复用峰值、训练步内零分配（R13）。
//
// 档位（环境变量 TRC_STRESS_SCALE）：
//   - 默认 small：1700 万元素档 + 小规模扫描（开发机/核显可跑）
//   - full：追加 6400 万 / 2.5 亿元素档与完整扫描（需大显存；Intel Arc 测试机）
//   设备用 TRC_TEST_DEVICE=cpu|vulkan 选择（同其他套件）。
//
// 说明：测试用确定性模式填充并抽样校验 + 全量 double 求和对照；
//       大档在 weak 设备上可能触发 TDR（尤其 get_rows_back 扫描），
//       每次重负载前会打印并 flush，便于定位最后完成的组合。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace traincpp;

namespace {

// ---------------------------------------------------------------- 档位与工具

bool stress_full() {
    const char* env = std::getenv("TRC_STRESS_SCALE");
    return env != nullptr && std::string(env) == "full";
}

void print_banner(const char* what) {
    static bool printed = false;
    if (printed) {
        return;
    }
    printed      = true;
    Device* dev  = trc_test::test_device();
    std::printf("[stress] %s ｜ 设备=%s ｜ 档位=%s\n", what, dev != nullptr ? dev->props().name.c_str() : "(无)",
                stress_full() ? "full（含 6400 万/2.5 亿）"
                              : "small（1700 万；大档请在 Arc/大显存机器设置 TRC_STRESS_SCALE=full）");
    std::fflush(stdout);
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

constexpr int64_t M17  = 17000000;
constexpr int64_t M64  = 64000000;
constexpr int64_t M250 = 250000000;

// 确定性伪随机模式：范围 [-1, 1)，与元素下标绑定（便于抽样与求和对照）
float pattern(uint64_t i) {
    uint32_t h = (uint32_t) (i * 2654435761u) ^ (uint32_t) (i >> 7);
    h ^= h >> 13;
    return (float) (h & 0xFFFFu) / 32768.0f - 1.0f;
}

// 分块写入模式（避免一次性主机大向量）
void fill_pattern(Tensor* t, uint64_t seed) {
    const int64_t n     = tensor_nelements(t);
    const size_t  chunk = 1u << 20;  // 4 MB
    std::vector<float> buf(chunk);
    for (int64_t off = 0; off < n; off += (int64_t) chunk) {
        const size_t m = (size_t) std::min<int64_t>((int64_t) chunk, n - off);
        for (size_t k = 0; k < m; ++k) {
            buf[k] = pattern(seed + (uint64_t) off + k);
        }
        tensor_set(t, buf.data(), (size_t) off * sizeof(float), m * sizeof(float));
    }
}

void fill_const(Tensor* t, float value) {
    const int64_t n     = tensor_nelements(t);
    const size_t  chunk = 1u << 20;
    std::vector<float> buf(chunk, value);
    for (int64_t off = 0; off < n; off += (int64_t) chunk) {
        const size_t m = (size_t) std::min<int64_t>((int64_t) chunk, n - off);
        tensor_set(t, buf.data(), (size_t) off * sizeof(float), m * sizeof(float));
    }
}

float read_at(const Tensor* t, int64_t i) {
    float v = 0.0f;
    tensor_get(t, &v, (size_t) i * sizeof(float), sizeof(float));
    return v;
}

std::vector<int64_t> sample_points(int64_t n) {
    return {0, 1, n / 4, n / 2 - 1, n / 2, n - 2, n - 1};
}

// add 的全量 double 期望和（分块）
double expected_sum_add(int64_t n, uint64_t seed_a, uint64_t seed_b) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += (double) pattern(seed_a + (uint64_t) i) + (double) pattern(seed_b + (uint64_t) i);
    }
    return s;
}

// 在单个 Context/Graph 中执行（图已构建、buffer 已分配），返回 ms
double compute_ms(Device* dev, Graph* g) {
    const double t0 = now_ms();
    const bool   ok = dev->graph_compute(g);
    const double t1 = now_ms();
    if (!ok) {
        trc_test::add_failure("graph_compute 返回失败（压力用例）");
    }
    return t1 - t0;
}

// 打印内存核算（M2.2a）：分配字节 / 峰值 / 上下文张量字节
void report_mem(const char* tag, Context* ctx, Buffer* buf) {
    const ContextMemoryStats cm = context_memory_stats(ctx);
    const AllocStats        as = backend_alloc_stats();
    std::printf("[stress] %s：buffer=%zu 字节（%.1f MiB）tensor_bytes=%zu persistent=%zu views=%zu "
                "alloc_current=%zu alloc_peak=%zu\n",
                tag, buf->size(), (double) buf->size() / (1024.0 * 1024.0), cm.tensor_bytes,
                cm.persistent_bytes, cm.view_count, as.current_bytes, as.peak_bytes);
    std::fflush(stdout);
}

} // namespace

// ---------------------------------------------------------------- 大张量逐元素

TRC_TEST(stress_elementwise) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("大张量逐元素");

    std::vector<int64_t> tiers = {M17};
    if (stress_full()) {
        tiers.push_back(M64);
        tiers.push_back(M250);
    }

    for (int64_t n : tiers) {
        Context* ctx = context_new(1 << 22);
        Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, n);
        Tensor*  b   = new_tensor_1d(ctx, TYPE_F32, n);
        Tensor*  c   = add(ctx, a, b);        // 输出（独占，compute 后可安全读取）
        Tensor*  c2  = add(ctx, a, b);        // 中间量：供全量和对照
        Tensor*  s   = sum(ctx, c2);

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, c);
        graph_build_forward_expand(ctx, g, s);
        backend_alloc_stats_reset();
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        fill_pattern(a, 0xA11CE);
        fill_pattern(b, 0xB0B);
        const double ms = compute_ms(dev, g);

        double max_abs_err = 0.0;
        for (int64_t i : sample_points(n)) {
            const float want = pattern(0xA11CE + (uint64_t) i) + pattern(0xB0B + (uint64_t) i);
            const float got  = read_at(c, i);
            max_abs_err      = std::max(max_abs_err, (double) std::fabs(got - want));
        }
        const double want_sum = expected_sum_add(n, 0xA11CE, 0xB0B);
        const double got_sum  = (double) read_at(s, 0);
        const double rel_err  = std::fabs(got_sum - want_sum) / std::max(1.0, std::fabs(want_sum));
        TRC_EXPECT(max_abs_err < 1e-4);
        TRC_EXPECT(rel_err < 1e-3);

        std::printf("[stress] elementwise n=%lld：%.1f ms，max_abs=%.3g 全量和 rel=%.3g\n",
                    (long long) n, ms, max_abs_err, rel_err);
        std::fflush(stdout);
        report_mem("elementwise", ctx, buf);

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }
}

// ---------------------------------------------------------------- 大张量拷贝（dup/cont）

TRC_TEST(stress_copy) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("大张量拷贝");

    std::vector<int64_t> tiers = {M17};
    if (stress_full()) {
        tiers.push_back(M64);
        tiers.push_back(M250);
    }

    for (int64_t n : tiers) {
        Context* ctx = context_new(1 << 22);
        Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, n);
        Tensor*  b   = neg(ctx, a);              // 输出：逐元素拷贝路径（取负，精确）
        Tensor*  sa  = sum(ctx, a);
        Tensor*  sb  = sum(ctx, neg(ctx, a));    // 中间量：全量和对照
        Tensor*  h   = cast(ctx, a, TYPE_F16);   // 大张量类型转换路径
        Tensor*  f   = cast(ctx, h, TYPE_F32);   // 输出

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, b);
        graph_build_forward_expand(ctx, g, sb);
        graph_build_forward_expand(ctx, g, sa);
        graph_build_forward_expand(ctx, g, f);
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        fill_pattern(a, 0xC0FFEE);
        const double ms = compute_ms(dev, g);

        double max_abs_err   = 0.0;
        double max_cast_err  = 0.0;
        for (int64_t i : sample_points(n)) {
            const float want = pattern(0xC0FFEE + (uint64_t) i);
            max_abs_err  = std::max(max_abs_err, (double) std::fabs(read_at(b, i) + want));
            max_cast_err = std::max(max_cast_err, (double) std::fabs(read_at(f, i) - want));
        }
        const double sum_a = (double) read_at(sa, 0);
        const double sum_b = (double) read_at(sb, 0);
        TRC_EXPECT(max_abs_err == 0.0f);
        TRC_EXPECT(sum_a == -sum_b);       // 取负逐位精确
        TRC_EXPECT(max_cast_err < 2e-3f);  // F16 往返精度

        std::printf("[stress] copy/cast n=%lld：%.1f ms，neg max_abs=%.3g，f16 往返 max_err=%.3g\n",
                    (long long) n, ms, max_abs_err, max_cast_err);
        std::fflush(stdout);
        report_mem("copy", ctx, buf);

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }
}

// ---------------------------------------------------------------- 大张量索引（get_rows）

TRC_TEST(stress_index) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("大张量索引");

    struct Case {
        int64_t ne0;
        int64_t vocab;
        int64_t nr;
    };
    std::vector<Case> cases = {{64, 8192, 65536}};
    if (stress_full()) {
        cases.push_back({64, 32768, 262144});
    }

    for (const Case& c : cases) {
        Context* ctx   = context_new(1 << 22);
        Tensor*  table = new_tensor_2d(ctx, TYPE_F32, c.ne0, c.vocab);
        Tensor*  idx   = new_tensor_1d(ctx, TYPE_I32, c.nr);
        Tensor*  out   = get_rows(ctx, table, idx);

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, out);
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        fill_pattern(table, 0x7AB1E);
        std::vector<int32_t> iv((size_t) c.nr);
        for (int64_t j = 0; j < c.nr; ++j) {
            iv[(size_t) j] = (int32_t) ((j * 7 + 3) % c.vocab);
        }
        tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int32_t));

        const double ms = compute_ms(dev, g);

        // 抽样：out[:, j] == table[:, idx[j]]
        double max_abs_err = 0.0;
        for (int64_t j : sample_points(c.nr)) {
            for (int64_t i = 0; i < c.ne0; i += 7) {
                const float want = read_at(table, i + (int64_t) iv[(size_t) j] * c.ne0);
                const float got  = read_at(out, i + j * c.ne0);
                max_abs_err = std::max(max_abs_err, (double) std::fabs(got - want));
            }
        }
        TRC_EXPECT(max_abs_err == 0.0f);
        TRC_EXPECT(std::isfinite(read_at(out, 0)));

        std::printf("[stress] index ne0=%lld vocab=%lld nr=%lld：%.1f ms，max_abs=%.3g\n",
                    (long long) c.ne0, (long long) c.vocab, (long long) c.nr, ms, max_abs_err);
        std::fflush(stdout);
        report_mem("index", ctx, buf);

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }
}

// ---------------------------------------------------------------- 大张量卷积（1D + 2D，im2col 全量物化路径）

TRC_TEST(stress_conv) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("大张量卷积");

    struct Case1D {
        int64_t kw;
        int64_t ic;
        int64_t oc;
        int64_t n;  // batch
        int64_t t;  // 序列长度
    };
    std::vector<Case1D> cases = {{3, 128, 64, 2, 17000}};  // col ≈ IC*KW*OW*N ≈ 1300 万元素
    if (stress_full()) {
        cases.push_back({3, 256, 64, 2, 32768});  // col ≈ 5000 万元素
    }

    for (const Case1D& c : cases) {
        Context* ctx = context_new(1 << 24);
        Tensor*  w   = new_tensor_3d(ctx, TYPE_F32, c.kw, c.ic, c.oc);
        Tensor*  x   = new_tensor_3d(ctx, TYPE_F32, c.t, c.ic, c.n);
        Tensor*  y   = conv_1d(ctx, w, x, 1, 0, 1);

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, y);
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        // 解析已知解：w = 1/(IC*KW)，x = 1 → 每个输出（无填充）= 1
        fill_const(w, 1.0f / (float) (c.ic * c.kw));
        fill_const(x, 1.0f);
        const double ms = compute_ms(dev, g);

        const int64_t ow    = y->ne[0];
        double        max_err = 0.0;
        for (int64_t o = 0; o < ow; o += std::max<int64_t>(1, ow / 64)) {
            max_err = std::max(max_err, (double) std::fabs(read_at(y, o) - 1.0f));
        }
        TRC_EXPECT(max_err < 1e-4);

        std::printf("[stress] conv1d kw=%lld ic=%lld oc=%lld n=%lld t=%lld ow=%lld：%.1f ms，max_err=%.3g\n",
                    (long long) c.kw, (long long) c.ic, (long long) c.oc, (long long) c.n,
                    (long long) c.t, (long long) ow, ms, max_err);
        std::fflush(stdout);
        report_mem("conv1d", ctx, buf);

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }

    // 2D 卷积（小一档；验证 KH/KW 组合路径）
    {
        const int64_t kw = 3, kh = 3, ic = 32, oc = 32, n = 2;
        const int64_t hh = stress_full() ? 256 : 128;
        const int64_t ww = stress_full() ? 256 : 128;

        Context* ctx = context_new(1 << 24);
        Tensor*  w   = new_tensor_4d(ctx, TYPE_F32, kw, kh, ic, oc);
        Tensor*  x   = new_tensor_4d(ctx, TYPE_F32, ww, hh, ic, n);
        Tensor*  y   = conv_2d(ctx, w, x, 1, 1, 0, 0, 1, 1);

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, y);
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        fill_const(w, 1.0f / (float) (ic * kw * kh));
        fill_const(x, 1.0f);
        const double ms = compute_ms(dev, g);

        const int64_t ow     = y->ne[0];
        const int64_t oh     = y->ne[1];
        double        max_err = 0.0;
        for (int64_t oy = 0; oy < oh; oy += std::max<int64_t>(1, oh / 16)) {
            for (int64_t ox = 0; ox < ow; ox += std::max<int64_t>(1, ow / 16)) {
                max_err = std::max(max_err, (double) std::fabs(read_at(y, ox + oy * ow) - 1.0f));
            }
        }
        TRC_EXPECT(max_err < 1e-4);
        std::printf("[stress] conv2d %lldx%lld ic=%lld n=%lld col≈%lld：%.1f ms，max_err=%.3g\n",
                    (long long) ww, (long long) hh, (long long) ic, (long long) n,
                    (long long) (ic * kw * kh * ow * oh * n), ms, max_err);
        std::fflush(stdout);
        report_mem("conv2d", ctx, buf);

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }
}

// ---------------------------------------------------------------- batch matmul（M2.3a）

TRC_TEST(stress_matmul_batch) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("batch matmul");

    struct Tier {
        int64_t k, m, n, b2, b3;
    };
    std::vector<Tier> tiers;
    if (stress_full()) {
        tiers.push_back({512, 2048, 2048, 4, 2});
    } else {
        tiers.push_back({128, 512, 512, 4, 2});
    }

    for (const Tier& t : tiers) {
        Context* ctx = context_new(1 << 24);
        // a 的 batch 固定为 (2,2)，b 的 batch 为 (4,2)：整除广播（i2%2, i3%2）
        Tensor* A = new_tensor_4d(ctx, TYPE_F32, t.k, t.m, 2, 2);
        Tensor* B = new_tensor_4d(ctx, TYPE_F32, t.k, t.n, t.b2, t.b3);
        Tensor* D = mul_mat(ctx, A, B);

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, D);
        backend_alloc_stats_reset();
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        // a：4 个平面各填一个 2 的幂（校验取模广播映射）；b 全填 0.25（精确二进制小数）
        const float av[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        std::vector<float> plane((size_t) (t.k * t.m));
        for (int p = 0; p < 4; ++p) {
            std::fill(plane.begin(), plane.end(), av[p]);
            tensor_set(A, plane.data(), (size_t) p * plane.size() * sizeof(float),
                       plane.size() * sizeof(float));
        }
        fill_const(B, 0.25f);

        const double ms = compute_ms(dev, g);

        double max_abs = 0.0;
        const int64_t plane_elems = t.m * t.n;
        for (int64_t i3 = 0; i3 < t.b3; ++i3) {
            for (int64_t i2 = 0; i2 < t.b2; ++i2) {
                const double want = (double) t.k * (double) av[(i2 % 2) + (i3 % 2) * 2] * 0.25;
                const int64_t base = (i2 + i3 * t.b2) * plane_elems;
                const int64_t pts[3] = {0, plane_elems / 2, plane_elems - 1};
                for (int64_t p : pts) {
                    max_abs = std::max(max_abs, (double) std::fabs((double) read_at(D, base + p) - want));
                }
            }
        }
        TRC_EXPECT(max_abs < 1e-3);
        std::printf("[stress] matmul batch k=%lld m=%lld n=%lld b=%lldx%lld（输出 %lld）：%.1f ms，max_err=%.3g\n",
                    (long long) t.k, (long long) t.m, (long long) t.n, (long long) t.b2,
                    (long long) t.b3, (long long) tensor_nelements(D), ms, max_abs);
        std::fflush(stdout);
        report_mem("matmul_batch", ctx, buf);

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }
}

// ---------------------------------------------------------------- batch≥2 + 长序列多步训练有限性

TRC_TEST(stress_training_finite) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("长序列多步训练");

    const int64_t kw = 3, ic = 64, oc = 64, n = 4, t = 4096;
    Context*      ctx = context_new(1 << 24);
    Graph*        g   = graph_new(ctx);

    Tensor* w = new_tensor_3d(ctx, TYPE_F32, kw, ic, oc);
    tensor_set_param(w);
    Tensor* x = new_tensor_3d(ctx, TYPE_F32, t, ic, n);
    Tensor* y    = conv_1d(ctx, w, x, 1, 0, 1);
    Tensor* z    = relu(ctx, y);
    // 全元素均值（mean 只沿 ne0；这里用 sum + scale 得到标量损失）
    Tensor* loss = scale(ctx, sum(ctx, sqr(ctx, z)), 1.0f / (float) tensor_nelements(z));
    tensor_set_loss(loss);

    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);
    Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);

    AdamwOptions opts;
    opts.lr           = 1e-3f;
    opts.weight_decay = 0.0f;
    Optimizer* opt    = optim_adamw_new(ctx, &w, 1, opts);
    optim_alloc_state(opt);

    fill_const(w, 0.01f);
    fill_pattern(x, 0x5EED);

    // 预热一步（scratch/状态分配），随后断言训练步内核心路径零分配（R13）
    graph_reset(g);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（训练预热步）");
    }
    optim_step(opt);
    backend_alloc_stats_reset();

    const int steps = 10;
    for (int step = 0; step < steps; ++step) {
        graph_reset(g);
        const double ms = compute_ms(dev, g);
        const float  lv = read_at(loss, 0);
        TRC_EXPECT(std::isfinite(lv));
        TRC_EXPECT(lv >= 0.0f);
        optim_step(opt);
        if (step == 0 || step == steps - 1) {
            std::printf("[stress] train step %d：loss=%.6g，%.1f ms\n", step, (double) lv, ms);
            std::fflush(stdout);
        }
    }

    const AllocStats after = backend_alloc_stats();
    TRC_EXPECT_EQ(after.n_allocations, 0);
    TRC_EXPECT_EQ(after.current_bytes, 0);

    // 参数有限（抽样）
    bool finite = true;
    const int64_t nw = tensor_nelements(w);
    for (int64_t i = 0; i < nw; i += std::max<int64_t>(1, nw / 64)) {
        finite = finite && std::isfinite(read_at(w, i));
    }
    TRC_EXPECT(finite);
    report_mem("training", ctx, buf);

    optim_free(opt);
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
}

// ---------------------------------------------------------------- get_rows_back 词表×nr 扫描（R12）

TRC_TEST(stress_get_rows_back_scan) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("get_rows_back 扫描");

    struct Case {
        int64_t vocab;
        int64_t nr;
    };
    std::vector<Case> cases = {{1024, 64}, {8192, 512}};
    if (stress_full()) {
        cases.push_back({32768, 512});
        cases.push_back({32768, 4096});
    }

    const int64_t ne0 = 32;
    for (const Case& c : cases) {
        std::printf("[stress] get_rows_back 扫描开始：vocab=%lld nr=%lld（工作 ≈ %lld 次内层迭代）\n",
                    (long long) c.vocab, (long long) c.nr, (long long) (c.vocab * c.nr));
        std::fflush(stdout);

        Context* ctx = context_new(1 << 22);
        Tensor*  dy  = new_tensor_2d(ctx, TYPE_F32, ne0, c.nr);
        Tensor*  idx = new_tensor_1d(ctx, TYPE_I32, c.nr);
        Tensor*  tab = new_tensor_2d(ctx, TYPE_F32, ne0, c.vocab);
        Tensor*  out = get_rows_back(ctx, dy, idx, tab);

        Graph* g = graph_new(ctx);
        graph_build_forward_expand(ctx, g, out);
        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);

        // dy[i, j] = j+1；idx[j] 唯一（7 与 2 的幂互素）；期望 out[i, idx[j]] = dy[i, j]，其余为 0
        {
            std::vector<float>   dv((size_t) (ne0 * c.nr));
            std::vector<int32_t> iv((size_t) c.nr);
            for (int64_t j = 0; j < c.nr; ++j) {
                iv[(size_t) j] = (int32_t) ((j * 7 + 3) % c.vocab);
                for (int64_t i = 0; i < ne0; ++i) {
                    dv[(size_t) (i + j * ne0)] = (float) (j + 1);
                }
            }
            tensor_set(dy, dv.data(), 0, dv.size() * sizeof(float));
            tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int32_t));

            const double t0 = now_ms();
            if (!dev->graph_compute(g)) {
                trc_test::add_failure("graph_compute 返回失败（get_rows_back 扫描）");
            }
            const double ms = now_ms() - t0;

            double max_err = 0.0;
            for (int64_t j : sample_points(c.nr)) {
                const int64_t v = iv[(size_t) j];
                for (int64_t i = 0; i < ne0; i += 5) {
                    max_err = std::max(max_err, (double) std::fabs(read_at(out, i + v * ne0) - (float) (j + 1)));
                }
            }
            // 非索引位置应为 0（抽一个）
            const int32_t v0 = iv[0];
            const int32_t other = (v0 + 1) % (int32_t) c.vocab;
            bool other_is_index = false;
            for (int32_t v : iv) {
                other_is_index = other_is_index || v == other;
            }
            if (!other_is_index) {
                max_err = std::max(max_err, (double) std::fabs(read_at(out, other * ne0)));
            }
            TRC_EXPECT(max_err == 0.0f);

            std::printf("[stress] get_rows_back vocab=%lld nr=%lld：%.1f ms（max_err=%.3g，完成）\n",
                        (long long) c.vocab, (long long) c.nr, ms, max_err);
            std::fflush(stdout);
        }

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
    }
}

// ---------------------------------------------------------------- 图级复用峰值（M2.2b 规模化）

TRC_TEST(stress_memory_peak_reuse) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    print_banner("图级复用峰值");

    const int64_t n = stress_full() ? M17 : M17 / 4;

    Context* ctx = context_new(1 << 22);
    Tensor*  x1  = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor*  x2  = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor*  y1  = mul(ctx, x1, x1);
    Tensor*  z1  = mul(ctx, y1, y1);
    Tensor*  o1  = add(ctx, z1, x1);
    Tensor*  y2  = mul(ctx, x2, x2);
    Tensor*  z2  = mul(ctx, y2, y2);
    Tensor*  o2  = add(ctx, z2, x2);

    Graph* g = graph_new(ctx);
    graph_build_forward_expand(ctx, g, o1);
    graph_build_forward_expand(ctx, g, o2);

    backend_alloc_stats_reset();
    Buffer* old_buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(old_buf != nullptr);
    const size_t old_size = old_buf->size();
    buffer_free(old_buf);

    Buffer* new_buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(new_buf != nullptr);
    const size_t new_size = new_buf->size();
    TRC_EXPECT(new_size < old_size);

    fill_pattern(x1, 0x1111);
    fill_pattern(x2, 0x2222);
    const double ms = compute_ms(dev, g);

    double max_err = 0.0;
    for (int64_t i : sample_points(n)) {
        const float a = pattern(0x1111 + (uint64_t) i);
        const float b = pattern(0x2222 + (uint64_t) i);
        max_err = std::max(max_err, (double) std::fabs(read_at(o1, i) - (a * a * a * a + a)));
        max_err = std::max(max_err, (double) std::fabs(read_at(o2, i) - (b * b * b * b + b)));
    }
    TRC_EXPECT(max_err < 1e-4);

    std::printf("[stress] reuse n=%lld：整 ctx=%zu 字节 → 图级=%zu 字节（降 %.1f%%），%.1f ms，max_err=%.3g\n",
                (long long) n, old_size, new_size,
                100.0 * (1.0 - (double) new_size / (double) std::max<size_t>(1, old_size)), ms, max_err);
    std::fflush(stdout);
    report_mem("reuse", ctx, new_buf);

    buffer_free(new_buf);
    graph_free(g);
    context_free(ctx);
}
