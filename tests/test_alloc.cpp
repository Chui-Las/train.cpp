// train.cpp - 分配生命周期测试（M2.1d / R3）
// 目标：buffer_free 之后张量不得再持有已释放内存的 data/buffer（杜绝悬垂指针）；
//       重新 buffer_alloc_ctx_tensors 后所有张量（含视图）必须重新绑定并计算正确。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace traincpp;

namespace {

void write_f32(Tensor* t, const std::vector<float>& v) {
    tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

std::vector<float> read_f32(const Tensor* t) {
    std::vector<float> v((size_t) tensor_nelements(t));
    tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// 写入固定输入并校验 a+b 的前向结果
bool check_forward(Device* dev, Graph* g, Tensor* a, Tensor* b, Tensor* c) {
    write_f32(a, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(b, {10.0f, 20.0f, 30.0f, 40.0f});
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败");
        return false;
    }
    const std::vector<float> want = {11.0f, 22.0f, 33.0f, 44.0f};
    const std::vector<float> got  = read_f32(c);
    if (got != want) {
        trc_test::add_failure("前向结果错误（期望 11,22,33,44）");
        return false;
    }
    return true;
}

} // namespace

// 释放 → 重新分配：所有归属张量（含视图）先失效、后重建，且计算仍正确
TRC_TEST(alloc_free_realloc) {
    Device* dev = trc_test::test_device();  // CPU / Vulkan 均跑
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* c = add(ctx, a, b);
    Tensor* m = new_tensor_2d(ctx, TYPE_F32, 4, 3);
    Tensor* v = transpose(ctx, m);  // 视图：data 由源张量解析

    graph_build_forward_expand(ctx, g, c);

    Buffer* buf1 = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf1 != nullptr);

    check_forward(dev, g, a, b, c);
    TRC_EXPECT(a->data != nullptr);
    TRC_EXPECT(m->data != nullptr);
    TRC_EXPECT(v->data == (uint8_t*) m->data + v->view_offs);

    buffer_free(buf1);

    // 失败先行点：释放后必须全部失效（当前实现仍指向已释放内存）
    TRC_EXPECT(a->data == nullptr);
    TRC_EXPECT(a->buffer == nullptr);
    TRC_EXPECT(b->data == nullptr);
    TRC_EXPECT(b->buffer == nullptr);
    TRC_EXPECT(c->data == nullptr);
    TRC_EXPECT(c->buffer == nullptr);
    TRC_EXPECT(m->data == nullptr);
    TRC_EXPECT(m->buffer == nullptr);
    TRC_EXPECT(v->data == nullptr);

    buffer_free(nullptr);  // 空指针安全，且不影响任何张量

    Buffer* buf2 = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf2 != nullptr);
    TRC_EXPECT(a->buffer == buf2);
    TRC_EXPECT(b->buffer == buf2);
    TRC_EXPECT(c->buffer == buf2);
    TRC_EXPECT(m->buffer == buf2);
    TRC_EXPECT(a->data != nullptr);
    TRC_EXPECT(c->data != nullptr);
    TRC_EXPECT(v->data == (uint8_t*) m->data + v->view_offs);

    if (check_forward(dev, g, a, b, c)) {
        TRC_EXPECT(read_f32(c)[0] == 11.0f);
    }

    graph_free(g);
    buffer_free(buf2);
    context_free(ctx);
}

// ---------------- M2.2a 内存核算 API ----------------

// context_memory_stats：字节统计与持久集合判定（无需设备）
TRC_TEST(context_memory_stats_counts) {
    Context* ctx = context_new(1 << 20);

    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);     // 16 B，叶子 → 持久
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);     // 16 B，叶子 → 持久
    Tensor* c = add(ctx, a, b);                      // 16 B，中间量 → 非持久
    Tensor* m = new_tensor_2d(ctx, TYPE_F32, 4, 3);  // 48 B，视图源 → 持久
    Tensor* v = transpose(ctx, m);                   // 视图，不计字节
    Tensor* w = new_tensor_1d(ctx, TYPE_F32, 4);     // 16 B，PARAM → 持久
    tensor_set_param(w);
    (void) c;
    (void) v;

    const ContextMemoryStats s = context_memory_stats(ctx);
    TRC_EXPECT_EQ(s.tensor_bytes, (long long) (16 + 16 + 16 + 48 + 16));
    TRC_EXPECT_EQ(s.persistent_bytes, (long long) (16 + 16 + 48 + 16));
    TRC_EXPECT_EQ(s.view_count, 1);

    context_free(ctx);
}

// context_memory_stats：set_rows 目标与 acc(inplace) 宿主张量（虽非叶子）必须计入持久
TRC_TEST(context_memory_stats_host_persistent) {
    Context* ctx = context_new(1 << 20);

    Tensor* m    = new_tensor_2d(ctx, TYPE_F32, 4, 3);        // 48 B，视图源
    Tensor* host = cont(ctx, transpose(ctx, m));              // [3,4] OP_CONT，非叶子
    Tensor* vals = new_tensor_2d(ctx, TYPE_F32, 3, 2);        // 叶子
    Tensor* idx  = new_tensor_1d(ctx, TYPE_I32, 2);           // 叶子

    const ContextMemoryStats before     = context_memory_stats(ctx);
    const size_t              host_bytes = tensor_nbytes(host);
    // 除 host 外全部非视图张量都是持久（m 视图源 / vals·idx 叶子）
    TRC_EXPECT_EQ(before.persistent_bytes + host_bytes, before.tensor_bytes);

    Tensor* sr = set_rows(ctx, host, vals, idx);  // 写入 host；结果是与 host 共享内存的视图
    (void) sr;
    const ContextMemoryStats after = context_memory_stats(ctx);
    TRC_EXPECT_EQ(after.tensor_bytes, before.tensor_bytes);
    TRC_EXPECT_EQ(after.persistent_bytes, before.persistent_bytes + host_bytes);

    // acc(inplace)：宿主同样必须持久
    Tensor* m2    = new_tensor_2d(ctx, TYPE_F32, 4, 3);
    Tensor* host2 = cont(ctx, transpose(ctx, m2));  // [3,4] 连续拷贝（完整区域 = 48 B）
    Tensor* src2  = new_tensor_2d(ctx, TYPE_F32, 3, 4);
    const ContextMemoryStats s1 = context_memory_stats(ctx);
    Tensor* ar = acc(ctx, host2, src2, 12, 12, 12, 0, true);  // 覆盖整个 [3,4] 区域
    (void) ar;
    const ContextMemoryStats s2 = context_memory_stats(ctx);
    TRC_EXPECT_EQ(s2.tensor_bytes, s1.tensor_bytes);
    TRC_EXPECT_EQ(s2.persistent_bytes, s1.persistent_bytes + tensor_nbytes(host2));

    context_free(ctx);
}

// context_memory_stats：反向展开后 grad_acc 计入持久集合（无需设备）
TRC_TEST(context_memory_stats_grad_acc) {
    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    Tensor* w = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* x    = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* y    = add(ctx, x, w);
    Tensor* loss = sum(ctx, y);
    tensor_set_loss(loss);
    graph_build_forward_expand(ctx, g, loss);

    const ContextMemoryStats before = context_memory_stats(ctx);
    graph_build_backward_expand(g);  // 为 PARAM/LOSS 创建 grad_acc 与新常量
    const ContextMemoryStats after = context_memory_stats(ctx);

    Tensor* gw = graph_get_grad_acc(g, w);
    Tensor* gl = graph_get_grad_acc(g, loss);
    TRC_EXPECT(gw != nullptr);
    TRC_EXPECT(gl != nullptr);
    TRC_EXPECT(after.tensor_bytes >=
               before.tensor_bytes + tensor_nbytes(gw) + tensor_nbytes(gl));
    TRC_EXPECT(after.persistent_bytes >=
               before.persistent_bytes + tensor_nbytes(gw) + tensor_nbytes(gl));

    graph_free(g);
    context_free(ctx);
}

// backend_alloc_stats：只统计核心路径分配（ctx buffer + 优化器状态块），buffer_free 扣减
TRC_TEST(backend_alloc_stats_track) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    backend_alloc_stats_reset();

    Context* ctx = context_new(1 << 20);
    Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor*  b   = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor*  c   = add(ctx, a, b);
    Tensor*  w   = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    (void) c;

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);
    // 计数口径 = 实际缓冲字节数（含后端对齐填充）；至少容纳 4 个 F32[4] 张量
    const size_t ctx_bytes = buf->size();
    TRC_EXPECT(ctx_bytes >= 16 * 4);

    AllocStats s = backend_alloc_stats();
    TRC_EXPECT_EQ(s.n_allocations, 1);
    TRC_EXPECT_EQ(s.current_bytes, ctx_bytes);
    TRC_EXPECT_EQ(s.peak_bytes, ctx_bytes);

    Optimizer* opt = optim_sgd_new(ctx, &w, 1, SgdOptions{});
    optim_alloc_state(opt);  // 参数已分配 → 状态走设备块，计入统计
    const size_t state_bytes = 16;  // SGD momentum 与参数同形状
    s = backend_alloc_stats();
    TRC_EXPECT_EQ(s.n_allocations, 2);
    TRC_EXPECT_EQ(s.current_bytes, ctx_bytes + state_bytes);
    TRC_EXPECT_EQ(s.peak_bytes, ctx_bytes + state_bytes);

    buffer_free(buf);
    s = backend_alloc_stats();
    TRC_EXPECT_EQ(s.current_bytes, state_bytes);
    TRC_EXPECT_EQ(s.peak_bytes, ctx_bytes + state_bytes);

    optim_free(opt);  // 内部 buffer_free → 扣减状态块
    s = backend_alloc_stats();
    TRC_EXPECT_EQ(s.current_bytes, 0);
    TRC_EXPECT_EQ(s.peak_bytes, ctx_bytes + state_bytes);

    backend_alloc_stats_reset();
    s = backend_alloc_stats();
    TRC_EXPECT_EQ(s.current_bytes, 0);
    TRC_EXPECT_EQ(s.peak_bytes, 0);
    TRC_EXPECT_EQ(s.n_allocations, 0);

    context_free(ctx);
}

// ---------------- M2.2b 图级内存复用 ----------------

namespace {

// 用指定分配 API 跑同一小模型 5 步 AdamW，返回最终参数（用于逐位对比）
std::vector<float> train_with_alloc(Device* dev, bool graph_alloc) {
    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    Tensor* w = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* x    = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* tgt  = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* pred = add(ctx, x, w);
    Tensor* loss = mse_loss(ctx, pred, tgt);
    tensor_set_loss(loss);

    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);

    Buffer* buf = graph_alloc ? buffer_alloc_graph_tensors(g, dev->default_buffer_type())
                              : buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);

    AdamwOptions opts;
    opts.lr = 0.05f;
    Optimizer* opt = optim_adamw_new(ctx, &w, 1, opts);
    optim_alloc_state(opt);

    write_f32(w, {0.5f, -0.5f, 0.25f, 0.75f});
    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(tgt, {0.0f, 1.0f, 2.0f, 3.0f});

    for (int step = 0; step < 5; ++step) {
        graph_reset(g);
        if (!dev->graph_compute(g)) {
            trc_test::add_failure("graph_compute 返回失败（分配等价性用例）");
            break;
        }
        optim_step(opt);
    }
    const std::vector<float> wv = read_f32(w);

    optim_free(opt);
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return wv;
}

// FIX-002 Phase 2：带视图根 + 多后继中间量的训练等价性（5 步 AdamW 逐位对比）
std::vector<float> train_with_alloc_view_multi(Device* dev, bool graph_alloc) {
    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    Tensor* w = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* x    = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* tgt  = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* r    = mul(ctx, w, w);         // 视图根
    Tensor* v    = view_1d(ctx, r, 4, 0);  // 视图（no-op，共享 r 的存储）
    Tensor* a    = add(ctx, v, x);         // 读 r（经视图）
    Tensor* b    = mul(ctx, a, x);         // a 的读者 1
    Tensor* c    = mul(ctx, a, a);         // a 的读者 2（多后继）
    Tensor* pred = add(ctx, b, c);
    Tensor* loss = mse_loss(ctx, pred, tgt);
    tensor_set_loss(loss);

    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);

    Buffer* buf = graph_alloc ? buffer_alloc_graph_tensors(g, dev->default_buffer_type())
                              : buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);

    AdamwOptions opts;
    opts.lr        = 0.05f;
    Optimizer* opt = optim_adamw_new(ctx, &w, 1, opts);
    optim_alloc_state(opt);

    write_f32(w, {0.5f, -0.5f, 0.25f, 0.75f});
    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(tgt, {0.0f, 1.0f, 2.0f, 3.0f});

    for (int step = 0; step < 5; ++step) {
        graph_reset(g);
        if (!dev->graph_compute(g)) {
            trc_test::add_failure("graph_compute 返回失败（视图+多后继等价性用例）");
            break;
        }
        optim_step(opt);
    }
    const std::vector<float> wv = read_f32(w);

    optim_free(opt);
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return wv;
}

} // namespace

// 图级 liveness 复用：两条互不依赖的中间链 → 峰值显著下降，且结果正确
TRC_TEST(alloc_graph_reuse) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    const int64_t n = 1024;  // 每张量 4 KiB
    Tensor* x1 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* x2 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* y1 = mul(ctx, x1, x1);  // 单后继中间量 → 可复用
    Tensor* z1 = mul(ctx, y1, y1);
    Tensor* o1 = add(ctx, z1, x1);  // 图输出（无后继）→ 独占
    Tensor* y2 = mul(ctx, x2, x2);
    Tensor* z2 = mul(ctx, y2, y2);
    Tensor* o2 = add(ctx, z2, x2);
    graph_build_forward_expand(ctx, g, o1);
    graph_build_forward_expand(ctx, g, o2);

    // 旧 API：整 ctx 一次性分配（8 张量 = 32 KiB）
    Buffer* old_buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(old_buf != nullptr);
    const size_t old_size = old_buf->size();
    buffer_free(old_buf);

    // 新 API：y1/z1 与 y2/z2 生命周期不重叠 → 复用（预期省 2 块 = 8 KiB）
    Buffer* new_buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(new_buf != nullptr);
    const size_t new_size = new_buf->size();
    TRC_EXPECT(new_size < old_size);
    TRC_EXPECT(new_size + 2 * 4096 <= old_size);

    // 复用不破坏计算：o = x^4 + x
    std::vector<float> v1((size_t) n), v2((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        v1[(size_t) i] = (float) (i % 7) * 0.5f - 1.5f;
        v2[(size_t) i] = (float) (i % 5) * 0.25f - 0.5f;
    }
    write_f32(x1, v1);
    write_f32(x2, v2);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（复用用例）");
    }
    const std::vector<float> got1 = read_f32(o1);
    const std::vector<float> got2 = read_f32(o2);
    for (int64_t i = 0; i < n; ++i) {
        const float a = v1[(size_t) i];
        const float b = v2[(size_t) i];
        TRC_EXPECT_NEAR(got1[(size_t) i], a * a * a * a + a, 1e-3);
        TRC_EXPECT_NEAR(got2[(size_t) i], b * b * b * b + b, 1e-3);
    }

    // R3 语义：释放后张量失效
    buffer_free(new_buf);
    TRC_EXPECT(x1->data == nullptr);
    TRC_EXPECT(y1->data == nullptr);
    TRC_EXPECT(o1->data == nullptr);

    graph_free(g);
    context_free(ctx);
}

// 图级分配与整 ctx 分配的训练轨迹必须逐位一致（覆盖反向复用交互）
TRC_TEST(alloc_graph_training_equiv) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::vector<float> w_ctx    = train_with_alloc(dev, false);
    const std::vector<float> w_graph  = train_with_alloc(dev, true);
    TRC_EXPECT(w_ctx.size() == 4 && w_graph.size() == 4);
    TRC_EXPECT(w_ctx == w_graph);
}

// R13 验收：训练步内核心路径零分配（连续 10 步统计不增长；CPU/Vulkan 双跑）
// 注：AllocStats 只统计核心路径 Buffer（ctx 分配 + 优化器设备状态块）；主机临时量
// 由实现改为 data 直写/scratch 复用，数值由 test_optim/test_train_trace 逐位守住。
TRC_TEST(alloc_stats_no_growth_in_training) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    Tensor* w = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* x    = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* tgt  = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* pred = add(ctx, x, w);
    Tensor* loss = mse_loss(ctx, pred, tgt);
    tensor_set_loss(loss);

    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);

    write_f32(w, {0.5f, -0.5f, 0.25f, 0.75f});
    write_f32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(tgt, {0.0f, 1.0f, 2.0f, 3.0f});

    SgdOptions opts;
    opts.lr           = 0.05f;
    opts.momentum     = 0.9f;
    opts.nesterov     = true;
    opts.weight_decay = 0.01f;  // 覆盖 SGD 的 weight_decay/Nesterov 临时量路径
    Optimizer* opt = optim_sgd_new(ctx, &w, 1, opts);
    optim_alloc_state(opt);

    // 预热一步：历史实现会在首步扩容 scratch / 分配全尺寸临时 vector
    graph_reset(g);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（预热步）");
    }
    optim_step(opt);

    backend_alloc_stats_reset();  // 基线：此后核心路径不得再分配

    for (int step = 0; step < 10; ++step) {
        graph_reset(g);
        if (!dev->graph_compute(g)) {
            trc_test::add_failure("graph_compute 返回失败（训练步）");
            break;
        }
        optim_step(opt);
    }

    const AllocStats after = backend_alloc_stats();
    TRC_EXPECT_EQ(after.n_allocations, 0);
    TRC_EXPECT_EQ(after.current_bytes, 0);
    TRC_EXPECT_EQ(after.peak_bytes, 0);

    const std::vector<float> wv = read_f32(w);
    for (float v : wv) {
        TRC_EXPECT(std::isfinite(v));
    }
    TRC_EXPECT(wv != std::vector<float>({0.5f, -0.5f, 0.25f, 0.75f}));

    graph_free(g);
    optim_free(opt);
    buffer_free(buf);
    context_free(ctx);
}

// M2.2c / R14：100 次"建图 → 分配 → 释放 → 回收"循环，数据内存不增长、持久张量保留
TRC_TEST(alloc_release_rebuild) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);

    // 固定持久张量：参数 + 损失（每轮复用同一损失，模拟同结构重建）
    Tensor* w = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* loss = sum(ctx, w);
    tensor_set_loss(loss);

    Graph* g = graph_new(ctx);

    const int iterations = 100;
    size_t    first_buf_size = 0;
    size_t    persistent_bytes = 0;

    for (int it = 0; it < iterations; ++it) {
        graph_clear(g);
        graph_build_forward_expand(ctx, g, loss);
        graph_build_backward_expand(g);

        Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
        TRC_EXPECT(buf != nullptr);
        if (it == 0) {
            first_buf_size = buf->size();
        } else {
            // 第 N 轮的分配大小与首轮一致（旧中间量已被回收，不累积）
            TRC_EXPECT_EQ(buf->size(), first_buf_size);
        }

        graph_reset(g);
        if (!dev->graph_compute(g)) {
            trc_test::add_failure("graph_compute 返回失败（回收重建用例）");
            break;
        }

        buffer_free(buf);
        const size_t released = context_release_graph_tensors(ctx, g);
        TRC_EXPECT(released > 0);
        TRC_EXPECT_EQ(context_release_graph_tensors(ctx, g), 0);  // 重复调用安全：无候选

        // 数据字节回到持久集合基线（每轮相同）
        if (it == 0) {
            persistent_bytes = context_memory_stats(ctx).tensor_bytes;
            TRC_EXPECT_EQ(persistent_bytes, 40);  // w(16) + loss(4) + 2×grad_acc(16+4)
        } else {
            TRC_EXPECT_EQ(context_memory_stats(ctx).tensor_bytes, persistent_bytes);
        }
    }

    // 参数与损失仍在 registry，可继续使用
    TRC_EXPECT(w->flags & TENSOR_FLAG_PARAM);
    TRC_EXPECT(loss->flags & TENSOR_FLAG_LOSS);

    graph_free(g);
    context_free(ctx);
}

// buffer_free 只清理归属自己上下文（buffer）的张量，不影响其他上下文
TRC_TEST(free_scoped_to_own_buffer) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx1 = context_new(1 << 20);
    Context* ctx2 = context_new(1 << 20);

    Tensor* t1 = new_tensor_1d(ctx1, TYPE_F32, 4);
    Tensor* t2 = new_tensor_1d(ctx2, TYPE_F32, 4);

    Buffer* buf1 = buffer_alloc_ctx_tensors(ctx1, dev->default_buffer_type());
    Buffer* buf2 = buffer_alloc_ctx_tensors(ctx2, dev->default_buffer_type());
    TRC_EXPECT(buf1 != nullptr);
    TRC_EXPECT(buf2 != nullptr);

    write_f32(t1, {1.0f, 2.0f, 3.0f, 4.0f});
    write_f32(t2, {5.0f, 6.0f, 7.0f, 8.0f});

    buffer_free(buf1);

    TRC_EXPECT(t1->data == nullptr);
    TRC_EXPECT(t1->buffer == nullptr);
    TRC_EXPECT(t2->data != nullptr);
    TRC_EXPECT(t2->buffer == buf2);

    const std::vector<float> got = read_f32(t2);
    TRC_EXPECT(got.size() == 4 && got[0] == 5.0f && got[3] == 8.0f);

    buffer_free(nullptr);
    TRC_EXPECT(t2->data != nullptr);  // 空指针 free 不得误伤其他 buffer

    buffer_free(buf2);
    context_free(ctx1);
    context_free(ctx2);
}

// ---------------- FIX-002 / Q23：视图根 liveness 复用 ----------------

// Phase 1：视图根按使用窗口参与 liveness——视图被消费后，根可被后续中间量复用
TRC_TEST(alloc_graph_view_root_reuse) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    const int64_t n = 1024;  // 每张量 4 KiB
    Tensor* x1 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* x2 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* big = mul(ctx, x1, x1);         // 视图根（单后继：视图节点 no-op，实际读者是 c）
    Tensor* v   = view_1d(ctx, big, n, 0);  // 视图与 big 共享内存
    Tensor* c   = mul(ctx, v, x2);          // 读取 big（经视图）
    Tensor* d   = mul(ctx, x2, x2);         // 生产晚于 c → 可复用 big 的块
    Tensor* e   = mul(ctx, d, d);
    Tensor* o   = add(ctx, c, e);
    graph_build_forward_expand(ctx, g, o);

    // 旧 API：整 ctx 一次性分配（7 个非视图张量 = 28 KiB）
    Buffer* old_buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(old_buf != nullptr);
    const size_t old_size = old_buf->size();
    buffer_free(old_buf);

    // 新 API：big 在 c 之后可复用 → 峰值下降至少一块
    Buffer* new_buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(new_buf != nullptr);
    const size_t new_size = new_buf->size();
    TRC_EXPECT(new_size < old_size);
    TRC_EXPECT(new_size + 4096 <= old_size);

    // 复用不破坏计算：o = x1^2 * x2 + x2^4
    std::vector<float> v1((size_t) n), v2((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        v1[(size_t) i] = (float) (i % 7) * 0.5f - 1.5f;
        v2[(size_t) i] = (float) (i % 5) * 0.25f - 0.5f;
    }
    write_f32(x1, v1);
    write_f32(x2, v2);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（视图根复用用例）");
    }
    const std::vector<float> got = read_f32(o);
    for (int64_t i = 0; i < n; ++i) {
        const float a = v1[(size_t) i];
        const float b = v2[(size_t) i];
        TRC_EXPECT_NEAR(got[(size_t) i], a * a * b + b * b * b * b, 1e-3);
    }

    buffer_free(new_buf);
    graph_free(g);
    context_free(ctx);
}

// Phase 1：带 TENSOR_FLAG_OUTPUT 的视图钉住其根（compute 后仍可读，不被复用覆盖）
TRC_TEST(alloc_graph_view_output_pin) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    const int64_t n = 1024;
    Tensor* x1 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* x2 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* big = mul(ctx, x1, x1);
    Tensor* v   = view_1d(ctx, big, n, 0);
    v->flags |= TENSOR_FLAG_OUTPUT;  // 声明 compute 后仍要读 → 钉住 big
    Tensor* c = mul(ctx, v, x2);
    Tensor* d = mul(ctx, x2, x2);
    Tensor* e = mul(ctx, d, d);
    Tensor* o = add(ctx, c, e);
    graph_build_forward_expand(ctx, g, o);

    Buffer* old_buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(old_buf != nullptr);
    const size_t old_size = old_buf->size();
    buffer_free(old_buf);

    // big 被钉住 → 无任何复用，大小与整 ctx 分配一致
    Buffer* new_buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(new_buf != nullptr);
    TRC_EXPECT_EQ(new_buf->size(), old_size);

    std::vector<float> v1((size_t) n), v2((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        v1[(size_t) i] = (float) (i % 7) * 0.5f - 1.5f;
        v2[(size_t) i] = (float) (i % 5) * 0.25f - 0.5f;
    }
    write_f32(x1, v1);
    write_f32(x2, v2);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（OUTPUT 视图钉住用例）");
    }
    // compute 后读取 OUTPUT 视图：必须仍是 x1^2（未被复用覆盖）
    const std::vector<float> got = read_f32(v);
    for (int64_t i = 0; i < n; ++i) {
        const float a = v1[(size_t) i];
        TRC_EXPECT_NEAR(got[(size_t) i], a * a, 1e-3);
    }

    buffer_free(new_buf);
    graph_free(g);
    context_free(ctx);
}

// Phase 1：图外视图（未被任何图节点引用）钉住其根——防主机后读被复用覆盖
TRC_TEST(alloc_graph_external_view_pin) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    const int64_t n = 1024;
    Tensor* x1 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* x2 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* big = mul(ctx, x1, x1);         // 图内被 o 直接读取
    Tensor* v   = view_1d(ctx, big, n, 0);  // 图外视图（不作为任何节点 src）
    Tensor* c   = mul(ctx, x2, x2);
    Tensor* o   = add(ctx, big, c);
    Tensor* d   = mul(ctx, c, c);           // 晚于 o → 若 big 可复用会复用
    Tensor* o2  = add(ctx, o, d);
    graph_build_forward_expand(ctx, g, o2);

    Buffer* old_buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(old_buf != nullptr);
    const size_t old_size = old_buf->size();
    buffer_free(old_buf);

    // 图外视图钉住 big → 无复用，大小与整 ctx 分配一致
    Buffer* new_buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(new_buf != nullptr);
    TRC_EXPECT_EQ(new_buf->size(), old_size);

    std::vector<float> v1((size_t) n), v2((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        v1[(size_t) i] = (float) (i % 7) * 0.5f - 1.5f;
        v2[(size_t) i] = (float) (i % 5) * 0.25f - 0.5f;
    }
    write_f32(x1, v1);
    write_f32(x2, v2);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（图外视图钉住用例）");
    }
    const std::vector<float> got = read_f32(v);
    for (int64_t i = 0; i < n; ++i) {
        const float a = v1[(size_t) i];
        TRC_EXPECT_NEAR(got[(size_t) i], a * a, 1e-3);
    }

    buffer_free(new_buf);
    graph_free(g);
    context_free(ctx);
}

// Phase 2：多后继中间量在"最后读取"之后可复用（a 有 2 个读者 b/c，d 复用 a 的块）
TRC_TEST(alloc_graph_multi_reader_reuse) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    const int64_t n = 1024;  // 每张量 4 KiB
    Tensor* x1 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* x2 = new_tensor_1d(ctx, TYPE_F32, n);
    Tensor* a  = add(ctx, x1, x1);   // 多后继中间量：读者 b、c
    Tensor* b  = mul(ctx, a, x2);    // 读者 1
    Tensor* c  = mul(ctx, a, a);     // 读者 2（最后读取）
    Tensor* d  = mul(ctx, x2, x2);   // 生产晚于 c → 可复用 a 的块
    Tensor* e  = mul(ctx, d, d);
    Tensor* o  = add(ctx, add(ctx, b, c), e);
    graph_build_forward_expand(ctx, g, o);

    Buffer* old_buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(old_buf != nullptr);
    const size_t old_size = old_buf->size();
    buffer_free(old_buf);

    // 旧行为（单后继限制）下 a 独占；Phase 2 放开后 d 可复用 a 的块
    Buffer* new_buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());
    TRC_EXPECT(new_buf != nullptr);
    TRC_EXPECT(new_buf->size() < old_size);
    TRC_EXPECT(new_buf->size() + 4096 <= old_size);

    // 数值：o = (x1+x1)*x2 + (x1+x1)^2 + (x2^2)^2
    std::vector<float> v1((size_t) n), v2((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        v1[(size_t) i] = (float) (i % 7) * 0.5f - 1.5f;
        v2[(size_t) i] = (float) (i % 5) * 0.25f - 0.5f;
    }
    write_f32(x1, v1);
    write_f32(x2, v2);
    if (!dev->graph_compute(g)) {
        trc_test::add_failure("graph_compute 返回失败（多后继复用用例）");
    }
    const std::vector<float> got = read_f32(o);
    for (int64_t i = 0; i < n; ++i) {
        const float sa = v1[(size_t) i] + v1[(size_t) i];
        const float sb = v2[(size_t) i];
        TRC_EXPECT_NEAR(got[(size_t) i], sa * sb + sa * sa + sb * sb * sb * sb, 1e-3);
    }

    buffer_free(new_buf);
    graph_free(g);
    context_free(ctx);
}

// Phase 2：视图根 + 多后继进入复用后，训练轨迹必须与整 ctx 分配逐位一致
TRC_TEST(alloc_graph_view_multi_reader_equiv) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }

    const std::vector<float> w_ctx   = train_with_alloc_view_multi(dev, false);
    const std::vector<float> w_graph = train_with_alloc_view_multi(dev, true);
    TRC_EXPECT(w_ctx.size() == 4 && w_graph.size() == 4);
    TRC_EXPECT(w_ctx == w_graph);
}
