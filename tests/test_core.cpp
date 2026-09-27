// train.cpp - core 层测试：类型表、张量形状、上下文、图展开
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cstring>

using namespace traincpp;

TRC_TEST(type_table) {
    TRC_EXPECT_EQ(type_size(TYPE_F32), 4);
    TRC_EXPECT_EQ(type_size(TYPE_F16), 2);
    TRC_EXPECT_EQ(type_size(TYPE_BF16), 2);
    TRC_EXPECT_EQ(type_blck_size(TYPE_F32), 1);
    TRC_EXPECT_EQ(type_blck_size(TYPE_Q4_0), 32);
    TRC_EXPECT_EQ(type_size(TYPE_Q4_0), 18);
    TRC_EXPECT_EQ(type_size(TYPE_Q8_0), 34);
    TRC_EXPECT_EQ(type_row_size(TYPE_F32, 3), 12);
    TRC_EXPECT_NEAR(type_sizef(TYPE_Q4_0), 18.0 / 32.0, 1e-12);
    TRC_EXPECT(type_is_quantized(TYPE_Q4_0));
    TRC_EXPECT(!type_is_quantized(TYPE_F32));
    TRC_EXPECT(strcmp(type_name(TYPE_F32), "f32") == 0);
}

TRC_TEST(tensor_shape) {
    Context* ctx = context_new(1 << 20);
    Tensor* t = new_tensor_2d(ctx, TYPE_F32, 3, 2);

    TRC_EXPECT_EQ(t->ne[0], 3);
    TRC_EXPECT_EQ(t->ne[1], 2);
    TRC_EXPECT_EQ(t->ne[2], 1);
    TRC_EXPECT_EQ(t->nb[0], 4);
    TRC_EXPECT_EQ(t->nb[1], 12);
    TRC_EXPECT_EQ(t->nb[2], 24);
    TRC_EXPECT_EQ(tensor_nelements(t), 6);
    TRC_EXPECT_EQ(tensor_nrows(t), 2);
    TRC_EXPECT_EQ(tensor_nbytes(t), 24);
    TRC_EXPECT_EQ(tensor_n_dims(t), 2);
    TRC_EXPECT(tensor_is_contiguous(t));

    // f16 形状步长
    Tensor* h = new_tensor_3d(ctx, TYPE_F16, 4, 5, 6);
    TRC_EXPECT_EQ(h->nb[0], 2);
    TRC_EXPECT_EQ(h->nb[1], 8);
    TRC_EXPECT_EQ(h->nb[2], 40);
    TRC_EXPECT_EQ(tensor_nbytes(h), 8 * 5 * 6);
    TRC_EXPECT_EQ(tensor_n_dims(h), 3);

    context_free(ctx);
}

TRC_TEST(context_arena_growth) {
    Context* ctx = context_new(4096);
    TRC_EXPECT_EQ(context_mem_used(ctx), (size_t) 0);

    for (int i = 0; i < 100; ++i) {
        new_tensor_1d(ctx, TYPE_F32, 16);
    }
    TRC_EXPECT_EQ(context_tensor_count(ctx), 100);
    TRC_EXPECT(context_mem_used(ctx) > 4096);  // 触发了 arena 追加
    TRC_EXPECT(context_mem_size(ctx) >= context_mem_used(ctx));

    context_free(ctx);
}

TRC_TEST(tensor_name_and_dup) {
    Context* ctx = context_new(1 << 20);
    Tensor* t = new_tensor_1d(ctx, TYPE_F32, 8);
    tensor_set_name(t, "w_%d", 3);
    TRC_EXPECT(strcmp(t->name, "w_3") == 0);

    Tensor* d = tensor_dup_meta(ctx, t);
    TRC_EXPECT(tensor_are_same_shape(d, t));
    TRC_EXPECT_EQ(d->type, TYPE_F32);
    TRC_EXPECT(strcmp(d->name, "w_3") == 0);
    TRC_EXPECT(d != t);
    context_free(ctx);
}

// 整图 pre-flight API（M2.1f）：返回第一个当前设备不支持的节点
TRC_TEST(graph_first_unsupported_api) {
    Context* ctx = context_new(1 << 16);
    Device*  dev = device_cpu();
    Graph*   g   = graph_new(ctx);

    // 纯 F32 图：全部支持
    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* r = add(ctx, a, b);
    graph_build_forward_expand(ctx, g, r);
    TRC_EXPECT(graph_first_unsupported(g, dev) == nullptr);

    // 追加 F16 节点：必须报告该节点（CPU 逐元素仅 F32）
    Tensor* ah = new_tensor_1d(ctx, TYPE_F16, 4);
    Tensor* bh = new_tensor_1d(ctx, TYPE_F16, 4);
    Tensor* rh = add(ctx, ah, bh);
    graph_build_forward_expand(ctx, g, rh);
    TRC_EXPECT(graph_first_unsupported(g, dev) == rh);

    graph_free(g);
    context_free(ctx);
}

TRC_TEST(graph_build_order) {
    Context* ctx = context_new(1 << 20);
    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* c = new_tensor_1d(ctx, TYPE_F32, 4);
    a->flags |= TENSOR_FLAG_PARAM;

    Tensor* d = add(ctx, a, b);
    Tensor* e = mul(ctx, d, c);

    Graph* g = graph_new(ctx);
    graph_build_forward_expand(ctx, g, e);
    TRC_EXPECT_EQ(graph_n_nodes(g), 5);
    TRC_EXPECT_EQ(graph_n_leafs(g), 3);

    // 验证拓扑序：d 必须出现在 e 之前，a/b 出现在 d 之前
    int idx_a = -1, idx_b = -1, idx_c = -1, idx_d = -1, idx_e = -1;
    for (int64_t i = 0; i < graph_n_nodes(g); ++i) {
        Tensor* n = graph_node(g, i);
        if (n == a) idx_a = (int) i;
        if (n == b) idx_b = (int) i;
        if (n == c) idx_c = (int) i;
        if (n == d) idx_d = (int) i;
        if (n == e) idx_e = (int) i;
    }
    TRC_EXPECT(idx_a >= 0 && idx_b >= 0 && idx_c >= 0 && idx_d >= 0 && idx_e >= 0);
    TRC_EXPECT(idx_a < idx_d && idx_b < idx_d && idx_d < idx_e);
    TRC_EXPECT(idx_c < idx_e);

    // 重复展开同一节点不应产生重复
    graph_build_forward_expand(ctx, g, e);
    TRC_EXPECT_EQ(graph_n_nodes(g), 5);

    graph_free(g);
    context_free(ctx);
}
