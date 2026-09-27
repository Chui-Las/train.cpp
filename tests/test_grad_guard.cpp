// train.cpp - autograd 构建期守卫死亡测试（M2.1e / R4 + R15）
//
// 语义：多图共享参数、同一图重复反向展开都必须在**构建期明确中止**
//       （否则梯度静默双倍 / 多图 grad 单指针互踩）。
//   - 本程序按参数选择场景，期望在 graph_build_backward_expand 处中止（退出码非 0）
//   - 未中止则打印错误并返回 0；CTest 经 tests/expect_abort.cmake 判定：中止=通过
#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstring>

using namespace traincpp;

namespace {

// 最小可反向图：loss = sum(sqr(w))
Tensor* make_loss(Context* ctx, Tensor* w) {
    return sum(ctx, sqr(ctx, w));
}

int run_repeat() {
    Context* ctx = context_new(1 << 20);
    Tensor*  w   = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* loss = make_loss(ctx, w);
    tensor_set_loss(loss);

    Graph* g = graph_new(ctx);
    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);
    graph_build_backward_expand(g);  // 期望在此中止（R15）

    std::printf("[错误] 同一 Graph 重复反向展开未中止（守卫失效：梯度会静默翻倍）\n");
    graph_free(g);
    context_free(ctx);
    return 0;
}

int run_shared() {
    Context* ctx = context_new(1 << 20);
    Tensor*  w   = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_param(w);
    Tensor* loss1 = make_loss(ctx, w);
    Tensor* loss2 = sum(ctx, neg(ctx, w));
    tensor_set_loss(loss1);
    tensor_set_loss(loss2);

    Graph* g1 = graph_new(ctx);
    Graph* g2 = graph_new(ctx);
    graph_build_forward_expand(ctx, g1, loss1);
    graph_build_forward_expand(ctx, g2, loss2);
    graph_build_backward_expand(g1);
    graph_build_backward_expand(g2);  // 期望在此中止（R4：共享 w）

    std::printf("[错误] 多图共享参数未中止（守卫失效：梯度会互相踩踏）\n");
    graph_free(g2);
    graph_free(g1);
    context_free(ctx);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "repeat") == 0) {
        return run_repeat();
    }
    if (std::strcmp(mode, "shared") == 0) {
        return run_shared();
    }
    std::printf("用法: %s <repeat|shared>\n", argc > 0 ? argv[0] : "test_grad_guard");
    return 2;
}
