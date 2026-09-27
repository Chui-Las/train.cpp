// train.cpp - 图级分配覆盖校验死亡测试（M2.2b / R10）
//
// 语义：buffer_alloc_graph_tensors 要求 Graph 覆盖 ctx 全部待分配的非视图张量
//       （否则无法计算生命周期，可能静默复用错内存）。图未覆盖时必须明确中止。
//   - 本程序期望在 buffer_alloc_graph_tensors 处中止（退出码非 0）
//   - 未中止则打印错误并返回 0；CTest 经 tests/expect_abort.cmake 判定：中止=通过
#include "traincpp/traincpp.h"

#include <cstdio>

using namespace traincpp;

int main() {
    Device* dev = device_cpu();
    if (dev == nullptr) {
        std::printf("跳过：CPU 后端不可用\n");
        return 77;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);

    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* c = add(ctx, a, b);
    graph_build_forward_expand(ctx, g, c);

    // 不参与图的张量：覆盖校验应中止
    Tensor* orphan = new_tensor_1d(ctx, TYPE_F32, 4);
    (void) orphan;

    Buffer* buf = buffer_alloc_graph_tensors(g, dev->default_buffer_type());

    std::printf("[错误] 图未覆盖张量但未中止（覆盖校验失效，可能静默复用错内存）\n");
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 0;
}
