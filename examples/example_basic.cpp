// train.cpp - 最小示例：构造 y = sigmoid(a * b) 并计算
#include "traincpp/traincpp.h"

#include <cstdio>

using namespace traincpp;

int main() {
    // 1. 创建上下文（元数据 arena）与计算图
    Context* ctx = context_new(1 << 20);
    Graph* graph = graph_new(ctx);

    // 2. 创建张量并构建前向图
    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* y = sigmoid(ctx, mul(ctx, a, b));

    // 3. 在 CPU 后端分配内存
    Device* dev = device_cpu();
    if (dev == nullptr) {
        std::fprintf(stderr, "CPU 后端未启用\n");
        return 1;
    }
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    // 4. 写入输入数据
    const float va[4] = {0.5f, -1.0f, 2.0f, -0.25f};
    const float vb[4] = {2.0f, 3.0f, -1.0f, 4.0f};
    tensor_set(a, va, 0, sizeof(va));
    tensor_set(b, vb, 0, sizeof(vb));

    // 5. 计算
    graph_build_forward_expand(ctx, graph, y);
    dev->graph_compute(graph);

    // 6. 读取输出
    float out[4] = {0};
    tensor_get(y, out, 0, sizeof(out));
    for (int i = 0; i < 4; ++i) {
        std::printf("y[%d] = %.6f\n", i, out[i]);
    }

    buffer_free(buf);
    graph_free(graph);
    context_free(ctx);
    return 0;
}
