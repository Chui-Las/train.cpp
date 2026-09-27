// train.cpp - 越界索引中止死亡测试（M2.1c）
//
// 语义：索引越界在 CPU 与 Vulkan 上都必须**明确中止**（不得静默跳过）。
//   - 本程序构造 get_rows 越界索引，graph_compute 后进程应中止（退出码非 0）
//   - 若未中止（检查失效）则打印错误并返回 0；CMake 用 WILL_FAIL 反转：
//       中止 = 通过；未中止 = 失败
//   - 无 Vulkan 设备时返回 77，CTest 记为跳过（SKIP_RETURN_CODE）
#include "traincpp/traincpp.h"

#include <cstdio>

using namespace traincpp;

int main() {
    Device* dev = device_by_type(DeviceType::GPU);
    if (dev == nullptr) {
        dev = device_by_type(DeviceType::IGPU);
    }
    if (dev == nullptr) {
        std::printf("跳过：无 Vulkan 设备（TRC_VULKAN 未启用或初始化失败）\n");
        return 77;
    }

    Context* ctx   = context_new(1 << 20);
    Graph*   g     = graph_new(ctx);
    Tensor*  table = new_tensor_2d(ctx, TYPE_F32, 4, 5);
    Tensor*  idx   = new_tensor_1d(ctx, TYPE_I32, 2);
    Tensor*  out   = get_rows(ctx, table, idx);
    Buffer*  buf   = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    const float   vt[20] = {0};
    const int32_t vi[2]  = {7, 0};  // 7 越界（词表大小 5）
    tensor_set(table, vt, 0, sizeof(vt));
    tensor_set(idx, vi, 0, sizeof(vi));

    graph_build_forward_expand(ctx, g, out);
    dev->graph_compute(g);  // 期望在此中止

    std::printf("[错误] 越界索引未中止（检查失效）\n");
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 0;
}
