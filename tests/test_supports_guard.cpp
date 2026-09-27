// train.cpp - "支持声明"死亡测试（M2.1f / R6）
//
// 语义：后端在 graph_compute 前必须校验节点是否真的被支持；Vulkan 对不支持的类型组合
//       （如 F32→I32 cast）必须**明确中止**，不得静默按 F16 路径错读/错写。
//   - 本程序按参数选择场景，期望 graph_compute 处中止（退出码非 0）
//   - 未中止则打印错误并返回 0；CTest 经 tests/expect_abort.cmake 判定：中止=通过
//   - 无 Vulkan 设备时返回 77（跳过）
#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstring>

using namespace traincpp;

namespace {

Device* find_vulkan_device() {
    Device* dev = device_by_type(DeviceType::GPU);
    if (dev == nullptr) {
        dev = device_by_type(DeviceType::IGPU);
    }
    return dev;
}

// F32→I32 cast：Vulkan 仅有 F32↔F16 shader，声明即不支持；compute 必须中止
int run_cast_i32() {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("跳过：无 Vulkan 设备（TRC_VULKAN 未启用或初始化失败）\n");
        return 77;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);
    Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor*  r   = cast(ctx, a, TYPE_I32);

    graph_build_forward_expand(ctx, g, r);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    const float va[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_set(a, va, 0, sizeof(va));

    dev->graph_compute(g);  // 期望在此中止（类型组合不被支持）

    std::printf("[错误] F32→I32 cast 未中止（Vulkan 类型校验失效，曾静默错读）\n");
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "cast_i32") == 0) {
        return run_cast_i32();
    }
    std::printf("用法: %s <cast_i32>\n", argc > 0 ? argv[0] : "test_supports_guard");
    return 2;
}
