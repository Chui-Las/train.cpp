// train.cpp - Vulkan 设备上限死亡测试（M2.1g / R5）+ 提交失败诊断死亡测试（FIX-004 / Q28）
//
// 语义：
//   1) storage_range：descriptor range 超过 `maxStorageBufferRange` 属未定义行为（可能静默读出 0），
//      必须在绑定前**明确中止**并打印"需要 X / 上限 Y"。
//      测试用环境变量 TRC_VK_MAX_STORAGE_RANGE=4096 模拟小上限设备（见 CTest 注册），
//      然后对 2048 元素（8192 字节）张量做 add → 期望在 graph_compute 绑定阶段中止
//   2) submit_result：用 TRC_VK_FORCE_SUBMIT_RESULT=<VkResult 数值> 伪造提交失败，
//      校验中止消息含 VkResult 名称/数值（device-lost 另含不可恢复说明）
//   - 未中止则打印错误并返回 0；CTest 经 tests/expect_abort.cmake 判定：中止=通过
//   - 无 Vulkan 设备时返回 77（跳过）
#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace traincpp;

namespace {

Device* find_vulkan_device() {
    Device* dev = device_by_type(DeviceType::GPU);
    if (dev == nullptr) {
        dev = device_by_type(DeviceType::IGPU);
    }
    return dev;
}

int run_storage_range() {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("跳过：无 Vulkan 设备（TRC_VULKAN 未启用或初始化失败）\n");
        return 77;
    }

    Context* ctx = context_new(1 << 22);
    Graph*   g   = graph_new(ctx);
    Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, 2048);  // 8192 字节 > 测试上限 4096
    Tensor*  b   = new_tensor_1d(ctx, TYPE_F32, 2048);
    Tensor*  r   = add(ctx, a, b);

    graph_build_forward_expand(ctx, g, r);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    const std::vector<float> v(2048, 1.5f);
    tensor_set(a, v.data(), 0, v.size() * sizeof(float));
    tensor_set(b, v.data(), 0, v.size() * sizeof(float));

    dev->graph_compute(g);  // 期望在此中止（descriptor range 超过测试上限）

    std::printf("[错误] descriptor range 超过 maxStorageBufferRange 未中止"
                "（检查 TRC_VK_MAX_STORAGE_RANGE 是否生效）\n");
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 0;
}

// FIX-004 / Q28：期望在 compute_end 按 TRC_VK_FORCE_SUBMIT_RESULT 中止并打印 VkResult 名称
int run_submit_result() {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("跳过：无 Vulkan 设备（TRC_VULKAN 未启用或初始化失败）\n");
        return 77;
    }

    Context* ctx = context_new(1 << 20);
    Graph*   g   = graph_new(ctx);
    Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, 128);
    Tensor*  b   = new_tensor_1d(ctx, TYPE_F32, 128);
    Tensor*  r   = add(ctx, a, b);

    graph_build_forward_expand(ctx, g, r);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    const std::vector<float> v(128, 1.0f);
    tensor_set(a, v.data(), 0, v.size() * sizeof(float));
    tensor_set(b, v.data(), 0, v.size() * sizeof(float));

    dev->graph_compute(g);  // 期望在 compute_end 按 TRC_VK_FORCE_SUBMIT_RESULT 中止

    std::printf("[错误] TRC_VK_FORCE_SUBMIT_RESULT 未触发中止（检查钩子是否生效）\n");
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "storage_range") == 0) {
        return run_storage_range();
    }
    if (std::strcmp(mode, "submit_result") == 0) {
        return run_submit_result();
    }
    std::printf("用法: %s <storage_range|submit_result>\n", argc > 0 ? argv[0] : "test_vulkan_limit_guard");
    return 2;
}
