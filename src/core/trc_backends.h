// train.cpp - 后端编译期清单（核心内部使用）
//
// 注册方式说明：
//   静态库中的“自注册全局对象”在 MSVC /OPT:REF 下可能被整体优化掉，
//   因此核心不再依赖后端的静态初始化，而是在设备注册表首次使用时按
//   编译期开关（TRC_USE_*，由 CMake 传入）显式调用各后端入口。
//   这与 ggml 的 ggml-backend-reg.cpp 编译期后端清单一致，行为确定。
#pragma once

namespace traincpp {

class Device;

#if TRC_USE_CPU
// 定义在 src/backends/cpu/trc_cpu.cpp
Device* cpu_device();
#endif

#if TRC_USE_VULKAN
// 定义在 src/backends/vulkan/trc_vulkan.cpp（M1.4 实现）
Device* vulkan_device();
#endif

} // namespace traincpp
