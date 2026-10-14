// train.cpp - Vulkan 后端（M1.4b：管线/描述符/逐元素 kernel 基建）
#define _CRT_SECURE_NO_WARNINGS  // getenv 用于可选环境变量，无需 MSVC 安全版本
#include "trc_vulkan.h"

#include "core/trc_impl.h"
#include "vulkan_shaders_gen.h"  // 构建期由 cmake/embed_spirv.cmake 生成

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string_view>

namespace traincpp {

// FIX-004 / Q28：失败时打印 VkResult 名称/数值（trc_vk_fail 定义见下）
#define TRC_VK_CHECK(expr, ...)                    \
    do {                                           \
        const VkResult trc_vk_res_ = (expr);       \
        if (trc_vk_res_ != VK_SUCCESS) {           \
            trc_vk_fail(trc_vk_res_, __VA_ARGS__); \
        }                                          \
    } while (0)

namespace {

constexpr size_t VK_BUFFER_ALIGN_MIN = 64;
constexpr uint32_t VK_DESC_POOL_SETS = 512;
constexpr uint32_t TRC_VK_ERROR_U32 = 4;  // 设备端错误标志 uint 数（flag/index/op/保留）

const char* vk_result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS:                        return "VK_SUCCESS";
        case VK_NOT_READY:                      return "VK_NOT_READY";
        case VK_TIMEOUT:                        return "VK_TIMEOUT";
        case VK_EVENT_SET:                      return "VK_EVENT_SET";
        case VK_EVENT_RESET:                    return "VK_EVENT_RESET";
        case VK_INCOMPLETE:                     return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:       return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:    return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:              return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:        return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:        return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:    return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:      return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:      return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:         return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:          return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN:                  return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY:       return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:  return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        default:                                return "VK_UNKNOWN_RESULT";
    }
}

// FIX-004 / Q28：统一失败消息 = VkResult 名称/数值 + 原消息
// （VkResult 放最前：纯 ASCII 前缀，便于日志/工具匹配，避免中文标点邻接时的编码歧义）
[[noreturn]] void trc_vk_fail(VkResult result, const char* fmt, ...) {
    char    msg[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    TRC_ABORT("VkResult=%s/%d：%s", vk_result_name(result), (int) result, msg);
}

// FIX-004 / Q28：提交/等待失败专用诊断（device-lost 给出不可恢复说明与缓解建议）
[[noreturn]] void trc_vk_fail_submit(VkResult result, const char* what, uint32_t dispatches,
                                     size_t live_bytes) {
    if (result == VK_ERROR_DEVICE_LOST) {
        TRC_ABORT("VkResult=VK_ERROR_DEVICE_LOST/%d：Vulkan 后端 %s 设备丢失（Windows TDR/驱动重置/"
                  "硬件挂起）。本次提交 %u 个 dispatch，进程存活缓冲 %.1f MiB；device-lost 后该设备与全部"
                  "资源失效，进程内无法恢复，训练必须终止。建议：1) 减小单次提交规模（TRC_VK_CHUNK_NODES，"
                  "见 FIX-004）；2) 检查/放宽 TdrDelay 与驱动，或改用 CPU 后端/另一块 GPU；"
                  "3) 复测时附本消息与 TRC_VK_VALIDATE=1 输出。",
                  (int) result, what, dispatches, (double) live_bytes / (1024.0 * 1024.0));
    }
    TRC_ABORT("VkResult=%s/%d：Vulkan 后端 %s 失败；本次提交 %u 个 dispatch，进程存活缓冲 %.1f MiB",
              vk_result_name(result), (int) result, what, dispatches,
              (double) live_bytes / (1024.0 * 1024.0));
}

bool has_instance_layer(const char* name) {
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkLayerProperties> layers(count);
    if (vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) {
        return false;
    }
    for (const VkLayerProperties& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) {
            return true;
        }
    }
    return false;
}

bool has_instance_extension(const char* name) {
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> exts(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data()) != VK_SUCCESS) {
        return false;
    }
    for (const VkExtensionProperties& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* /*user_data*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        TRC_LOG_ERROR("[vulkan] %s", data->pMessage);
    } else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        TRC_LOG_WARN("[vulkan] %s", data->pMessage);
    } else {
        TRC_LOG_DEBUG("[vulkan] %s", data->pMessage);
    }
    return VK_FALSE;
}

int device_type_score(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 100;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 50;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 20;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 5;
        default:                                     return 1;
    }
}

bool find_compute_queue_family(VkPhysicalDevice phys, uint32_t& family_out) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    if (count == 0) {
        return false;
    }
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && families[i].queueCount > 0) {
            family_out = i;
            return true;
        }
    }
    return false;
}

// 内存类型偏好评分（FIX-003 / Q26）：device-local（VRAM）优先，其次 host-cached
// （CPU 读更快；优化器与 tensor_get 走主机端标量读写）。required 已含 HOST_VISIBLE|HOST_COHERENT。
int memory_type_score(VkMemoryPropertyFlags flags) {
    int score = 0;
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
        score += 2;
    }
    if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) {
        score += 1;
    }
    return score;
}

// 在满足 required 的候选类型中取评分最高者（同分取先出现者）。
// prefer_device_local=false 等价旧行为：返回第一个匹配类型。
uint32_t find_memory_type(const VulkanState& s, uint32_t type_bits, VkMemoryPropertyFlags required,
                          bool prefer_device_local = false) {
    uint32_t best       = UINT32_MAX;
    int      best_score = -1;
    for (uint32_t i = 0; i < s.mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) {
            continue;
        }
        const VkMemoryPropertyFlags flags = s.mem_props.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) {
            continue;
        }
        const int score = prefer_device_local ? memory_type_score(flags) : 0;
        if (score > best_score) {
            best       = i;
            best_score = score;
        }
    }
    return best;
}

// 计算缓冲默认内存类型（M4.5e）：优先 device-local。
//   ① 无可用"可映射 device-local" → 不可映射 device-local（真显存 + staging）；
//   ② 可映射与不可映射 device-local 同在：仅当不可映射堆显著更大（独显小 BAR 窗 vs 全显存）才选它，
//      否则选可映射（核显统一内存/ReBAR，避免无谓 staging）。
//   ③ 无 device-local → host-visible。
uint32_t default_compute_memory_type(const VulkanState& s, uint32_t type_bits);

// 打印设备内存类型/堆（FIX-003 / Q26；2026-09-27 默认回退 host-visible；M4.5e 默认切回优先 device-local）。
void log_memory_types(const VulkanState& s) {
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint32_t host_type = find_memory_type(s, ~0u, required, false);
    const uint32_t dl_map    = find_memory_type(s, ~0u, required, true);  // TRC_VK_DEVICE_LOCAL=1 时
    const uint32_t def_type  = default_compute_memory_type(s, ~0u);
    const bool def_unmapped =
        def_type != UINT32_MAX &&
        (s.mem_props.memoryTypes[def_type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0;
    for (uint32_t i = 0; i < s.mem_props.memoryTypeCount; ++i) {
        const VkMemoryType& t = s.mem_props.memoryTypes[i];
        const VkMemoryHeap& h = s.mem_props.memoryHeaps[t.heapIndex];
        const char* tag = (i == def_type && def_unmapped) ? " [计算缓冲默认：真显存+staging]"
                          : (i == def_type)              ? " [计算缓冲默认]"
                          : (i == dl_map && i != def_type) ? " [TRC_VK_DEVICE_LOCAL=1 时优先]"
                          : (i == host_type && i != def_type) ? " [host-visible 基线]"
                                                              : "";
        TRC_LOG_INFO("[vulkan] memory type %u: flags=0x%03x heap=%u（%.0f MiB%s）%s%s%s%s%s", i,
                     (unsigned) t.propertyFlags, t.heapIndex, (double) h.size / (1024.0 * 1024.0),
                     (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? ", device-local heap" : "",
                     (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
                     (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
                     (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
                     (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " HOST_CACHED" : "", tag);
    }
}

// 不可映射的 device-local 内存类型（M4.4b）：DEVICE_LOCAL 且无 HOST_VISIBLE。
uint32_t find_device_local_unmappable(const VulkanState& s, uint32_t type_bits) {
    for (uint32_t i = 0; i < s.mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) {
            continue;
        }
        const VkMemoryPropertyFlags f = s.mem_props.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            return i;
        }
    }
    return UINT32_MAX;
}

// 计算缓冲默认内存类型（M4.5e）：优先 device-local（见 log_memory_types 上方说明）
uint32_t default_compute_memory_type(const VulkanState& s, uint32_t type_bits) {
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint32_t dl_map   = find_memory_type(s, type_bits, required, /*prefer_device_local=*/true);
    const bool     map_is_dl = dl_map != UINT32_MAX &&
                               (s.mem_props.memoryTypes[dl_map].propertyFlags &
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    const uint32_t dl_unmap = find_device_local_unmappable(s, type_bits);

    if (dl_unmap == UINT32_MAX) {
        return map_is_dl ? dl_map : find_memory_type(s, type_bits, required, false);
    }
    if (!map_is_dl) {
        return dl_unmap;  // 无可映射 device-local → 真显存 + staging
    }
    // 两者都存在：仅当不可映射堆显著更大（独显小 BAR 窗 vs 全显存）才选不可映射；
    // 统一内存（核显）两堆同源 → 选可映射，避免无谓 staging。
    const VkDeviceSize h_unmap =
        s.mem_props.memoryHeaps[s.mem_props.memoryTypes[dl_unmap].heapIndex].size;
    const VkDeviceSize h_map = s.mem_props.memoryHeaps[s.mem_props.memoryTypes[dl_map].heapIndex].size;
    return h_unmap > h_map ? dl_unmap : dl_map;
}

// ---------------- 共享 staging（M4.4b）----------------
// 不可映射 device-local 缓冲的主机↔设备传输：一个 host-visible、常驻映射的共享 staging 缓冲 +
// 专用 transfer 命令池/fence；每次同步拷贝（提交并等待）。只承载初始化/IO/checkpoint 等低频块，
// 不进入每步热路径（每步重活为设备端 kernel）。

// 确保 staging 缓冲至少 size 字节；返回是否就绪（失败中止）。
void staging_ensure(size_t size) {
    VulkanState& s = vulkan_state();
    const size_t need = size == 0 ? VK_BUFFER_ALIGN_MIN : size;
    if (s.stage_mapped != nullptr && s.stage_size >= need) {
        return;
    }
    if (s.stage_mapped != nullptr) {
        vkUnmapMemory(s.device, s.stage_memory);
        vkFreeMemory(s.device, s.stage_memory, nullptr);
        vkDestroyBuffer(s.device, s.stage_buffer, nullptr);
        s.stage_buffer = VK_NULL_HANDLE;
        s.stage_memory = VK_NULL_HANDLE;
        s.stage_mapped = nullptr;
        s.stage_size   = 0;
    }

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = need;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    TRC_VK_CHECK(vkCreateBuffer(s.device, &bci, nullptr, &s.stage_buffer),
                 "Vulkan staging：缓冲创建失败（%zu 字节）", need);

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(s.device, s.stage_buffer, &req);
    const uint32_t mt = find_memory_type(
        s, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        /*prefer_device_local=*/false);
    TRC_ASSERT(mt != UINT32_MAX, "Vulkan staging：找不到 host-visible 内存类型");

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mt;
    TRC_VK_CHECK(vkAllocateMemory(s.device, &ai, nullptr, &s.stage_memory),
                 "Vulkan staging：内存分配失败（%zu 字节）", (size_t) req.size);
    TRC_VK_CHECK(vkBindBufferMemory(s.device, s.stage_buffer, s.stage_memory, 0),
                 "Vulkan staging：绑定失败");
    TRC_VK_CHECK(vkMapMemory(s.device, s.stage_memory, 0, VK_WHOLE_SIZE, 0, &s.stage_mapped),
                 "Vulkan staging：映射失败");
    s.stage_size = need;
}

void transfer_begin(VkCommandBuffer& cmd) {
    VulkanState& s = vulkan_state();
    if (s.transfer_pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = s.queue_family;
        TRC_VK_CHECK(vkCreateCommandPool(s.device, &pci, nullptr, &s.transfer_pool),
                     "Vulkan staging：transfer 命令池创建失败");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        TRC_VK_CHECK(vkCreateFence(s.device, &fci, nullptr, &s.transfer_fence),
                     "Vulkan staging：transfer fence 创建失败");
    }
    vkResetCommandPool(s.device, s.transfer_pool, 0);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = s.transfer_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    TRC_VK_CHECK(vkAllocateCommandBuffers(s.device, &ai, &cmd), "Vulkan staging：命令缓冲分配失败");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    TRC_VK_CHECK(vkBeginCommandBuffer(cmd, &bi), "Vulkan staging：命令缓冲 begin 失败");
}

void transfer_end(VkCommandBuffer cmd) {
    VulkanState& s = vulkan_state();
    TRC_VK_CHECK(vkEndCommandBuffer(cmd), "Vulkan staging：命令缓冲 end 失败");
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    TRC_VK_CHECK(vkResetFences(s.device, 1, &s.transfer_fence), "Vulkan staging：fence 重置失败");
    const VkResult r = vkQueueSubmit(s.queue, 1, &si, s.transfer_fence);
    if (r != VK_SUCCESS) {
        trc_vk_fail_submit(r, "staging 传输提交", 0, s.live_buffer_bytes);
    }
    const VkResult w = vkWaitForFences(s.device, 1, &s.transfer_fence, VK_TRUE, UINT64_MAX);
    if (w != VK_SUCCESS) {
        trc_vk_fail_submit(w, "staging 传输等待", 0, s.live_buffer_bytes);
    }
    vkFreeCommandBuffers(s.device, s.transfer_pool, 1, &cmd);
}

// 主机 → 设备：写 staging 后拷入 dst[dst_off, dst_off+size)
void staging_upload(VkBuffer dst, size_t dst_off, const void* src, size_t size) {
    staging_ensure(size);
    std::memcpy(vulkan_state().stage_mapped, src, size);
    VkCommandBuffer cmd;
    transfer_begin(cmd);
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = (VkDeviceSize) dst_off;
    region.size      = (VkDeviceSize) size;
    vkCmdCopyBuffer(cmd, vulkan_state().stage_buffer, dst, 1, &region);
    transfer_end(cmd);
}

// 设备 → 主机：拷出 src[src_off, src_off+size) 到 staging 后读回
void staging_download(VkBuffer src, size_t src_off, void* dst, size_t size) {
    staging_ensure(size);
    VkCommandBuffer cmd;
    transfer_begin(cmd);
    VkBufferCopy region{};
    region.srcOffset = (VkDeviceSize) src_off;
    region.dstOffset = 0;
    region.size      = (VkDeviceSize) size;
    vkCmdCopyBuffer(cmd, src, vulkan_state().stage_buffer, 1, &region);
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    transfer_end(cmd);
    std::memcpy(dst, vulkan_state().stage_mapped, size);
}

// 设备端填充 dst[off, off+size) = value（不可映射缓冲的 clear）
void staging_fill(VkBuffer dst, size_t off, size_t size, uint32_t value) {
    TRC_ASSERT(off % 4 == 0 && size % 4 == 0, "Vulkan staging fill 要求 4 字节对齐");
    VkCommandBuffer cmd;
    transfer_begin(cmd);
    vkCmdFillBuffer(cmd, dst, (VkDeviceSize) off, (VkDeviceSize) size, value);
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
    transfer_end(cmd);
}

// 分块提交节点数（FIX-004 / Q28）：TRC_VK_CHUNK_NODES=n，默认 0 = 整图单次提交。
// 每 n 个节点 end/submit/wait 一次再续录，把秒级长提交拆小以降低 Windows TDR 暴露。
uint32_t vulkan_chunk_nodes() {
    const char* env = std::getenv("TRC_VK_CHUNK_NODES");
    if (env == nullptr || *env == '\0') {
        return 0;
    }
    char*      end = nullptr;
    const long v   = std::strtol(env, &end, 10);
    if (end == env || v <= 0) {
        return 0;
    }
    return v > 0x7fffffffL ? 0x7fffffffu : (uint32_t) v;
}

bool init_instance(VulkanState& s) {
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "traincpp";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.pEngineName        = "traincpp";
    app.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion         = VK_API_VERSION_1_2;

    std::vector<const char*> layers;
    std::vector<const char*> exts;
    const bool validate = std::getenv("TRC_VK_VALIDATE") != nullptr;
    if (validate && has_instance_layer("VK_LAYER_KHRONOS_validation")) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    if (validate && has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app;
    ci.enabledLayerCount       = (uint32_t) layers.size();
    ci.ppEnabledLayerNames     = layers.empty() ? nullptr : layers.data();
    ci.enabledExtensionCount   = (uint32_t) exts.size();
    ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

    if (vkCreateInstance(&ci, nullptr, &s.instance) != VK_SUCCESS) {
        TRC_LOG_WARN("Vulkan 后端：vkCreateInstance 失败（无运行库/驱动？）");
        return false;
    }

    if (!exts.empty()) {
        auto create_messenger = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
            s.instance, "vkCreateDebugUtilsMessengerEXT");
        if (create_messenger != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debug_callback;
            create_messenger(s.instance, &dci, nullptr, &s.debug_messenger);
        }
    }
    return true;
}

bool init_physical_device(VulkanState& s) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(s.instance, &count, nullptr) != VK_SUCCESS || count == 0) {
        TRC_LOG_WARN("Vulkan 后端：未找到物理设备");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(s.instance, &count, devices.data());

    // TRC_VK_DEVICE=<序号> 或 <名称子串> 可覆盖自动选择
    const char* env          = std::getenv("TRC_VK_DEVICE");
    int         want_index   = -1;
    std::string want_name;
    if (env != nullptr) {
        char*       end = nullptr;
        const long  idx = std::strtol(env, &end, 10);
        if (end != env && *end == '\0') {
            want_index = (int) idx;
        } else {
            want_name = env;
        }
    }

    VkPhysicalDevice         best        = VK_NULL_HANDLE;
    int                      best_score  = -1;
    uint32_t                 best_family = 0;
    VkPhysicalDeviceProperties best_props{};
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (want_index >= 0 && (uint32_t) want_index != i) {
            continue;
        }
        if (!want_name.empty() && std::strstr(props.deviceName, want_name.c_str()) == nullptr) {
            continue;
        }
        uint32_t family = 0;
        if (!find_compute_queue_family(devices[i], family)) {
            continue;
        }
        const int score = device_type_score(props.deviceType);
        if (score > best_score) {
            best        = devices[i];
            best_score  = score;
            best_family = family;
            best_props  = props;
        }
    }
    if (best == VK_NULL_HANDLE) {
        TRC_LOG_WARN("Vulkan 后端：没有满足条件的计算设备（TRC_VK_DEVICE=%s）", env ? env : "(未设置)");
        return false;
    }

    s.physical_device = best;
    s.queue_family    = best_family;
    s.props           = best_props;
    vkGetPhysicalDeviceMemoryProperties(best, &s.mem_props);

    s.min_storage_offset_align =
        (uint32_t) std::max<VkDeviceSize>(1, s.props.limits.minStorageBufferOffsetAlignment);
    s.max_storage_buffer_range = (size_t) s.props.limits.maxStorageBufferRange;

    // maxBufferSize / maxMemoryAllocationSize（R5 校验用）：
    //   maxMemoryAllocationSize 来自 VkPhysicalDeviceMaintenance3Properties（Vulkan 1.1 核心）；
    //   maxBufferSize 来自 VkPhysicalDeviceMaintenance4Properties（Vulkan 1.3 / VK_KHR_maintenance4）。
    //   未报告时保持 0，表示该项不做单独校验（由另一项/驱动兜底）。
    s.max_buffer_size            = 0;
    s.max_memory_allocation_size = 0;
    {
        VkPhysicalDeviceMaintenance3Properties maint3{};
        VkPhysicalDeviceMaintenance4Properties maint4{};
        void*                                  chain = nullptr;
        if (s.props.apiVersion >= VK_API_VERSION_1_3) {
            maint4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
            maint4.pNext = chain;
            chain        = &maint4;
        }
        if (s.props.apiVersion >= VK_API_VERSION_1_1) {
            maint3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
            maint3.pNext = chain;
            chain        = &maint3;
        }
        if (chain != nullptr) {
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = chain;
            vkGetPhysicalDeviceProperties2(best, &props2);
            if (s.props.apiVersion >= VK_API_VERSION_1_3) {
                s.max_buffer_size = (size_t) maint4.maxBufferSize;
            }
            if (s.props.apiVersion >= VK_API_VERSION_1_1) {
                s.max_memory_allocation_size = (size_t) maint3.maxMemoryAllocationSize;
            }
        }
    }
    s.integrated               = (best_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    s.unified_memory           = s.integrated;
    s.name                     = best_props.deviceName;
    s.driver                   = std::to_string(VK_VERSION_MAJOR(best_props.driverVersion)) + "." +
                                 std::to_string(VK_VERSION_MINOR(best_props.driverVersion)) + "." +
                                 std::to_string(VK_VERSION_PATCH(best_props.driverVersion));

    s.memory_total = 0;
    for (uint32_t i = 0; i < s.mem_props.memoryHeapCount; ++i) {
        if ((s.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            s.memory_total += (size_t) s.mem_props.memoryHeaps[i].size;
        }
    }

    // 16 位存储（F16 cast 需要；Vulkan 1.1 核心特性，仍需查询并显式启用）
    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &storage16;
    vkGetPhysicalDeviceFeatures2(best, &features2);
    s.storage_16bit = storage16.storageBuffer16BitAccess == VK_TRUE;
    return true;
}

bool init_device(VulkanState& s) {
    const float             priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = s.queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDevice16BitStorageFeatures storage16{};
    storage16.sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16.storageBuffer16BitAccess = s.storage_16bit ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo dci{};
    dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    // 仅启用 16 位存储（F16 cast 需要）；无此特性的设备在 supports_op 中关闭 cast
    dci.pNext = s.storage_16bit ? &storage16 : nullptr;

    if (vkCreateDevice(s.physical_device, &dci, nullptr, &s.device) != VK_SUCCESS) {
        TRC_LOG_WARN("Vulkan 后端：vkCreateDevice 失败");
        return false;
    }
    vkGetDeviceQueue(s.device, s.queue_family, 0, &s.queue);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = s.queue_family;
    TRC_VK_CHECK(vkCreateCommandPool(s.device, &pci, nullptr, &s.command_pool),
                 "Vulkan 后端：命令池创建失败");

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    TRC_VK_CHECK(vkCreateFence(s.device, &fci, nullptr, &s.fence), "Vulkan 后端：fence 创建失败");

    // descriptor set layout：binding 0..VK_MAX_BINDINGS-1 全为 storage buffer
    std::vector<VkDescriptorSetLayoutBinding> bindings(VK_MAX_BINDINGS);
    for (uint32_t i = 0; i < VK_MAX_BINDINGS; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl{};
    dsl.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = (uint32_t) bindings.size();
    dsl.pBindings    = bindings.data();
    TRC_VK_CHECK(vkCreateDescriptorSetLayout(s.device, &dsl, nullptr, &s.descriptor_layout),
                 "Vulkan 后端：descriptor set layout 创建失败");
    return true;
}

// 设备端错误标志缓冲：host-visible + coherent 常驻映射（index shader 的 binding=3）
bool init_error_buffer(VulkanState& s) {
    const VkDeviceSize bytes = (VkDeviceSize) TRC_VK_ERROR_U32 * sizeof(uint32_t);

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = bytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(s.device, &bci, nullptr, &s.error_buffer) != VK_SUCCESS) {
        TRC_LOG_WARN("Vulkan 后端：错误标志缓冲创建失败");
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(s.device, s.error_buffer, &req);
    const uint32_t mem_type = find_memory_type(
        s, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type == UINT32_MAX) {
        TRC_LOG_WARN("Vulkan 后端：错误标志缓冲无 host-visible 内存类型");
        return false;
    }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mem_type;
    if (vkAllocateMemory(s.device, &ai, nullptr, &s.error_memory) != VK_SUCCESS) {
        TRC_LOG_WARN("Vulkan 后端：错误标志缓冲内存分配失败");
        return false;
    }
    if (vkBindBufferMemory(s.device, s.error_buffer, s.error_memory, 0) != VK_SUCCESS ||
        vkMapMemory(s.device, s.error_memory, 0, VK_WHOLE_SIZE, 0, (void**) &s.error_mapped) !=
            VK_SUCCESS) {
        TRC_LOG_WARN("Vulkan 后端：错误标志缓冲绑定/映射失败");
        return false;
    }
    std::memset(s.error_mapped, 0, (size_t) bytes);
    return true;
}

bool init_state(VulkanState& s) {
    if (!init_instance(s)) {
        return false;
    }
    if (!init_physical_device(s) || !init_device(s)) {
        if (s.device != VK_NULL_HANDLE) {
            vkDestroyDevice(s.device, nullptr);
            s.device = VK_NULL_HANDLE;
        }
        vkDestroyInstance(s.instance, nullptr);
        s.instance = VK_NULL_HANDLE;
        return false;
    }
    if (!init_error_buffer(s)) {
        vkDestroyDevice(s.device, nullptr);
        s.device = VK_NULL_HANDLE;
        vkDestroyInstance(s.instance, nullptr);
        s.instance = VK_NULL_HANDLE;
        return false;
    }
    log_memory_types(s);
    if (std::getenv("TRC_VK_FORCE_STAGING") != nullptr) {
        TRC_LOG_INFO("[vulkan] TRC_VK_FORCE_STAGING=1：计算缓冲走不可映射 device-local + staging"
                     "（无该内存类型时为 host-visible 但不映射，用于本地覆盖 staging 路径）");
    }
    TRC_LOG_INFO("Vulkan 后端就绪：%s（驱动 %s，%s，显存 %.0f MiB，storage offset align=%u，"
                 "maxWorkGroupCount=%u/%u/%u，maxStorageBufferRange=%.0f MiB，maxBufferSize=%.0f MiB，"
                 "maxMemoryAllocationSize=%.0f MiB）",
                 s.name.c_str(), s.driver.c_str(), s.integrated ? "核显" : "独显",
                 (double) s.memory_total / (1024.0 * 1024.0), s.min_storage_offset_align,
                 s.props.limits.maxComputeWorkGroupCount[0], s.props.limits.maxComputeWorkGroupCount[1],
                 s.props.limits.maxComputeWorkGroupCount[2],
                 (double) s.props.limits.maxStorageBufferRange / (1024.0 * 1024.0),
                 (double) s.max_buffer_size / (1024.0 * 1024.0),
                 (double) s.max_memory_allocation_size / (1024.0 * 1024.0));
    return true;
}

void create_descriptor_pool(VulkanState& s) {
    VkDescriptorPoolSize size{};
    size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = VK_DESC_POOL_SETS * VK_MAX_BINDINGS;

    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = VK_DESC_POOL_SETS;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    TRC_VK_CHECK(vkCreateDescriptorPool(s.device, &pci, nullptr, &pool),
                 "Vulkan 后端：descriptor pool 创建失败（maxSets=%u）", VK_DESC_POOL_SETS);
    s.descriptor_pools.push_back(pool);
}

// graph_compute 的生命周期：一个命令缓冲录制整图，最后一次性提交并等待
void compute_begin(VulkanGraphContext& gc) {
    VulkanState& s = vulkan_state();
    vkResetCommandPool(s.device, s.command_pool, 0);

    // FIX-004 / Q28：本次提交的 dispatch 计数（失败诊断上下文）
    s.graph_dispatch_count = 0;

    // 清零设备端错误标志（本次 graph_compute 内发生时在结束时中止）
    if (s.error_mapped != nullptr) {
        std::memset(s.error_mapped, 0, TRC_VK_ERROR_U32 * sizeof(uint32_t));
    }

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = s.command_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    TRC_VK_CHECK(vkAllocateCommandBuffers(s.device, &ai, &gc.cmd), "Vulkan 后端：命令缓冲分配失败");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    TRC_VK_CHECK(vkBeginCommandBuffer(gc.cmd, &bi), "Vulkan 后端：命令缓冲 begin 失败");

    for (VkDescriptorPool pool : s.descriptor_pools) {
        vkResetDescriptorPool(s.device, pool, 0);
    }
    s.descriptor_pool_index = 0;
}

void compute_end(VulkanGraphContext& gc, bool submit) {
    VulkanState& s = vulkan_state();
    if (gc.cmd == VK_NULL_HANDLE) {
        return;
    }
    if (submit) {
        TRC_VK_CHECK(vkEndCommandBuffer(gc.cmd), "Vulkan 后端：命令缓冲 end 失败");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &gc.cmd;

        TRC_VK_CHECK(vkResetFences(s.device, 1, &s.fence), "Vulkan 后端：fence 重置失败");
        // FIX-004 / Q28：提交/等待失败时附 VkResult 与上下文；device-lost 单独说明
        const VkResult submit_res = vkQueueSubmit(s.queue, 1, &si, s.fence);
        if (submit_res != VK_SUCCESS) {
            trc_vk_fail_submit(submit_res, "vkQueueSubmit", s.graph_dispatch_count, s.live_buffer_bytes);
        }
        const VkResult wait_res = vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        if (wait_res != VK_SUCCESS) {
            trc_vk_fail_submit(wait_res, "vkWaitForFences", s.graph_dispatch_count, s.live_buffer_bytes);
        }

        // 测试钩子（FIX-004 / Q28）：伪造提交结果以验证诊断路径
        //（TRC_VK_FORCE_SUBMIT_RESULT=<VkResult 数值>；仅死亡测试使用）
        if (const char* force = std::getenv("TRC_VK_FORCE_SUBMIT_RESULT");
            force != nullptr && *force != '\0') {
            trc_vk_fail_submit((VkResult) std::strtol(force, nullptr, 10), "vkQueueSubmit",
                               s.graph_dispatch_count, s.live_buffer_bytes);
        }

        // 索引越界等设备端错误：与 CPU 后端的 abort 语义统一（不再静默跳过）
        if (s.error_mapped != nullptr && s.error_mapped[0] != 0) {
            const uint32_t code = s.error_mapped[2];
            const char*    op_desc = code == 1u ? "get_rows"
                                     : code == 2u ? "set_rows"
                                     : code == 3u ? "get_rows_back"
                                                  : "索引算子";
            TRC_ABORT("Vulkan 后端：%s 索引越界（索引值 %d，词表/目标行数见张量形状）——"
                      "CPU 后端同样中止，请检查索引数据范围",
                      op_desc, (int32_t) s.error_mapped[1]);
        }
    }
    vkFreeCommandBuffers(s.device, s.command_pool, 1, &gc.cmd);
    gc.cmd = VK_NULL_HANDLE;
}

const vk_shaders::SpirvBinary* find_spirv(const char* name) {
    for (const vk_shaders::SpirvBinary& b : vk_shaders::kAll) {
        if (std::strcmp(b.name, name) == 0) {
            return &b;
        }
    }
    return nullptr;
}

class VulkanDevice final : public Device {
public:
    VulkanDevice() {
        const VulkanState& s = vulkan_state();
        props_.name           = std::string("Vulkan: ") + s.name;
        props_.description    = "Vulkan 后端（M1.4b：逐元素/一元 kernel）";
        props_.type           = s.integrated ? DeviceType::IGPU : DeviceType::GPU;
        props_.memory_total   = s.memory_total;
        props_.memory_free    = s.memory_total;  // 精确值需 VK_EXT_memory_budget，后续再加
        props_.unified_memory = s.unified_memory;
        buft_ = new VulkanBufferType(this);
    }
    ~VulkanDevice() override { delete buft_; }

    const DeviceProps& props() const override { return props_; }
    BufferType*        default_buffer_type() override { return buft_; }

    bool supports_op(const Tensor* t) const override {
        TRC_ASSERT(t != nullptr, "VulkanDevice::supports_op: 张量为空");
        return vulkan_supports_op(t);
    }

    bool graph_compute(Graph* graph) override {
        TRC_ASSERT(graph != nullptr, "VulkanDevice::graph_compute: graph 为空");
        VulkanGraphContext gc;
        compute_begin(gc);
        // FIX-004 / Q28：可选分块提交（TRC_VK_CHUNK_NODES=n，0=整图单次提交）。
        // 每 n 个节点 end/submit/wait 一次再续录；每块提交后已等待，命令池/descriptor 池重置安全。
        // 注意：分块模式下若中途遇到内部"不支持节点"（`graph_first_unsupported` 预检兜底），
        // 已提交的块会保留执行；默认（0）行为与旧版完全一致。
        const uint32_t chunk = vulkan_chunk_nodes();
        const size_t   n     = graph->nodes.size();
        size_t         i     = 0;
        for (;;) {
            const size_t stop = chunk == 0 ? n : std::min(n, i + (size_t) chunk);
            for (; i < stop; ++i) {
                if (!vulkan_compute_node(gc, graph->nodes[i])) {
                    compute_end(gc, false);
                    return false;
                }
            }
            compute_end(gc, true);  // 空图 = 空提交（保持既有行为）
            if (i >= n) {
                break;
            }
            compute_begin(gc);
        }
        return true;
    }

    // 批量填充（M4.2）：单次提交内对每个 F32 连续张量执行 vkCmdFillBuffer，
    // 末尾一个 transfer-write → shader 屏障，保证后续 compute 可见。
    bool fill_tensors(const TensorFill* fills, size_t count) override {
        if (count == 0) {
            return true;
        }
        VulkanGraphContext gc;
        compute_begin(gc);
        bool any = false;
        for (size_t i = 0; i < count; ++i) {
            Tensor* t = fills[i].tensor;
            TRC_ASSERT(t != nullptr && t->type == TYPE_F32, "Vulkan fill_tensors 仅支持 F32 张量");
            TRC_ASSERT(tensor_is_contiguous(t), "Vulkan fill_tensors 要求张量 %s 连续", t->name);

            Buffer*       buf   = t->buffer;
            const Tensor* owner = t;
            while (buf == nullptr && owner->view_src != nullptr) {
                owner = owner->view_src;
                buf   = owner->buffer;
            }
            TRC_ASSERT(buf != nullptr, "Vulkan fill_tensors：张量 %s 未绑定 buffer", t->name);

            const size_t bytes = (size_t) tensor_nelements(t) * sizeof(float);
            if (bytes == 0) {
                continue;
            }
            // M4.4：按 data_offset 定位（不可映射缓冲同样适用）
            const size_t offset = t->data_offset;
            uint32_t pattern = 0;
            std::memcpy(&pattern, &fills[i].value, sizeof(float));
            vkCmdFillBuffer(gc.cmd, static_cast<const VulkanBuffer*>(buf)->vk_buffer(),
                            (VkDeviceSize) offset, (VkDeviceSize) bytes, pattern);
            any = true;
        }
        if (any) {
            VkMemoryBarrier mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(gc.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0,
                                 nullptr);
        }
        compute_end(gc, true);
        return true;
    }

private:
    DeviceProps       props_;
    VulkanBufferType* buft_ = nullptr;
};

} // namespace

// ---------------------------------------------------------------- Buffer 实现

VulkanBuffer* VulkanBuffer::create(VulkanBufferType* buft, size_t size) {
    const VulkanState& s = vulkan_state();
    const size_t alloc_size = size == 0 ? VK_BUFFER_ALIGN_MIN : size;

    // R5：单缓冲不得超过设备 maxBufferSize（超限时驱动报错信息不清晰，这里明确中止；0=设备未报告）
    const size_t max_buffer_size = s.max_buffer_size;
    if (max_buffer_size != 0 && alloc_size > max_buffer_size) {
        TRC_ABORT("Vulkan 后端：请求缓冲 %zu 字节（%.1f MiB）超过设备 maxBufferSize 上限 %zu 字节"
                  "（%.1f MiB）；请减小 batch/segment 或分块（单缓冲上限是设备硬限制，"
                  "device-local/staging 不改变它）",
                  alloc_size, (double) alloc_size / (1024.0 * 1024.0), max_buffer_size,
                  (double) max_buffer_size / (1024.0 * 1024.0));
    }

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = alloc_size;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(s.device, &bci, nullptr, &buffer) != VK_SUCCESS) {
        TRC_LOG_WARN("Vulkan 后端：VkBuffer 创建失败（%zu 字节）", alloc_size);
        return nullptr;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(s.device, buffer, &req);
    // R5：实际分配量（req.size）不得超过 maxMemoryAllocationSize（0=设备未报告）
    const size_t max_alloc_size = s.max_memory_allocation_size;
    if (max_alloc_size != 0 && (size_t) req.size > max_alloc_size) {
        vkDestroyBuffer(s.device, buffer, nullptr);
        TRC_ABORT("Vulkan 后端：缓冲内存需求 %zu 字节（%.1f MiB）超过设备 maxMemoryAllocationSize "
                  "上限 %zu 字节（%.1f MiB）；请减小 batch/segment 或分块",
                  (size_t) req.size, (double) req.size / (1024.0 * 1024.0), max_alloc_size,
                  (double) max_alloc_size / (1024.0 * 1024.0));
    }
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint32_t fallback_type =
        find_memory_type(s, req.memoryTypeBits, required, /*prefer_device_local=*/false);
    if (fallback_type == UINT32_MAX) {
        vkDestroyBuffer(s.device, buffer, nullptr);
        TRC_LOG_WARN("Vulkan 后端：找不到 host-visible+coherent 内存类型（M1.4 需要）");
        return nullptr;
    }
    // 放置策略（2026-09-27 修订 FIX-003 / Q26；2026-10-01 M4.4b 追加 staging；M4.5e 默认切换）：
    //   **默认优先 device-local**（M4.5e，用户拍板"能用 device-local 就用"）：
    //     ① 有"不可映射的 device-local"（独显真显存）→ 用它 + 共享 staging（`base()==nullptr`）；
    //     ② 否则有"可映射的 device-local"（核显/ReBAR）→ 用它并映射（`base()` 可解引用）；
    //     ③ 都没有 → host-visible（系统内存）。
    //   TRC_VK_HOST_MEMORY=1 强制 host-visible（基线/测试/A-B）。
    //   TRC_VK_DEVICE_LOCAL=1 只优先"可映射的 device-local"（A-B 对照，不进入 staging）。
    //   TRC_VK_FORCE_STAGING=1 强制"不可映射 device-local + staging"（测试模式）。
    const char* dl_env   = std::getenv("TRC_VK_DEVICE_LOCAL");
    const char* host_env = std::getenv("TRC_VK_HOST_MEMORY");
    const char* stg_env  = std::getenv("TRC_VK_FORCE_STAGING");
    const bool  want_device_local = dl_env != nullptr && *dl_env != '\0';
    const bool  force_host        = host_env != nullptr && *host_env != '\0';
    const bool  force_staging     = stg_env != nullptr && *stg_env != '\0';

    const uint32_t dl_unmappable = find_device_local_unmappable(s, req.memoryTypeBits);
    uint32_t       preferred_type = fallback_type;
    bool           map_memory     = true;  // 是否映射出可解引用主机指针（false=走 staging）
    if (force_host) {
        // 基线：host-visible（fallback）
    } else if (force_staging) {
        preferred_type = dl_unmappable != UINT32_MAX ? dl_unmappable : fallback_type;
        map_memory     = false;
    } else if (want_device_local) {
        const uint32_t dl = find_memory_type(s, req.memoryTypeBits, required, true);
        if (dl != UINT32_MAX) {
            preferred_type = dl;
        }
    } else {
        // M4.5e 默认：优先 device-local（详见 default_compute_memory_type）
        const uint32_t def = default_compute_memory_type(s, req.memoryTypeBits);
        preferred_type = def != UINT32_MAX ? def : fallback_type;
        map_memory = (s.mem_props.memoryTypes[preferred_type].propertyFlags &
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    const auto     allocate_type = [&](uint32_t mem_type, bool map) -> bool {
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = mem_type;
        if (vkAllocateMemory(s.device, &ai, nullptr, &memory) != VK_SUCCESS) {
            memory = VK_NULL_HANDLE;
            return false;
        }
        if (vkBindBufferMemory(s.device, buffer, memory, 0) != VK_SUCCESS) {
            vkFreeMemory(s.device, memory, nullptr);
            memory = VK_NULL_HANDLE;
            return false;
        }
        if (map && vkMapMemory(s.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            vkFreeMemory(s.device, memory, nullptr);
            memory = VK_NULL_HANDLE;
            mapped = nullptr;
            return false;
        }
        return true;
    };

    bool placed = allocate_type(preferred_type, map_memory);
    if (!placed && preferred_type != fallback_type) {
        TRC_LOG_WARN("Vulkan 后端：首选缓冲分配/映射失败（%zu 字节，memory type %u），回退 host-visible",
                     alloc_size, preferred_type);
        placed = allocate_type(fallback_type, map_memory);
    }
    if (!placed) {
        vkDestroyBuffer(s.device, buffer, nullptr);
        // M2.2a-2：保留"失败返回 nullptr"约定，日志给出需要/上限/设备总量
        TRC_LOG_WARN("Vulkan 后端：显存分配失败：需要 %zu 字节（%.2f MiB），设备总显存 %zu 字节"
                     "（%.2f MiB），maxMemoryAllocationSize=%zu 字节（%.2f MiB）；请减小 batch/segment "
                     "或换设备（上层将明确中止）",
                     (size_t) req.size, (double) req.size / (1024.0 * 1024.0), s.memory_total,
                     (double) s.memory_total / (1024.0 * 1024.0), s.max_memory_allocation_size,
                     (double) s.max_memory_allocation_size / (1024.0 * 1024.0));
        return nullptr;
    }
    if (mapped != nullptr) {
        std::memset(mapped, 0, alloc_size);
    } else if (alloc_size % 4 == 0) {
        // M4.4b：不可映射 device-local 显式设备端清零，保持与 host-visible 一致的"新建缓冲为零"
        staging_fill(buffer, 0, alloc_size, 0);
    }

    return new VulkanBuffer(buft, buffer, memory, mapped, alloc_size);
}

VulkanBuffer::VulkanBuffer(VulkanBufferType* buft, VkBuffer buffer, VkDeviceMemory memory, void* mapped,
                           size_t size)
    : buft_(buft), buffer_(buffer), memory_(memory), mapped_(mapped), size_(size) {
    vulkan_state().live_buffer_bytes += size;  // FIX-004 / Q28：诊断计数
}

VulkanBuffer::~VulkanBuffer() {
    VulkanState& s = vulkan_state();
    if (s.device != VK_NULL_HANDLE) {
        if (mapped_ != nullptr) {
            vkUnmapMemory(s.device, memory_);
        }
        vkFreeMemory(s.device, memory_, nullptr);
        vkDestroyBuffer(s.device, buffer_, nullptr);
    }
    s.live_buffer_bytes = s.live_buffer_bytes >= size_ ? s.live_buffer_bytes - size_ : 0;
}

BufferType* VulkanBuffer::buffer_type() const { return buft_; }

void VulkanBuffer::set_tensor(Tensor* t, size_t offset, const void* data, size_t size) {
    // M4.4：按 data_offset 定位（视图也传入本 buffer）；不可映射缓冲经 staging 上传
    TRC_ASSERT(t != nullptr, "VulkanBuffer::set_tensor: 张量为空");
    const size_t off = t->data_offset + offset;
    if (mapped_ != nullptr) {
        std::memcpy((uint8_t*) mapped_ + off, data, size);
        return;
    }
    staging_upload(buffer_, off, data, size);
}

void VulkanBuffer::get_tensor(const Tensor* t, size_t offset, void* data, size_t size) const {
    TRC_ASSERT(t != nullptr, "VulkanBuffer::get_tensor: 张量为空");
    const size_t off = t->data_offset + offset;
    if (mapped_ != nullptr) {
        std::memcpy(data, (const uint8_t*) mapped_ + off, size);
        return;
    }
    staging_download(buffer_, off, data, size);
}

void VulkanBuffer::clear() {
    if (mapped_ != nullptr) {
        std::memset(mapped_, 0, size_);
        return;
    }
    staging_fill(buffer_, 0, size_, 0);
}

size_t VulkanBufferType::alignment() const {
    const size_t a = (size_t) vulkan_state().min_storage_offset_align;
    return std::max(VK_BUFFER_ALIGN_MIN, a);
}

Buffer* VulkanBufferType::alloc_buffer(size_t size) { return VulkanBuffer::create(this, size); }

// ---------------------------------------------------------------- 管线 / 调度

VulkanPipelineHandle vulkan_pipeline(const char* name, uint32_t push_constant_size) {
    // R13：异构查找（string_view 键）——热路径每次 compute 每节点查找不再构造临时 std::string
    static std::map<std::string, VulkanPipelineHandle, std::less<>> cache;
    const auto it = cache.find(std::string_view(name));
    if (it != cache.end()) {
        return it->second;
    }

    const VulkanState&    s   = vulkan_state();
    const vk_shaders::SpirvBinary* bin = find_spirv(name);
    TRC_ASSERT(bin != nullptr, "Vulkan 后端：找不到 shader %s（构建期未编译？）", name);

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = bin->word_count * sizeof(uint32_t);
    smci.pCode    = bin->words;
    VkShaderModule module = VK_NULL_HANDLE;
    TRC_VK_CHECK(vkCreateShaderModule(s.device, &smci, nullptr, &module),
                 "Vulkan 后端：shader module 创建失败（%s）", name);

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset     = 0;
    range.size       = push_constant_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &s.descriptor_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &range;

    VulkanPipelineHandle handle;
    TRC_VK_CHECK(vkCreatePipelineLayout(s.device, &plci, nullptr, &handle.layout),
                 "Vulkan 后端：pipeline layout 创建失败（%s）", name);

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName  = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = handle.layout;
    TRC_VK_CHECK(vkCreateComputePipelines(s.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &handle.pipeline),
                 "Vulkan 后端：compute pipeline 创建失败（%s）", name);

    vkDestroyShaderModule(s.device, module, nullptr);
    cache[name] = handle;
    return handle;
}

VkDescriptorSet vulkan_alloc_descriptor_set() {
    VulkanState& s = vulkan_state();
    if (s.descriptor_pools.empty()) {
        create_descriptor_pool(s);
    }
    for (;;) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = s.descriptor_pools[s.descriptor_pool_index];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &s.descriptor_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult  r   = vkAllocateDescriptorSets(s.device, &ai, &set);
        if (r == VK_SUCCESS) {
            return set;
        }
        if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
            if (s.descriptor_pool_index + 1 < s.descriptor_pools.size()) {
                ++s.descriptor_pool_index;
                vkResetDescriptorPool(s.device, s.descriptor_pools[s.descriptor_pool_index], 0);
                continue;
            }
            create_descriptor_pool(s);
            s.descriptor_pool_index = s.descriptor_pools.size() - 1;
            continue;
        }
        TRC_ABORT("Vulkan 后端：descriptor set 分配失败 (VkResult=%d)", (int) r);
    }
}

VulkanTensorBinding vulkan_bind_tensor(const Tensor* t) {
    const VulkanState& s = vulkan_state();
    TRC_ASSERT(t->type == TYPE_F32 || t->type == TYPE_I32 || t->type == TYPE_F16,
               "Vulkan 后端仅支持 F32/I32/F16 类型（%s，类型 %s）", t->name, type_name(t->type));

    Buffer*       buf   = t->buffer;  // base() 为非 const 接口（不修改内容）
    const Tensor* owner = t;
    while (buf == nullptr && owner->view_src != nullptr) {
        owner = owner->view_src;
        buf   = owner->buffer;
    }
    TRC_ASSERT(buf != nullptr, "Vulkan 后端：张量 %s 尚未绑定 buffer", t->name);

    const size_t elem   = type_size(t->type);
    // M4.4：descriptor 偏移直接取 data_offset（绝对，缓冲内字节；不可映射缓冲无 data 指针）
    const size_t offset = t->data_offset;
    // 注意：不能用 tensor_nbytes —— 它假设 nb[0]=type_size，对转置等视图会低估跨度
    // （与核心 tensor_last_element_offset 的越界检查同思路，见 AGENTS.md 已知坑）
    const size_t bytes = (size_t) (t->ne[0] - 1) * t->nb[0] + (size_t) (t->ne[1] - 1) * t->nb[1] +
                         (size_t) (t->ne[2] - 1) * t->nb[2] + (size_t) (t->ne[3] - 1) * t->nb[3] +
                         elem;
    const size_t align  = (size_t) s.min_storage_offset_align;
    const size_t desc_offset = offset & ~(align - 1);
    const size_t misalign    = offset - desc_offset;
    TRC_ASSERT(misalign % elem == 0,
               "Vulkan 后端要求张量数据按类型宽度对齐（%s, offset=%zu, elem=%zu）", t->name, offset,
               elem);
    TRC_ASSERT(desc_offset + misalign + bytes <= static_cast<const VulkanBuffer*>(buf)->size(),
               "Vulkan 后端：张量 %s 超出缓冲范围", t->name);

    // R5：descriptor range 超过 maxStorageBufferRange 属未定义行为（通常静默读出 0），必须中止
    const size_t range     = misalign + bytes;
    const size_t max_range = vulkan_max_storage_range();
    TRC_ASSERT(range <= max_range,
               "Vulkan 后端：张量 %s 的 descriptor range %zu 字节（%.2f MiB）超过 "
               "maxStorageBufferRange 上限 %zu 字节（%.2f MiB）；请减小 batch/segment 或分块"
               "（单 descriptor 上限是设备硬限制）",
               t->name[0] != '\0' ? t->name : "<未命名>", range, (double) range / (1024.0 * 1024.0),
               max_range, (double) max_range / (1024.0 * 1024.0));

    VulkanTensorBinding b;
    b.info.buffer = static_cast<const VulkanBuffer*>(buf)->vk_buffer();
    b.info.offset = desc_offset;
    b.info.range  = misalign + bytes;
    b.base_elem   = (uint32_t) (misalign / elem);
    return b;
}

VulkanTensorBinding vulkan_bind_error_buffer() {
    const VulkanState& s = vulkan_state();
    VulkanTensorBinding b;
    b.info.buffer = s.error_buffer;
    b.info.offset = 0;
    b.info.range  = (VkDeviceSize) TRC_VK_ERROR_U32 * sizeof(uint32_t);
    b.base_elem   = 0;
    return b;
}

void vulkan_dispatch(VulkanGraphContext& gc, const VulkanPipelineHandle& pipe, const void* pc,
                     uint32_t pc_size, const VulkanTensorBinding* bindings, uint32_t n_bindings,
                     uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    const VulkanState& s = vulkan_state();
    TRC_ASSERT(n_bindings <= VK_MAX_BINDINGS, "Vulkan 后端：绑定数 %u 超限", n_bindings);

    // 网格上限保险：所有调用方都必须钳制（kernel 内用 gl_NumWorkGroups 做 stride 循环）。
    // 超限直接报错而不是交给驱动（超限 dispatch 是未定义行为，可能静默漏算）。
    const uint32_t max_x = vulkan_max_grid_x();
    TRC_ASSERT(groups_x >= 1u && groups_x <= max_x && groups_y <= s.props.limits.maxComputeWorkGroupCount[1] &&
                   groups_z <= s.props.limits.maxComputeWorkGroupCount[2],
               "Vulkan dispatch: 网格 (%u,%u,%u) 超设备/测试上限（x<=%u, y<=%u, z<=%u）", groups_x,
               groups_y, groups_z, max_x, s.props.limits.maxComputeWorkGroupCount[1],
               s.props.limits.maxComputeWorkGroupCount[2]);

    VkDescriptorSet set = vulkan_alloc_descriptor_set();

    // R13：固定上限（VK_MAX_BINDINGS）用栈上 std::array，避免每次 dispatch 的堆分配
    std::array<VkDescriptorBufferInfo, VK_MAX_BINDINGS> infos{};
    std::array<VkWriteDescriptorSet, VK_MAX_BINDINGS>   writes{};
    for (uint32_t i = 0; i < n_bindings; ++i) {
        infos[i] = bindings[i].info;
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(s.device, n_bindings, writes.data(), 0, nullptr);

    vkCmdBindPipeline(gc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(gc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(gc.cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    vkCmdDispatch(gc.cmd, groups_x, groups_y, groups_z);
    ++vulkan_state().graph_dispatch_count;  // FIX-004 / Q28：诊断计数

    // 保守屏障：后续 dispatch 可能读取本步写入（与 ggml 的全局 shader barrier 同思路）
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(gc.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
}

VulkanState& vulkan_state() {
    static VulkanState state = [] {
        VulkanState s;
        s.available = init_state(s);
        return s;
    }();
    return state;
}

uint32_t vulkan_max_grid_x() {
    uint32_t max_groups = vulkan_state().props.limits.maxComputeWorkGroupCount[0];
    const char* env = std::getenv("TRC_VK_MAX_GROUPS");
    if (env != nullptr && *env != '\0') {
        char*      end = nullptr;
        const long v   = std::strtol(env, &end, 10);
        if (end != env && v > 0) {
            max_groups = std::min<uint32_t>(max_groups, (uint32_t) v);
        }
    }
    return std::max<uint32_t>(1u, max_groups);
}

uint32_t vulkan_max_grid_z() {
    uint32_t max_groups = vulkan_state().props.limits.maxComputeWorkGroupCount[2];
    const char* env = std::getenv("TRC_VK_MAX_GROUPS");
    if (env != nullptr && *env != '\0') {
        char*      end = nullptr;
        const long v   = std::strtol(env, &end, 10);
        if (end != env && v > 0) {
            max_groups = std::min<uint32_t>(max_groups, (uint32_t) v);
        }
    }
    return std::max<uint32_t>(1u, max_groups);
}

size_t vulkan_max_storage_range() {
    size_t      max_range = vulkan_state().max_storage_buffer_range;
    const char* env       = std::getenv("TRC_VK_MAX_STORAGE_RANGE");
    if (env != nullptr && *env != '\0') {
        char*            end = nullptr;
        const long long  v   = std::strtoll(env, &end, 10);
        if (end != env && v > 0) {
            max_range = std::min<size_t>(max_range, (size_t) v);
        }
    }
    return max_range;
}

// 编译期清单入口（src/core/trc_backends.h 声明）：不可用时返回 nullptr
Device* vulkan_device() {
    static Device* device = vulkan_state().available ? (Device*) new VulkanDevice() : nullptr;
    return device;
}

} // namespace traincpp
