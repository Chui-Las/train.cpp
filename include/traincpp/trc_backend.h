// train.cpp - 后端抽象接口
// 设计目标：核心与后端实现彻底分离；新增后端只需实现 Device/BufferType/Buffer
// 以及算子 kernel，不需要改核心代码。
#pragma once

#include "trc_graph.h"
#include "trc_tensor.h"

#include <cstddef>
#include <string>

namespace traincpp {

enum class DeviceType : int32_t {
    CPU   = 0,
    GPU   = 1,
    IGPU  = 2,
    ACCEL = 3,
};

struct DeviceProps {
    std::string name;
    std::string description;
    DeviceType  type            = DeviceType::CPU;
    size_t      memory_free     = 0;
    size_t      memory_total    = 0;
    bool        unified_memory  = false;  // 统一内存架构（核显等）
};

class Device;

// 缓冲区类型：由设备创建，负责实际内存分配
class BufferType {
public:
    virtual ~BufferType() = default;

    virtual Device* device() const = 0;
    virtual const char* name() const = 0;
    virtual size_t alignment() const = 0;
    // 分配缓冲区；失败时返回 nullptr
    virtual Buffer* alloc_buffer(size_t size) = 0;
};

// 缓冲区：一块设备内存 + 主机侧读写接口
class Buffer {
public:
    virtual ~Buffer() = default;

    virtual BufferType* buffer_type() const = 0;
    virtual size_t size() const = 0;
    virtual void*  base() = 0;
    virtual bool   is_host() const = 0;  // 主机可直接访问（统一内存/CPU）

    // 张量数据上传/下载；offset 为相对张量起点的字节偏移
    virtual void set_tensor(Tensor* t, size_t offset, const void* data, size_t size) = 0;
    virtual void get_tensor(const Tensor* t, size_t offset, void* data, size_t size) const = 0;
    virtual void clear() = 0;
};

// 设备：后端对外的统一句柄
class Device {
public:
    virtual ~Device() = default;

    virtual const DeviceProps& props() const = 0;
    virtual BufferType* default_buffer_type() = 0;
    virtual Buffer* alloc_buffer(size_t size) { return default_buffer_type()->alloc_buffer(size); }

    // 能力查询：该设备是否可以计算该算子（如实声明类型/布局约束，M2.1f）
    virtual bool supports_op(const Tensor* t) const = 0;

    // 执行计算图（异步语义由后端自行决定，一期均为同步）
    virtual bool graph_compute(Graph* graph) = 0;
};

// 整图 pre-flight：返回第一个当前设备不支持的节点（nullptr = 全部支持）。
// 建议在 graph_compute 前调用：不同后端对类型/布局的支持范围不同；不支持时会中止。
const Tensor* graph_first_unsupported(const Graph* graph, const Device* dev);

// ------------------------------------------------------------------
// 全局设备注册表（对应 ggml_backend_reg / ggml_backend_dev）
// ------------------------------------------------------------------
size_t  device_count();
Device* device_get(size_t index);
// 按类型返回第一个匹配设备；找不到返回 nullptr
Device* device_by_type(DeviceType type);
// CPU 设备快捷入口（未启用 CPU 后端时返回 nullptr）
Device* device_cpu();

// 后端注册（供后端实现调用；注册顺序即枚举顺序）
void backend_register_device(Device* device);

// ------------------------------------------------------------------
// 高层便捷函数（对应 ggml_backend_* 的部分 API）
// ------------------------------------------------------------------
// 为上下文中所有张量在指定 buffer 类型上分配内存，并设置 t->buffer / t->data
// （整 ctx 一次性分配；语义与一期一致，不作生命周期复用）
Buffer* buffer_alloc_ctx_tensors(Context* ctx, BufferType* buft);

// 图级生命周期分配（M2.2b / R10；FIX-002 / Q23 起视图根与多后继参与）：
// 按 graph->nodes 执行序计算每个内存所有者（视图根）的最后读取，生命周期不重叠的
// 中间量复用同一段缓冲（free_at < produce），显著降低峰值。
//   - 视图算子是 no-op：真实读取记在消费视图的计算节点上；多后继（如前向+反向读者）
//     也参与复用；同一节点内重复引用按所有者去重
//   - 独占（不参与复用）：PARAM/LOSS/OUTPUT、无输入叶子、grad_acc、
//     set_rows/acc(inplace) 宿主、无后继张量（图输出）；以及三类视图的根——
//     带 TENSOR_FLAG_OUTPUT 的视图、图外视图、图输出视图（在图中但无节点引用）
//   - 要求 Graph 覆盖 ctx 全部待分配的非视图张量（前向+反向训练图天然满足），否则中止
//   - 中间量存储是复用的：compute 之后只有参数/损失/图输出，以及显式标
//     TENSOR_FLAG_OUTPUT 的张量/视图可安全读取；其余中间量与视图可能已被覆盖
//   - 分配成功后同样登记 buffer_free 归属（R3 语义）；同一 ctx 只能分配一次
//   - 反复建图/变长样本场景：buffer_free 后用 context_release_graph_tensors 回收
//     图的中间量（见 trc_context.h；M2.2c / R14）
Buffer* buffer_alloc_graph_tensors(Graph* graph, BufferType* buft);

void    buffer_free(Buffer* buf);

// ---------------- 进程级核心分配统计（M2.2a / R10，诊断用）----------------
// 统计范围仅核心路径：buffer_alloc_ctx_tensors 与优化器设备状态块（alloc_states_block）；
// buffer_free 统一扣减。主机回退块（alloc_aligned）与用户直接调用 Device::alloc_buffer
// 不计入（库内核心路径不经过后者）。
struct AllocStats {
    size_t current_bytes = 0;  // 当前存活字节数（自上次 reset 起）
    size_t peak_bytes    = 0;  // 峰值字节数（自上次 reset 起）
    size_t n_allocations = 0;  // 分配次数（自上次 reset 起）
};

AllocStats backend_alloc_stats();
// 重置统计基线：清零计数并停止跟踪已存在的 buffer
// （重置之前分配的 buffer 之后释放不会再扣减，避免下溢）
void backend_alloc_stats_reset();

void tensor_set(Tensor* t, const void* data, size_t offset, size_t size);
void tensor_get(const Tensor* t, void* data, size_t offset, size_t size);

} // namespace traincpp
