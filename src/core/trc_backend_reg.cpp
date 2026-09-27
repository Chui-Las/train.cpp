// train.cpp - 设备注册表与 buffer 高层 API
#include "traincpp/trc_backend.h"

#include "core/trc_backends.h"
#include "core/trc_impl.h"
#include "traincpp/trc_context.h"
#include "traincpp/trc_rng.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace traincpp {

namespace {

// Buffer → Context 归属表（M2.1d / R3）：
// 分配时登记，buffer_free 据此把张量的 data/buffer 置空，杜绝悬垂指针。
// 不放在公共 Buffer 接口里：后端无需感知，映射完全由核心维护。
struct BufferOwner {
    Buffer*  buffer = nullptr;
    Context* ctx    = nullptr;
};

std::vector<BufferOwner>& buffer_owners() {
    static std::vector<BufferOwner> owners;
    return owners;
}

std::vector<BufferOwner>::iterator find_buffer_owner(Buffer* buf) {
    std::vector<BufferOwner>& owners = buffer_owners();
    return std::find_if(owners.begin(), owners.end(),
                        [buf](const BufferOwner& o) { return o.buffer == buf; });
}

// 核心分配统计（M2.2a / R10）：记录已计入的 Buffer 以便释放时精确扣减。
// 只统计核心路径（ctx 张量 buffer + 优化器设备状态块），主机回退块不计。
struct AllocCounters {
    size_t current = 0;
    size_t peak    = 0;
    size_t n       = 0;
    std::vector<std::pair<Buffer*, size_t>> tracked;
};

AllocCounters& alloc_counters() {
    static AllocCounters counters;
    return counters;
}

// 视图的根源张量（沿 view_src 链）
Tensor* view_root(Tensor* t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

// 函数内静态变量：避免静态初始化顺序问题；首次使用时注册编译期已知后端
std::vector<Device*>& device_registry() {
    static std::vector<Device*> devices = [] {
        std::vector<Device*> v;
#if TRC_USE_CPU
        v.push_back(cpu_device());
#endif
#if TRC_USE_VULKAN
        // Vulkan 初始化失败（无设备/SDK 运行库缺失）时返回 nullptr：视为后端不可用
        if (Device* vulkan = vulkan_device()) {
            v.push_back(vulkan);
        }
#endif
        return v;
    }();
    return devices;
}

} // namespace

void backend_register_device(Device* device) {
    TRC_ASSERT(device != nullptr, "backend_register_device: device 为空");
    device_registry().push_back(device);
}

size_t device_count() {
    return device_registry().size();
}

Device* device_get(size_t index) {
    if (index >= device_registry().size()) {
        return nullptr;
    }
    return device_registry()[index];
}

Device* device_by_type(DeviceType type) {
    for (Device* device : device_registry()) {
        if (device->props().type == type) {
            return device;
        }
    }
    return nullptr;
}

Device* device_cpu() {
    return device_by_type(DeviceType::CPU);
}

const Tensor* graph_first_unsupported(const Graph* graph, const Device* dev) {
    TRC_ASSERT(graph != nullptr && dev != nullptr, "graph_first_unsupported: 参数为空");
    for (int64_t i = 0; i < graph_n_nodes(graph); ++i) {
        const Tensor* node = graph_node(graph, i);
        if (node != nullptr && !dev->supports_op(node)) {
            return node;
        }
    }
    return nullptr;
}

// 记录核心路径的一次 Buffer 分配（内部钩子；供优化器状态块分配调用）
void backend_note_buffer_alloc(Buffer* buf, size_t bytes) {
    if (buf == nullptr || bytes == 0) {
        return;
    }
    AllocCounters& c = alloc_counters();
    c.current += bytes;
    if (c.current > c.peak) {
        c.peak = c.current;
    }
    c.n += 1;
    c.tracked.push_back({buf, bytes});
}

AllocStats backend_alloc_stats() {
    const AllocCounters& c = alloc_counters();
    AllocStats stats;
    stats.current_bytes = c.current;
    stats.peak_bytes    = c.peak;
    stats.n_allocations = c.n;
    return stats;
}

void backend_alloc_stats_reset() {
    AllocCounters& c = alloc_counters();
    c.current = 0;
    c.peak    = 0;
    c.n       = 0;
    c.tracked.clear();
}

namespace {

// 分配失败的统一报错（M2.2a-2：需要 X / 可用或上限 Y）
[[noreturn]] void abort_buffer_alloc(const char* api, size_t total, BufferType* buft) {
    const DeviceProps& props    = buft->device()->props();
    const double       need_mib = (double) total / (1024.0 * 1024.0);
    if (props.memory_total != 0 || props.memory_free != 0) {
        TRC_ABORT("%s: buffer 分配失败：需要 %zu 字节（%.2f MiB），"
                  "设备 '%s' 可用 %zu 字节（%.2f MiB）/ 总 %zu 字节（%.2f MiB）；"
                  "请减小 batch/segment、分块或换设备",
                  api, total, need_mib, props.name.c_str(), props.memory_free,
                  (double) props.memory_free / (1024.0 * 1024.0), props.memory_total,
                  (double) props.memory_total / (1024.0 * 1024.0));
    }
    TRC_ABORT("%s: buffer 分配失败：需要 %zu 字节（%.2f MiB）；"
              "请减小 batch/segment 或分块（设备 '%s' 未报告内存量）",
              api, total, need_mib, props.name.c_str());
}

// 视图数据解析（旧/新分配 API 共用；视图按创建顺序，其源一定先于视图解析）
void resolve_view_data(Context* ctx, int n_tensors) {
    for (int i = 0; i < n_tensors; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t->view_src == nullptr) {
            continue;
        }
        TRC_ASSERT(t->view_src->data != nullptr, "视图源张量 %s 的数据未解析", t->view_src->name);
        t->data = (uint8_t*) t->view_src->data + t->view_offs;
    }
}

// 兑现延迟填充（autograd 标量常量、权重初始化等；旧/新分配 API 共用）
void apply_pending_fills(Context* ctx) {
    for (const Context::PendingFill& fill : ctx->pending_fills) {
        TRC_ASSERT(fill.tensor->data != nullptr, "延迟填充 %s 的数据未分配", fill.tensor->name);
        switch (fill.kind) {
            case Context::PendingFill::CONST: {
                // 注意：常量填充可能是多元素张量（如权重初始化为 1/0）
                // R13：直接写主机数据，不再分配临时 vector
                Tensor* t = fill.tensor;
                TRC_ASSERT(tensor_is_contiguous(t), "延迟常量填充 %s 必须连续", t->name);
                const size_t n = (size_t) tensor_nelements(t);
                float*       p = (float*) t->data;
                std::fill(p, p + n, fill.a);
            } break;
            case Context::PendingFill::UNIFORM: {
                Rng rng;
                rng.state = fill.seed;
                rng_fill_uniform(ctx, &rng, fill.tensor, fill.a, fill.b);
            } break;
            case Context::PendingFill::NORMAL: {
                Rng rng;
                rng.state = fill.seed;
                rng_fill_normal(ctx, &rng, fill.tensor, fill.a, fill.b);
            } break;
            default:
                TRC_ABORT("未知的延迟填充类型 %d", (int) fill.kind);
        }
    }
    ctx->pending_fills.clear();
}

} // namespace

Buffer* buffer_alloc_ctx_tensors(Context* ctx, BufferType* buft) {
    TRC_ASSERT(ctx != nullptr && buft != nullptr, "buffer_alloc_ctx_tensors: 参数为空");

    const size_t align = buft->alignment();
    const int n_tensors = context_tensor_count(ctx);

    // 第一遍：计算总大小（跳过视图与已分配数据的张量：
    // 优化器状态等可能已通过自己的 buffer / 主机内存分配，见 optim_step）
    size_t total = 0;
    for (int i = 0; i < n_tensors; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t->view_src != nullptr || t->data != nullptr) {
            continue;
        }
        total = align_up(total, align) + tensor_nbytes(t);
    }

    Buffer* buf = buft->alloc_buffer(total);
    if (buf == nullptr) {
        // M2.2a-2：失败信息统一为"需要 X / 上限或可用 Y"
        const DeviceProps& props    = buft->device()->props();
        const double       need_mib = (double) total / (1024.0 * 1024.0);
        if (props.memory_total != 0 || props.memory_free != 0) {
            TRC_ABORT("buffer_alloc_ctx_tensors: buffer 分配失败：需要 %zu 字节（%.2f MiB），"
                      "设备 '%s' 可用 %zu 字节（%.2f MiB）/ 总 %zu 字节（%.2f MiB）；"
                      "请减小 batch/segment、分块或换设备",
                      total, need_mib, props.name.c_str(), props.memory_free,
                      (double) props.memory_free / (1024.0 * 1024.0), props.memory_total,
                      (double) props.memory_total / (1024.0 * 1024.0));
        }
        TRC_ABORT("buffer_alloc_ctx_tensors: buffer 分配失败：需要 %zu 字节（%.2f MiB）；"
                  "请减小 batch/segment 或分块（设备 '%s' 未报告内存量）",
                  total, need_mib, props.name.c_str());
    }
    buffer_owners().push_back({buf, ctx});
    backend_note_buffer_alloc(buf, buf->size());  // M2.2a：核心分配统计（按实际缓冲字节数）

    // 第二遍：分配偏移
    size_t offset = 0;
    for (int i = 0; i < n_tensors; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t->view_src != nullptr || t->data != nullptr) {
            continue;
        }
        offset = align_up(offset, align);
        t->buffer = buf;
        t->data = (uint8_t*) buf->base() + offset;
        offset += tensor_nbytes(t);
    }

    // 第三遍：解析视图数据指针
    resolve_view_data(ctx, n_tensors);

    // 第四遍：兑现延迟填充
    apply_pending_fills(ctx);

    return buf;
}

Buffer* buffer_alloc_graph_tensors(Graph* graph, BufferType* buft) {
    TRC_ASSERT(graph != nullptr && buft != nullptr, "buffer_alloc_graph_tensors: 参数为空");
    Context* ctx = graph->ctx;
    TRC_ASSERT(ctx != nullptr, "buffer_alloc_graph_tensors: graph->ctx 为空");

    const size_t  align     = buft->alignment();
    const int     n_tensors = context_tensor_count(ctx);
    const int64_t n_nodes   = graph_n_nodes(graph);
    TRC_ASSERT(n_nodes > 0, "buffer_alloc_graph_tensors: 图为空（先 graph_build_forward_expand）");

    std::unordered_map<Tensor*, int64_t> node_index;  // 节点 → 执行序下标
    node_index.reserve((size_t) n_nodes * 2);
    for (int64_t i = 0; i < n_nodes; ++i) {
        node_index[graph->nodes[(size_t) i]] = i;
    }

    // 1) 覆盖校验：所有待分配的非视图张量必须在图内，否则无法计算生命周期
    for (int i = 0; i < n_tensors; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t->view_src != nullptr || t->data != nullptr) {
            continue;
        }
        TRC_ASSERT(node_index.count(t) != 0,
                   "buffer_alloc_graph_tensors: 图未覆盖张量 %s（op=%s）；请先把该张量加入图，"
                   "或改用 buffer_alloc_ctx_tensors（整 ctx 一次性分配）",
                   t->name[0] != '\0' ? t->name : "<未命名>", op_name(t->op));
    }

    // 2) 图内存计划（FIX-002 / Q23）：根感知 liveness + 钉住。
    //    - owner = 视图根；纯视图节点是 no-op，真实读取记在消费视图的计算节点上
    //    - persistent = 基持久（无视图根）∪ 钉住（OUTPUT 视图/图外视图/图输出视图的根）
    const GraphMemoryPlan plan = graph_memory_plan(ctx, graph);

    // 3) 分块分配：
    //    Phase A 独占：持久/钉住张量、无后继中间量（图输出/死张量，保守保留）
    //    Phase B 复用：有后继且非持久/非钉住的中间量（FIX-002 起含视图根与多后继），
    //                  按生产节点序 + free-list 复用（free_at < produce；free_at=最后读取）
    struct Candidate {
        Tensor* t       = nullptr;
        size_t  bytes   = 0;
        int64_t produce = 0;  // 生产节点下标
        int64_t free_at = 0;  // 最后读取节点下标（该节点执行后可复用）
    };
    struct Placement {
        Tensor* t      = nullptr;
        size_t  offset = 0;
    };
    std::vector<Candidate> candidates;
    std::vector<Placement> placements;

    const auto aligned_bytes = [align](size_t n) { return align_up(n, align); };
    size_t     end = 0;  // 高水位（峰值 = 最终 end - 已复用部分）
    const auto alloc_from_end = [&end, align](size_t bytes) {
        end = align_up(end, align) + bytes;
        return end - bytes;
    };

    for (int i = 0; i < n_tensors; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t->view_src != nullptr || t->data != nullptr) {
            continue;
        }
        const auto nr          = plan.n_readers.find(t);
        const bool has_readers = nr != plan.n_readers.end();
        if (has_readers && plan.persistent.count(t) == 0) {
            const auto ni = node_index.find(t);
            TRC_ASSERT(ni != node_index.end(), "buffer_alloc_graph_tensors: 张量 %s 不在图内", t->name);
            Candidate c;
            c.t       = t;
            c.bytes   = aligned_bytes(tensor_nbytes(t));
            c.produce = ni->second;
            c.free_at = plan.last_read.at(t);
            TRC_ASSERT(c.produce < c.free_at,
                       "buffer_alloc_graph_tensors: 张量 %s 的生命周期非法（生产 %lld >= 最后使用 "
                       "%lld）；图不是拓扑序？",
                       t->name, (long long) c.produce, (long long) c.free_at);
            candidates.push_back(c);
            continue;
        }
        placements.push_back({t, alloc_from_end(aligned_bytes(tensor_nbytes(t)))});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.produce < b.produce; });

    std::vector<std::pair<size_t, size_t>> free_blocks;  // {offset, size}（均为 align 倍数）
    struct InUse {
        size_t  offset  = 0;
        size_t  size    = 0;
        int64_t free_at = 0;
    };
    std::vector<InUse> in_use;

    for (const Candidate& c : candidates) {
        // 释放本候选生产前已经"用完"的块（free_at < produce 才可复用；同一节点读写不可共享）
        for (auto it = in_use.begin(); it != in_use.end();) {
            if (it->free_at < c.produce) {
                free_blocks.push_back({it->offset, it->size});
                it = in_use.erase(it);
            } else {
                ++it;
            }
        }
        // 合并相邻空闲块（按偏移排序）
        std::sort(free_blocks.begin(), free_blocks.end());
        std::vector<std::pair<size_t, size_t>> merged;
        merged.reserve(free_blocks.size());
        for (const auto& b : free_blocks) {
            if (!merged.empty() && merged.back().first + merged.back().second == b.first) {
                merged.back().second += b.second;
            } else {
                merged.push_back(b);
            }
        }
        free_blocks.swap(merged);

        // first-fit
        size_t off = SIZE_MAX;
        for (auto& b : free_blocks) {
            if (b.second >= c.bytes) {
                off     = b.first;
                b.first += c.bytes;
                b.second -= c.bytes;
                break;
            }
        }
        if (off == SIZE_MAX) {
            off = alloc_from_end(c.bytes);
        }
        placements.push_back({c.t, off});
        in_use.push_back({off, c.bytes, c.free_at});
    }

    const size_t total = end;

    // 5) 分配并绑定
    Buffer* buf = buft->alloc_buffer(total);
    if (buf == nullptr) {
        abort_buffer_alloc("buffer_alloc_graph_tensors", total, buft);
    }
    buffer_owners().push_back({buf, ctx});
    backend_note_buffer_alloc(buf, buf->size());  // M2.2a：核心分配统计

    for (const Placement& p : placements) {
        p.t->buffer = buf;
        p.t->data   = (uint8_t*) buf->base() + p.offset;
    }
    resolve_view_data(ctx, n_tensors);
    apply_pending_fills(ctx);

    return buf;
}

// 丢弃上下文的所有 Buffer 归属记录（context_free 调用；不释放 Buffer 本身）
void backend_forget_ctx_buffers(Context* ctx) {
    std::vector<BufferOwner>& owners = buffer_owners();
    owners.erase(std::remove_if(owners.begin(), owners.end(),
                                [ctx](const BufferOwner& o) { return o.ctx == ctx; }),
                 owners.end());
}

void buffer_free(Buffer* buf) {
    if (buf == nullptr) {
        return;
    }

    // 核心分配统计扣减（M2.2a）：只对已计入的 Buffer 扣减，避免重置基线后下溢
    {
        AllocCounters& c = alloc_counters();
        const auto it = std::find_if(c.tracked.begin(), c.tracked.end(),
                                     [buf](const std::pair<Buffer*, size_t>& p) { return p.first == buf; });
        if (it != c.tracked.end()) {
            c.current -= it->second;
            c.tracked.erase(it);
        }
    }

    std::vector<BufferOwner>& owners = buffer_owners();
    const auto it = find_buffer_owner(buf);
    if (it != owners.end()) {
        Context* ctx = it->ctx;
        owners.erase(it);

        const int n_tensors = context_tensor_count(ctx);
        // 先失效视图（根的 buffer 被释放；根在后一遍才置空）
        for (int i = 0; i < n_tensors; ++i) {
            Tensor* t = context_tensor(ctx, i);
            if (t->view_src != nullptr && view_root(t)->buffer == buf) {
                t->data = nullptr;
            }
        }
        // 再失效直接归属该 buffer 的张量
        for (int i = 0; i < n_tensors; ++i) {
            Tensor* t = context_tensor(ctx, i);
            if (t->buffer == buf) {
                t->buffer = nullptr;
                t->data   = nullptr;
            }
        }
    }
    delete buf;
}

void tensor_set(Tensor* t, const void* data, size_t offset, size_t size) {
    TRC_ASSERT(t != nullptr, "tensor_set: 张量为空");
    TRC_ASSERT(offset + size <= tensor_nbytes(t), "tensor_set: 越界 (offset=%zu size=%zu nbytes=%zu)",
               offset, size, tensor_nbytes(t));
    if (t->buffer != nullptr) {
        t->buffer->set_tensor(t, offset, data, size);
        return;
    }
    // 视图/主机内存直写
    TRC_ASSERT(t->data != nullptr, "tensor_set: 张量既未绑定 buffer 也无数据");
    std::memcpy((uint8_t*) t->data + offset, data, size);
}

void tensor_get(const Tensor* t, void* data, size_t offset, size_t size) {
    TRC_ASSERT(t != nullptr, "tensor_get: 张量为空");
    TRC_ASSERT(offset + size <= tensor_nbytes(t), "tensor_get: 越界 (offset=%zu size=%zu nbytes=%zu)",
               offset, size, tensor_nbytes(t));
    if (t->buffer != nullptr) {
        t->buffer->get_tensor(t, offset, data, size);
        return;
    }
    TRC_ASSERT(t->data != nullptr, "tensor_get: 张量既未绑定 buffer 也无数据");
    std::memcpy(data, (const uint8_t*) t->data + offset, size);
}

} // namespace traincpp
