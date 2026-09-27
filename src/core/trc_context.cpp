// train.cpp - 上下文（多 arena 元数据分配器）
#include "traincpp/trc_context.h"

#include "core/trc_impl.h"
#include "traincpp/trc_graph.h"

#include <unordered_map>
#include <unordered_set>

namespace traincpp {

namespace {

// 视图的内存所有者（沿 view_src 链的根；FIX-002 / Q23）
Tensor* view_root_of(Tensor* t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

// 纯视图算子：compute 为 no-op，不读取源数据（真实读取在消费视图的节点上）
bool is_pure_view_op(Op op) {
    return op == OP_VIEW || op == OP_RESHAPE || op == OP_PERMUTE || op == OP_TRANSPOSE;
}

} // namespace

Context* context_new(size_t mem_size) {
    ContextParams params;
    params.mem_size = mem_size;
    params.no_alloc = true;
    return context_new(params);
}

Context* context_new(const ContextParams& params) {
    TRC_ASSERT(params.mem_size > 0 || params.mem_buffer != nullptr,
               "context_new: mem_size 必须大于 0（或提供 mem_buffer）");

    Context* ctx = new Context();
    ctx->no_alloc = params.no_alloc;

    Context::Arena arena;
    if (params.mem_buffer != nullptr) {
        arena.data = (uint8_t*) params.mem_buffer;
        arena.size = params.mem_size;
        ctx->fixed_mem = true;
    } else {
        arena.size = params.mem_size;
        arena.data = alloc_aligned(arena.size, MEM_ALIGN);
    }
    TRC_ASSERT(arena.data != nullptr,
               "context_new: 内存申请失败：需要 %zu 字节（%.2f MiB）；请减小 mem_size 或检查可用物理内存",
               arena.size, (double) arena.size / (1024.0 * 1024.0));
    arena.used = 0;
    ctx->arenas.push_back(arena);
    return ctx;
}

void context_free(Context* ctx) {
    if (ctx == nullptr) {
        return;
    }
    // 丢弃 Buffer 归属记录：之后 buffer_free 不再触碰已销毁上下文的张量（M2.1d / R3）
    backend_forget_ctx_buffers(ctx);
    for (auto& arena : ctx->arenas) {
        if (!ctx->fixed_mem) {
            free_aligned(arena.data);
        }
    }
    delete ctx;
}

size_t context_mem_size(const Context* ctx) {
    size_t total = 0;
    for (const auto& arena : ctx->arenas) {
        total += arena.size;
    }
    return total;
}

size_t context_mem_used(const Context* ctx) {
    return ctx->mem_used;
}

size_t context_mem_available(const Context* ctx) {
    return context_mem_size(ctx) - ctx->mem_used;
}

bool context_no_alloc(const Context* ctx) {
    return ctx->no_alloc;
}

void context_set_no_alloc(Context* ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

void* context_alloc(Context* ctx, size_t size, size_t align) {
    TRC_ASSERT(size > 0, "context_alloc: size 必须大于 0");
    TRC_ASSERT(align > 0 && align <= MEM_ALIGN, "context_alloc: 对齐 %zu 超出支持范围（最大 %zu）", align,
               MEM_ALIGN);

    // 1) 尝试最后一个 arena
    if (!ctx->arenas.empty()) {
        Context::Arena& arena = ctx->arenas.back();
        const size_t before = arena.used;
        const size_t offset = align_up(before, align);
        if (offset + size <= arena.size) {
            arena.used = offset + size;
            ctx->mem_used += arena.used - before;  // 含对齐填充
            return arena.data + offset;
        }
    }

    // 2) 追加新 arena
    TRC_ASSERT(!ctx->fixed_mem,
               "context_alloc: 外部内存已用尽：需要 %zu 字节（%.2f MiB），上限 %zu 字节（%.2f MiB，"
               "mem_buffer 合计），已用 %zu 字节；请增大 mem_size 或减少张量",
               size, (double) size / (1024.0 * 1024.0), context_mem_size(ctx),
               (double) context_mem_size(ctx) / (1024.0 * 1024.0), context_mem_used(ctx));

    const size_t last_size = ctx->arenas.empty() ? 0 : ctx->arenas.back().size;
    Context::Arena arena;
    arena.size = std::max(align_up(size, align) + align, std::max(last_size * 2, (size_t) 4096));
    arena.data = alloc_aligned(arena.size, MEM_ALIGN);
    TRC_ASSERT(arena.data != nullptr,
               "context_alloc: arena 申请失败：需要 %zu 字节（%.2f MiB，含按上次 arena 翻倍）；"
               "请减小模型/批量或检查可用物理内存",
               arena.size, (double) arena.size / (1024.0 * 1024.0));
    arena.used = align_up(size, align);

    void* ptr = arena.data;
    ctx->mem_used += arena.used;
    ctx->arenas.push_back(arena);
    return ptr;
}

// 持久集合判定（保守启发式；M2.2a 统计与 M2.2b 图级分配共用）：
// PARAM/LOSS/OUTPUT 标志、无输入叶子（图输入/常量/grad_acc）、视图源（可选）、
// set_rows 目标与 acc(inplace) 宿主张量（结果与它们共享内存）。
std::unordered_set<Tensor*> context_persistent_tensors(const Context* ctx, bool include_view_roots) {
    TRC_ASSERT(ctx != nullptr, "context_persistent_tensors: ctx 为空");

    const size_t              n = ctx->tensors.size();
    std::unordered_set<Tensor*> persistent;
    persistent.reserve(n / 2 + 8);

    for (size_t i = 0; i < n; ++i) {
        Tensor* t = ctx->tensors[i];
        if (t->view_src != nullptr) {
            if (include_view_roots) {
                persistent.insert(view_root_of(t));
            }
            continue;
        }
        if ((t->flags & (TENSOR_FLAG_PARAM | TENSOR_FLAG_LOSS | TENSOR_FLAG_OUTPUT)) != 0) {
            persistent.insert(t);
        }
        if (t->nsrc == 0) {
            persistent.insert(t);
        }
        if (t->grad_acc != nullptr) {
            persistent.insert(t->grad_acc);
        }
        if (t->op == OP_SET_ROWS && t->src[2] != nullptr) {
            persistent.insert(t->src[2]);
        }
        if (t->op == OP_ACC && t->src[0] != nullptr && t->op_params[4] != 0) {
            persistent.insert(t->src[0]);
        }
    }
    return persistent;
}

// 图内存计划（FIX-002 / Q23）：根感知 liveness + 钉住，供图级分配与回收共用。
GraphMemoryPlan graph_memory_plan(const Context* ctx, const Graph* graph) {
    TRC_ASSERT(ctx != nullptr && graph != nullptr, "graph_memory_plan: 参数为空");

    GraphMemoryPlan plan;
    const size_t    n_nodes = graph->nodes.size();
    plan.n_readers.reserve(n_nodes * 2);
    plan.last_read.reserve(n_nodes * 2);

    // 1) 图内张量集合（钉住判定需要）
    std::unordered_set<Tensor*> node_set;
    node_set.reserve(n_nodes * 2);
    for (Tensor* node : graph->nodes) {
        if (node != nullptr) {
            node_set.insert(node);
        }
    }

    // 2) 读取统计：跳过纯视图节点；src 归到内存所有者（视图根）
    std::unordered_set<Tensor*> src_used;  // 被图内节点当 src 的张量
    src_used.reserve(n_nodes * 2);
    for (size_t i = 0; i < n_nodes; ++i) {
        Tensor* node = graph->nodes[i];
        if (node == nullptr || (node->view_src != nullptr && is_pure_view_op(node->op))) {
            continue;
        }
        Tensor* seen[MAX_SRC];
        int     n_seen = 0;
        for (int32_t j = 0; j < node->nsrc; ++j) {
            Tensor* s = node->src[j];
            if (s == nullptr) {
                continue;
            }
            src_used.insert(s);
            Tensor* owner = view_root_of(s);
            bool    duplicate = false;
            for (int k = 0; k < n_seen; ++k) {
                if (seen[k] == owner) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
            if (n_seen < MAX_SRC) {
                seen[n_seen++] = owner;
            }
            const auto it = plan.n_readers.find(owner);
            if (it == plan.n_readers.end()) {
                plan.n_readers[owner] = 1;
            } else {
                it->second += 1;
            }
            plan.last_read[owner] = (int64_t) i;  // i 递增，最后写入即最大值
        }
    }

    // 3) 基持久（不含视图根；视图根由 liveness + 钉住规则决定）
    plan.persistent = context_persistent_tensors(ctx, /*include_view_roots=*/false);

    // 4) 钉住：外部/输出视图的根必须独占（FIX-002 / Q23）
    for (size_t i = 0; i < ctx->tensors.size(); ++i) {
        Tensor* t = ctx->tensors[i];
        if (t->view_src == nullptr) {
            continue;
        }
        if ((t->flags & TENSOR_FLAG_OUTPUT) != 0) {
            plan.persistent.insert(view_root_of(t));  // compute 后仍需读取的视图
        } else if (node_set.count(t) == 0) {
            plan.persistent.insert(view_root_of(t));  // 图外视图（可能被主机后读）
        } else if (src_used.count(t) == 0) {
            plan.persistent.insert(view_root_of(t));  // 视图即图输出
        }
    }
    return plan;
}

ContextMemoryStats context_memory_stats(const Context* ctx) {
    TRC_ASSERT(ctx != nullptr, "context_memory_stats: ctx 为空");

    ContextMemoryStats stats;
    const size_t       n = ctx->tensors.size();
    const std::unordered_set<Tensor*> persistent = context_persistent_tensors(ctx);

    for (size_t i = 0; i < n; ++i) {
        const Tensor* t = ctx->tensors[i];
        if (t->view_src != nullptr) {
            ++stats.view_count;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        Tensor* t = ctx->tensors[i];
        if (t->view_src != nullptr) {
            continue;  // 视图不拥有内存
        }
        const size_t bytes = tensor_nbytes(t);
        stats.tensor_bytes += bytes;
        if (persistent.count(t) != 0) {
            stats.persistent_bytes += bytes;
        }
    }
    return stats;
}

size_t context_release_graph_tensors(Context* ctx, Graph* graph) {
    TRC_ASSERT(ctx != nullptr && graph != nullptr, "context_release_graph_tensors: 参数为空");
    TRC_ASSERT(graph->ctx == ctx, "context_release_graph_tensors: graph 与 ctx 不一致");

    // 1) 图内存计划（FIX-002 / Q23）：与 buffer_alloc_graph_tensors 同源——
    //    根感知 liveness + 钉住（OUTPUT/图外/图输出视图的根独占）
    const GraphMemoryPlan plan = graph_memory_plan(ctx, graph);

    // 2) 候选：非视图、非持久/钉住、被图内节点读取（无读者 = 图输出/死张量，保留），
    //    且仍在 registry。Phase 2 起含多后继中间量（复用窗口由 last_read 决定）。
    const std::unordered_set<Tensor*> in_registry(ctx->tensors.begin(), ctx->tensors.end());
    std::unordered_set<Tensor*>       release;
    release.reserve(graph->nodes.size() / 2 + 8);
    for (Tensor* node : graph->nodes) {
        if (node == nullptr || node->view_src != nullptr) {
            continue;
        }
        if (plan.persistent.count(node) != 0) {
            continue;
        }
        const auto nr = plan.n_readers.find(node);
        if (nr == plan.n_readers.end()) {
            continue;
        }
        if (in_registry.count(node) == 0) {
            continue;  // 已被回收（重复调用安全）
        }
        release.insert(node);
    }
    if (release.empty()) {
        return 0;
    }

    // 3) 清数据指针（下次分配不再包含这些张量）
    for (Tensor* t : release) {
        t->buffer = nullptr;
        t->data   = nullptr;
    }

    // 4) 从 registry 移除（其余张量相对顺序不变：视图源仍先于视图）
    std::vector<Tensor*>& tensors = ctx->tensors;
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [&release](Tensor* t) { return release.count(t) != 0; }),
                  tensors.end());

    // 5) 丢弃指向被释放张量的延迟填充（避免悬垂）
    std::vector<Context::PendingFill>& fills = ctx->pending_fills;
    fills.erase(std::remove_if(fills.begin(), fills.end(),
                               [&release](const Context::PendingFill& f) {
                                   return release.count(f.tensor) != 0;
                               }),
                fills.end());

    return release.size();
}

int context_tensor_count(const Context* ctx) {
    return (int) ctx->tensors.size();
}

Tensor* context_tensor(const Context* ctx, int index) {
    TRC_ASSERT(index >= 0 && index < (int) ctx->tensors.size(), "context_tensor: 下标越界 %d", index);
    return ctx->tensors[index];
}

} // namespace traincpp
