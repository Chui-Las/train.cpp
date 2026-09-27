// train.cpp - 内部实现头文件（不对外暴露）
#pragma once

#include "traincpp/traincpp.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace traincpp {

// 中断执行并打印错误信息（对应 ggml_abort）
[[noreturn]] void abort_impl(const char* file, int line, const char* fmt, ...);

// 日志级别：1=DEBUG 2=INFO 3=WARN 4=ERROR
void log_impl(int level, const char* fmt, ...);

// Context 内部结构（公共头中为不透明类型）
struct Context {
    struct Arena {
        uint8_t* data = nullptr;
        size_t   size = 0;
        size_t   used = 0;
    };

    // 待填充的数据（autograd 标量常量 / 权重初始化用）：
    // no_alloc=true 时数据在后端 buffer 里，需等 buffer_alloc_ctx_tensors 完成后再写入
    struct PendingFill {
        enum Kind : int32_t {
            CONST   = 0,  // a = 常量值
            UNIFORM = 1,  // [a, b) 均匀分布（seed 派生）
            NORMAL  = 2,  // N(a, b)（seed 派生）
        };

        Tensor*  tensor = nullptr;
        int32_t  kind   = CONST;
        float    a      = 0.0f;
        float    b      = 0.0f;
        uint64_t seed   = 0;
    };

    std::vector<Arena>   arenas;
    bool                 no_alloc  = true;
    bool                 fixed_mem = false;  // 外部内存：不允许追加 arena
    std::vector<Tensor*> tensors;
    size_t               mem_used = 0;       // 累计分配字节数（含对齐填充）
    std::vector<PendingFill> pending_fills;  // 由 buffer_alloc_ctx_tensors 兑现
};

// 创建 1 元素 F32 常量张量并填写值；no_alloc 模式下延迟到 buffer 分配完成时写入
Tensor* new_scalar_const(Context* ctx, float value);

// 丢弃上下文的 Buffer 归属记录（context_free 调用；M2.1d / R3；不释放 Buffer 本身）
void backend_forget_ctx_buffers(Context* ctx);

// 持久张量集合（保守启发式；M2.2a 统计与 M2.2b 图级分配共用）：
// PARAM/LOSS/OUTPUT 标志、无输入叶子（图输入/常量/grad_acc）、
// set_rows 目标（src[2]）、acc(inplace) 宿主（src[0]）。返回按指针去重的集合。
// include_view_roots=true 时另含所有视图源（统计口径，保守）；图级分配用 false，
// 视图根改由 GraphMemoryPlan 按使用窗口精确判定（FIX-002 / Q23）。
std::unordered_set<Tensor*> context_persistent_tensors(const Context* ctx,
                                                       bool include_view_roots = true);

// 图内存计划（FIX-002 / Q23）：按"内存所有者"（视图根）统计完整 liveness。
// - owner = 沿 view_src 链的根；纯视图节点（VIEW/RESHAPE/PERMUTE/TRANSPOSE）不产生读取，
//   真实读取记在消费视图的计算节点上（两后端视图 compute 均为 no-op）
// - last_read/n_readers 覆盖"经自身或任意视图"的读取；同一节点内同根去重
// - persistent = 基持久（不含视图根）∪ 钉住根：带 OUTPUT 的视图、图外视图、
//   图输出视图（在图中但无节点引用）的根必须独占，不参与复用
struct GraphMemoryPlan {
    std::unordered_map<Tensor*, int64_t> last_read;  // owner → 最后读取节点下标
    std::unordered_map<Tensor*, int64_t> n_readers;  // owner → 读取节点数
    std::unordered_set<Tensor*>          persistent; // 基持久 ∪ 钉住
};
GraphMemoryPlan graph_memory_plan(const Context* ctx, const Graph* graph);

// 核心分配统计钩子（M2.2a；计数本体在 trc_backend_reg.cpp）：
// 记录一个经核心路径分配的 Buffer 及其请求字节数，buffer_free 时统一扣减。
// 供 buffer_alloc_ctx_tensors 与优化器状态块分配（trc_optim.cpp）调用；
// 用户直接 Device::alloc_buffer 的分配不计。
void backend_note_buffer_alloc(Buffer* buf, size_t bytes);

// 向上对齐
inline size_t align_up(size_t v, size_t align) {
    return (v + align - 1) & ~(align - 1);
}

// 对齐内存分配/释放（各后端共用）
inline uint8_t* alloc_aligned(size_t size, size_t align) {
#ifdef _MSC_VER
    return (uint8_t*) _aligned_malloc(size, align);
#else
    return (uint8_t*) std::aligned_alloc(align, align_up(size, align));
#endif
}

inline void free_aligned(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace traincpp

#define TRC_ABORT(...) ::traincpp::abort_impl(__FILE__, __LINE__, __VA_ARGS__)

#define TRC_ASSERT(x, ...)      \
    do {                        \
        if (!(x)) {             \
            TRC_ABORT(__VA_ARGS__); \
        }                       \
    } while (0)

#define TRC_LOG_DEBUG(...) ::traincpp::log_impl(1, __VA_ARGS__)
#define TRC_LOG_INFO(...)  ::traincpp::log_impl(2, __VA_ARGS__)
#define TRC_LOG_WARN(...)  ::traincpp::log_impl(3, __VA_ARGS__)
#define TRC_LOG_ERROR(...) ::traincpp::log_impl(4, __VA_ARGS__)
