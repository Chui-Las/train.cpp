// train.cpp - 计算上下文（元数据 arena + 可选数据分配）
// 与 ggml 的 ggml_context 对应：张量元数据从 arena 分配，永不单独释放。
#pragma once

#include "trc_tensor.h"

#include <cstddef>

namespace traincpp {

struct Context;

struct ContextParams {
    size_t mem_size   = 0;         // 首个 arena 的字节数
    void*  mem_buffer = nullptr;   // 外部提供的内存；nullptr 表示内部申请
    bool   no_alloc   = true;      // true：张量数据不分配，由后端 buffer 接管
};

// 创建上下文；mem_size 是首个 arena 的大小，后续不够时自动追加 arena（外部 buffer 除外）
Context* context_new(size_t mem_size);
Context* context_new(const ContextParams& params);
void context_free(Context* ctx);

size_t context_mem_size(const Context* ctx);       // 已申请的 arena 总字节数
size_t context_mem_used(const Context* ctx);       // 已使用的字节数
size_t context_mem_available(const Context* ctx);  // 剩余可用字节数
bool   context_no_alloc(const Context* ctx);
void   context_set_no_alloc(Context* ctx, bool no_alloc);

// 从 arena 分配对齐内存。size 不足时会自动追加 arena；
// 若上下文使用外部 mem_buffer，则耗尽时直接报错。
void* context_alloc(Context* ctx, size_t size, size_t align = MEM_ALIGN);

// 上下文中按创建顺序记录的全部张量
int     context_tensor_count(const Context* ctx);
Tensor* context_tensor(const Context* ctx, int index);

// ---------------- 图张量回收（M2.2c / R14）----------------
// 释放 Graph 中"可回收"的张量并从 ctx 的 registry 移除，用于长训练/变长样本的
// "反复建图 → 释放"循环（否则 registry 与数据只增不减）：
//   - 回收对象：graph 节点中**非持久/非钉住、非视图、且被图内节点读取**的中间量
//     （FIX-002 / Q23 起含视图根与多后继）；独占集合与 buffer_alloc_graph_tensors
//     同源（PARAM/LOSS/OUTPUT、无输入叶子、grad_acc、set_rows/acc(inplace) 宿主、
//     带 TENSOR_FLAG_OUTPUT 的视图/图外视图/图输出视图的根），无后继的图输出也保留
//   - 清除这些张量的 data/buffer；指向它们的延迟填充条目一并丢弃
//   - 张量元数据（arena 内约 300B/张量，含视图）不回收，反复建图仍会缓慢增长
//   - 调用方必须保证之后不再使用这些张量；参数/损失/grad_acc 等持久张量不受影响
// 返回回收的张量数量。建议先 buffer_free 再调用（数据字节随 Buffer 释放）。
size_t context_release_graph_tensors(Context* ctx, Graph* graph);

// ---------------- 内存核算（M2.2a / R10）----------------
// tensor_bytes     = 全部非视图张量的 tensor_nbytes 之和
//                    （= 分配器不跳过任何张量时应分配的总量）
// persistent_bytes = 持久集合之和（PARAM/LOSS、grad_acc、无输入叶子、视图源、
//                    set_rows/acc(inplace) 的宿主张量）。**保守口径**：仍把视图源
//                    计为持久，与图级分配器的精确复用口径不同（FIX-002 / Q23）
// view_count       = 视图张量数（诊断用）
struct ContextMemoryStats {
    size_t tensor_bytes     = 0;
    size_t persistent_bytes = 0;
    size_t view_count       = 0;
};

ContextMemoryStats context_memory_stats(const Context* ctx);

} // namespace traincpp
