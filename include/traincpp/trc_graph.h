// train.cpp - 计算图
// 与 ggml 的 ggml_cgraph 对应：后序展开、节点去重、记录叶子节点。
#pragma once

#include "trc_tensor.h"

#include <unordered_set>
#include <vector>

namespace traincpp {

struct Graph {
    Context* ctx = nullptr;

    std::vector<Tensor*> nodes;   // 拓扑序（后序 DFS）
    std::vector<Tensor*> leafs;   // 无输入的节点（权重/输入张量）
    std::unordered_set<Tensor*> visited;

    // 训练扩展：反向图由 autograd 模块在同一 Graph 上继续追加（M1.3 实现）
    // 已构建反向标记（M2.1e / R15）：重复 graph_build_backward_expand 会中止
    bool backward_built = false;
};

Graph* graph_new(Context* ctx);
void   graph_free(Graph* graph);
// 清空图结构（节点/叶子/访问集合）；梯度字段与累加器保留（见 trc_autograd.h）；
// 同时释放反向构建标记与张量归属，允许在同一 Graph 上重新构建（M2.1e）。
//
// ⚠️ 反复建图（每 epoch/变长样本）会在同一 Context 中不断创建张量，registry 与
// 数据内存只增不减（M2.2c / R14）。长循环请配合 context_release_graph_tensors
// 回收图中间量（张量元数据仍在 arena 中，不回收；见 trc_context.h）。
void   graph_clear(Graph* graph);

// 与 ggml_build_forward_expand 语义一致：后序 DFS 展开并去重，
// 叶子节点（nsrc == 0）同时记录到 leafs。
// 注意：新算子节点会持续登记到 ctx（含反向展开新增的中间量），反复建图请配合
// context_release_graph_tensors（M2.2c / R14）。
void graph_build_forward_expand(Context* ctx, Graph* graph, Tensor* tensor);

int64_t graph_n_nodes(const Graph* graph);
Tensor* graph_node(const Graph* graph, int64_t index);
int64_t graph_n_leafs(const Graph* graph);
Tensor* graph_leaf(const Graph* graph, int64_t index);

} // namespace traincpp
