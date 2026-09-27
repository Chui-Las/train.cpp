// train.cpp - 计算图展开
#include "traincpp/trc_graph.h"

#include "core/trc_impl.h"

namespace traincpp {

Graph* graph_new(Context* ctx) {
    TRC_ASSERT(ctx != nullptr, "graph_new: ctx 为空");
    Graph* graph = new Graph();
    graph->ctx = ctx;
    return graph;
}

void graph_free(Graph* graph) {
    if (graph == nullptr) {
        return;
    }
    // 释放多图共享守卫的张量归属（M2.1e / R4）：图销毁后其参数/中间量可重新用于新图
    for (Tensor* node : graph->nodes) {
        if (node != nullptr && node->backward_graph == graph) {
            node->backward_graph = nullptr;
        }
    }
    delete graph;
}

void graph_clear(Graph* graph) {
    for (Tensor* node : graph->nodes) {
        if (node != nullptr && node->backward_graph == graph) {
            node->backward_graph = nullptr;
        }
    }
    graph->nodes.clear();
    graph->leafs.clear();
    graph->visited.clear();
    graph->backward_built = false;
}

void graph_build_forward_expand(Context* ctx, Graph* graph, Tensor* tensor) {
    TRC_ASSERT(ctx == graph->ctx, "graph_build_forward_expand: ctx 与 graph->ctx 不一致");
    if (tensor == nullptr) {
        return;
    }

    // 迭代式后序 DFS，避免深图递归爆栈。
    // 注意：用下标访问栈顶，不用 Frame&（push_back 可能重分配使引用失效，M2.1h / R17）。
    struct Frame {
        Tensor* tensor;
        int32_t next_src;
    };

    std::vector<Frame> stack;
    stack.push_back({tensor, 0});

    while (!stack.empty()) {
        const size_t top = stack.size() - 1;

        if (stack[top].next_src < stack[top].tensor->nsrc &&
            stack[top].tensor->src[stack[top].next_src] != nullptr) {
            Tensor* child = stack[top].tensor->src[stack[top].next_src++];
            if (graph->visited.count(child) == 0) {
                stack.push_back({child, 0});
            }
            continue;
        }

        Tensor* node = stack[top].tensor;
        stack.pop_back();
        if (graph->visited.insert(node).second) {
            graph->nodes.push_back(node);
            if (node->nsrc == 0) {
                graph->leafs.push_back(node);
            }
        }
    }
}

int64_t graph_n_nodes(const Graph* graph) {
    return (int64_t) graph->nodes.size();
}

Tensor* graph_node(const Graph* graph, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) graph->nodes.size(), "graph_node: 下标越界 %lld",
               (long long) index);
    return graph->nodes[(size_t) index];
}

int64_t graph_n_leafs(const Graph* graph) {
    return (int64_t) graph->leafs.size();
}

Tensor* graph_leaf(const Graph* graph, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) graph->leafs.size(), "graph_leaf: 下标越界 %lld",
               (long long) index);
    return graph->leafs[(size_t) index];
}

} // namespace traincpp
