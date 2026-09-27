// train.cpp - CPU 参考后端内部接口
#pragma once

#include "core/trc_backends.h"
#include "traincpp/traincpp.h"

namespace traincpp {

// CPU 设备单例（声明见 core/trc_backends.h）

// 该设备是否支持某节点的算子
bool cpu_supports_op(const Tensor* t);

// 执行单个计算节点（OP_NONE 叶子节点返回 true 但不做计算）
bool cpu_compute_node(Tensor* node);

} // namespace traincpp
