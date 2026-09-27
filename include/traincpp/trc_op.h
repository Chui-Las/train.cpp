// train.cpp - 算子枚举与名称
//
// 说明：ggml 的 enum ggml_op 数值不进入模型文件（GGUF 只保存 dtype 与张量数据），
// 因此本库使用自定义枚举顺序；算子“名称与数学语义”与 ggml 对齐（见 docs/兼容性.md）。
#pragma once

#include <cstdint>

namespace traincpp {

struct Tensor;

enum Op : int32_t {
    OP_NONE = 0,

    // ---- 数据移动/形状 ----
    OP_DUP,
    OP_CPY,
    OP_CONT,
    OP_RESHAPE,
    OP_VIEW,
    OP_PERMUTE,
    OP_TRANSPOSE,
    OP_CAST,
    OP_CONCAT,
    OP_REPEAT,
    OP_REPEAT_BACK,
    OP_PAD,
    OP_PAD_BACK,
    OP_ACC,

    // ---- 逐元素二元（支持 ggml 广播语义）----
    OP_ADD,
    OP_ADD1,
    OP_SUB,
    OP_MUL,
    OP_DIV,

    // ---- 逐元素一元 ----
    OP_SQR,
    OP_SQRT,
    OP_LOG,
    OP_EXP,
    OP_NEG,
    OP_ABS,
    OP_SGN,
    OP_STEP,
    OP_SIN,
    OP_COS,
    OP_CLAMP,

    // ---- 一元激活（对应 ggml_unary_op）----
    OP_RELU,
    OP_LEAKY_RELU,
    OP_GELU,
    OP_GELU_ERF,
    OP_ERF,
    OP_SILU,
    OP_SIGMOID,
    OP_TANH,
    OP_SOFTPLUS,
    OP_HARDSWISH,

    // ---- 缩放 ----
    OP_SCALE,

    // ---- 归约 ----
    OP_SUM,
    OP_MEAN,
    OP_SUM_ROWS,

    // ---- 索引 ----
    OP_GET_ROWS,
    OP_GET_ROWS_BACK,
    OP_SET_ROWS,

    // ---- 线性代数 ----
    OP_MUL_MAT,
    OP_OUT_PROD,

    // ---- 归一化 ----
    OP_NORM,
    OP_NORM_BACK,
    OP_RMS_NORM,
    OP_RMS_NORM_BACK,
    OP_GROUP_NORM,
    OP_GROUP_NORM_BACK,

    // ---- 注意力/损失 ----
    OP_SOFT_MAX,
    OP_CROSS_ENTROPY_LOSS,
    OP_CROSS_ENTROPY_LOSS_BACK,

    // ---- 卷积 ----
    OP_IM2COL,
    OP_IM2COL_BACK,
    OP_COL2IM,
    OP_CONV_1D,
    OP_CONV_2D,
    OP_CONV_TRANSPOSE_1D,
    // ---- 池化/转置卷积 2D（M2.3g）----
    OP_POOL_2D,
    OP_POOL_2D_BACK,  // 仅作为 pool_2d 反向结果使用，不参与二次反向
    OP_CONV_TRANSPOSE_2D,

    // ---- 优化器（设备端融合算子）----
    OP_OPT_STEP_ADAMW,
    OP_OPT_STEP_SGD,

    OP_COUNT,
};

// 算子名称（与 ggml_op_name 输出一致的部分优先沿用 ggml 命名）
const char* op_name(Op op);

// 算子的数学符号（用于日志/调试，与 ggml_op_symbol 对齐）
const char* op_symbol(Op op);

// 张量的算子描述：一元算子返回 unary 名称，其余返回 op_name
const char* op_desc(const Tensor* t);

} // namespace traincpp
