// train.cpp - 算子名称表
#include "traincpp/trc_op.h"

#include "core/trc_impl.h"

#include <array>

namespace traincpp {

namespace {

struct OpInfo {
    const char* name;
    const char* symbol;
};

using OpTable = std::array<OpInfo, OP_COUNT>;

const OpTable& op_table() {
    static const OpTable table = [] {
        OpTable t;
        t.fill({"unknown", "unknown"});

        t[OP_NONE]              = {"none", "none"};
        t[OP_DUP]               = {"dup", "dup"};
        t[OP_CPY]               = {"cpy", "cpy"};
        t[OP_CONT]              = {"cont", "cont"};
        t[OP_RESHAPE]           = {"reshape", "reshape"};
        t[OP_VIEW]              = {"view", "view"};
        t[OP_PERMUTE]           = {"permute", "permute"};
        t[OP_TRANSPOSE]         = {"transpose", "transpose"};
        t[OP_CAST]              = {"cast", "cast"};
        t[OP_CONCAT]            = {"concat", "concat"};
        t[OP_REPEAT]            = {"repeat", "repeat"};
        t[OP_REPEAT_BACK]       = {"repeat_back", "repeat_back"};
        t[OP_PAD]               = {"pad", "pad"};
        t[OP_PAD_BACK]          = {"pad_back", "pad_back(x)"};
        t[OP_ACC]               = {"acc", "acc"};
        t[OP_ADD]               = {"add", "x+y"};
        t[OP_ADD1]              = {"add1", "x+y"};
        t[OP_SUB]               = {"sub", "x-y"};
        t[OP_MUL]               = {"mul", "x*y"};
        t[OP_DIV]               = {"div", "x/y"};
        t[OP_SQR]               = {"sqr", "x^2"};
        t[OP_SQRT]              = {"sqrt", "sqrt(x)"};
        t[OP_LOG]               = {"log", "log(x)"};
        t[OP_EXP]               = {"exp", "exp(x)"};
        t[OP_NEG]               = {"neg", "-x"};
        t[OP_ABS]               = {"abs", "|x|"};
        t[OP_SGN]               = {"sgn", "sgn(x)"};
        t[OP_STEP]              = {"step", "step(x)"};
        t[OP_SIN]               = {"sin", "sin(x)"};
        t[OP_COS]               = {"cos", "cos(x)"};
        t[OP_CLAMP]             = {"clamp", "clamp(x)"};
        t[OP_RELU]              = {"relu", "relu(x)"};
        t[OP_LEAKY_RELU]        = {"leaky_relu", "leaky_relu(x)"};
        t[OP_GELU]              = {"gelu", "gelu(x)"};
        t[OP_GELU_ERF]          = {"gelu_erf", "gelu_erf(x)"};
        t[OP_ERF]               = {"erf", "erf(x)"};
        t[OP_SILU]              = {"silu", "silu(x)"};
        t[OP_SIGMOID]           = {"sigmoid", "sigmoid(x)"};
        t[OP_TANH]              = {"tanh", "tanh(x)"};
        t[OP_SOFTPLUS]          = {"softplus", "softplus(x)"};
        t[OP_HARDSWISH]         = {"hardswish", "hardswish(x)"};
        t[OP_SCALE]             = {"scale", "x*s"};
        t[OP_SUM]               = {"sum", "sum(x)"};
        t[OP_MEAN]              = {"mean", "mean(x)"};
        t[OP_SUM_ROWS]          = {"sum_rows", "sum_rows(x)"};
        t[OP_GET_ROWS]          = {"get_rows", "get_rows(x)"};
        t[OP_GET_ROWS_BACK]     = {"get_rows_back", "get_rows_back(x)"};
        t[OP_SET_ROWS]          = {"set_rows", "set_rows(x)"};
        t[OP_MUL_MAT]           = {"mul_mat", "X*Y"};
        t[OP_OUT_PROD]          = {"out_prod", "X*Y"};
        t[OP_NORM]              = {"norm", "norm(x)"};
        t[OP_NORM_BACK]         = {"norm_back", "norm_back(x)"};
        t[OP_RMS_NORM]          = {"rms_norm", "rms_norm(x)"};
        t[OP_RMS_NORM_BACK]     = {"rms_norm_back", "rms_norm_back(x)"};
        t[OP_GROUP_NORM]        = {"group_norm", "group_norm(x)"};
        t[OP_GROUP_NORM_BACK]   = {"group_norm_back", "group_norm_back(x)"};
        t[OP_SOFT_MAX]          = {"soft_max", "soft_max(x)"};
        t[OP_CROSS_ENTROPY_LOSS] = {"cross_entropy_loss", "cross_entropy_loss(x)"};
        t[OP_CROSS_ENTROPY_LOSS_BACK] = {"cross_entropy_loss_back", "cross_entropy_loss_back(x)"};
        t[OP_IM2COL]            = {"im2col", "im2col(x)"};
        t[OP_IM2COL_BACK]       = {"im2col_back", "im2col_back(x)"};
        t[OP_COL2IM]            = {"col2im_1d", "col2im_1d(x)"};
        t[OP_CONV_1D]           = {"conv_1d", "conv_1d(x)"};
        t[OP_CONV_2D]           = {"conv_2d", "conv_2d(x)"};
        t[OP_CONV_TRANSPOSE_1D] = {"conv_transpose_1d", "conv_transpose_1d(x)"};
        t[OP_POOL_2D]           = {"pool_2d", "pool_2d(x)"};
        t[OP_POOL_2D_BACK]      = {"pool_2d_back", "pool_2d_back(x)"};
        t[OP_CONV_TRANSPOSE_2D] = {"conv_transpose_2d", "conv_transpose_2d(x)"};
        t[OP_OPT_STEP_ADAMW]    = {"opt_step_adamw", "opt_step_adamw(x)"};
        t[OP_OPT_STEP_SGD]      = {"opt_step_sgd", "opt_step_sgd(x)"};
        return t;
    }();
    return table;
}

} // namespace

const char* op_name(Op op) {
    const int32_t v = (int32_t) op;
    if (v < 0 || v >= OP_COUNT) {
        return "unknown";
    }
    return op_table()[(size_t) v].name;
}

const char* op_symbol(Op op) {
    const int32_t v = (int32_t) op;
    if (v < 0 || v >= OP_COUNT) {
        return "unknown";
    }
    return op_table()[(size_t) v].symbol;
}

const char* op_desc(const Tensor* t) {
    return op_name(t->op);
}

} // namespace traincpp
