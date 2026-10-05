// train.cpp - 自动求导引擎
// 反向通过前向算子组合实现：每个规则只调用 include/traincpp/trc_ops.h 中的公共构造。
#include "traincpp/trc_autograd.h"

#include "core/trc_impl.h"
#include "traincpp/trc_backend.h"
#include "traincpp/trc_ops.h"

#include <cstring>
#include <unordered_set>
#include <vector>

namespace traincpp {

namespace {

// ---------------------------------------------------------------- 小工具

bool has_flag(const Tensor* t, int32_t flag) {
    return (t->flags & flag) != 0;
}

// 读取 op_params 中第 index 个 float
float param_f32(const Tensor* t, int index) {
    float v = 0.0f;
    std::memcpy(&v, (const uint8_t*) t->op_params + (size_t) index * sizeof(float), sizeof(float));
    return v;
}

bool grad_needed(const std::unordered_set<Tensor*>& needed, const Tensor* t) {
    return t != nullptr && needed.count(const_cast<Tensor*>(t)) != 0;
}

// 反向不需要梯度的输入（与 ggml 的 ignore_src 对应）
bool ignore_src(const Tensor* t, int32_t j) {
    switch (t->op) {
        case OP_IM2COL:
            return j == 0;  // 卷积核的梯度经 reshape/mul_mat 路径回流，im2col 数值不依赖核
        case OP_GET_ROWS:
            return j == 1;  // 索引不可微
        case OP_SOFT_MAX:
            return j == 1;  // mask 视为常量（同 ggml）
        case OP_CROSS_ENTROPY_LOSS:
            return j == 1;  // target 视为常量（同 ggml_cross_entropy_loss_back）
        default:
            return false;
    }
}

void build_expand(Graph* graph, Tensor* t) {
    graph_build_forward_expand(graph->ctx, graph, t);
}

// ---------------------------------------------------------------- 梯度累加

// 持久累加器（PARAM/LOSS）的 inplace 累加（与 ggml 的 inplace add 语义一致）：
//   把 contribution 累加进 target->grad_acc（跨多次 compute 保留 → 支持梯度累积）；
//   region 为 nullptr 时按 contribution 的连续步长整体累加；
//   否则按 region 视图的步长/偏移写入对应区域（视图反向）。
void grad_accumulate_inplace(Graph* graph, Tensor* target, Tensor* contribution,
                             const Tensor* region) {
    TRC_ASSERT(target->grad_acc != nullptr, "grad_accumulate_inplace: 目标没有梯度累加器");
    Context* ctx = graph->ctx;

    // acc 要求源连续：非连续贡献（转置/带步长视图）先物化
    Tensor* c = tensor_is_contiguous(contribution) ? contribution : cont(ctx, contribution);

    const size_t nb1    = region != nullptr ? region->nb[1] : c->nb[1];
    const size_t nb2    = region != nullptr ? region->nb[2] : c->nb[2];
    const size_t nb3    = region != nullptr ? region->nb[3] : c->nb[3];
    const size_t offset = region != nullptr ? region->view_offs : 0;

    Tensor* node = acc(ctx, target->grad_acc, c, nb1, nb2, nb3, offset, /*inplace=*/true);
    target->grad = target->grad_acc;
    tensor_set_name(target->grad, "grad for %s", target->name);
    build_expand(graph, node);
}

// target->grad += contribution（形状必须一致）
void grad_add_or_set(Graph* graph, Tensor* target, Tensor* contribution) {
    TRC_ASSERT(contribution != nullptr, "grad_add_or_set: 贡献梯度为空");
    TRC_ASSERT(tensor_are_same_shape(target, contribution),
               "梯度形状不一致: %s(%s) [%lld,%lld,%lld,%lld] vs %s [%lld,%lld,%lld,%lld]",
               target->name, op_name(target->op), (long long) target->ne[0], (long long) target->ne[1],
               (long long) target->ne[2], (long long) target->ne[3], op_name(contribution->op),
               (long long) contribution->ne[0], (long long) contribution->ne[1],
               (long long) contribution->ne[2], (long long) contribution->ne[3]);
    if (target->grad_acc != nullptr) {
        grad_accumulate_inplace(graph, target, contribution, nullptr);
        return;
    }
    if (target->grad == nullptr) {
        target->grad = contribution;
    } else {
        target->grad = add(graph->ctx, target->grad, contribution);
    }
    tensor_set_name(target->grad, "grad for %s", target->name);
    build_expand(graph, target->grad);
}

// target->grad -= contribution
void grad_sub_or_set(Graph* graph, Tensor* target, Tensor* contribution) {
    TRC_ASSERT(contribution != nullptr, "grad_sub_or_set: 贡献梯度为空");
    TRC_ASSERT(tensor_are_same_shape(target, contribution), "梯度形状不一致: %s vs %s", target->name,
               contribution->name);
    if (target->grad_acc != nullptr) {
        grad_accumulate_inplace(graph, target, neg(graph->ctx, contribution), nullptr);
        return;
    }
    if (target->grad == nullptr) {
        target->grad = neg(graph->ctx, contribution);
    } else {
        target->grad = sub(graph->ctx, target->grad, contribution);
    }
    tensor_set_name(target->grad, "grad for %s", target->name);
    build_expand(graph, target->grad);
}

// 标量贡献：当 target 还没有梯度时把标量 repeat 成 target 形状（与 ggml 的 add1_or_set 一致）
void grad_add1_or_set(Graph* graph, Tensor* target, Tensor* scalar_grad) {
    TRC_ASSERT(tensor_nelements(scalar_grad) == 1, "grad_add1_or_set: 贡献必须是标量");
    if (target->grad_acc != nullptr) {
        grad_accumulate_inplace(graph, target, repeat(graph->ctx, scalar_grad, target), nullptr);
        return;
    }
    if (target->grad == nullptr) {
        target->grad = repeat(graph->ctx, scalar_grad, target);
    } else {
        target->grad = add1(graph->ctx, target->grad, scalar_grad);
    }
    tensor_set_name(target->grad, "grad for %s", target->name);
    build_expand(graph, target->grad);
}

// 视图区域累加：contribution 的形状 = 视图 view 的形状，写入 target->grad 的对应区域
void grad_acc_or_set(Graph* graph, Tensor* target, Tensor* contribution, const Tensor* view) {
    TRC_ASSERT(view->view_src == target, "grad_acc_or_set: 视图与目标不匹配");
    TRC_ASSERT(tensor_is_contiguous(target), "grad_acc_or_set: 目标张量必须连续");

    if (target->grad_acc != nullptr) {
        // 参数/损失：inplace 累加进持久累加器（region 布局取自视图步长）
        grad_accumulate_inplace(graph, target, contribution, view);
        return;
    }

    if (target->grad == nullptr) {
        // 非参数中间节点：先构造全零基础张量
        target->grad = scale(graph->ctx, target, 0.0f);
        tensor_set_name(target->grad, "grad for %s", target->name);
        build_expand(graph, target->grad);
    }
    // 非 inplace：结果是与 target 同形状的完整张量（region 之外保持不变）
    target->grad = acc(graph->ctx, target->grad, contribution, view->nb[1], view->nb[2], view->nb[3],
                       view->view_offs, /*inplace=*/false);
    tensor_set_name(target->grad, "grad for %s", target->name);
    build_expand(graph, target->grad);
}

// ---------------------------------------------------------------- 逐算子反向规则

void backward_op(Graph* graph, Tensor* tensor, const std::unordered_set<Tensor*>& needed) {
    Tensor* grad = tensor->grad;
    if (grad == nullptr) {
        return;
    }

    Context* ctx = graph->ctx;
    Tensor*  src0 = tensor->src[0];
    Tensor*  src1 = tensor->src[1];
    const bool need0 = grad_needed(needed, src0);
    const bool need1 = grad_needed(needed, src1);

    switch (tensor->op) {
        case OP_NONE:
            break;

        // ---------------- 数据移动/形状 ----------------
        case OP_DUP:
        case OP_CONT:
            if (need0) {
                grad_add_or_set(graph, src0, grad);
            }
            break;
        case OP_RESHAPE: {
            if (need0) {
                Tensor* gc = cont(ctx, grad);
                Tensor* r = reshape_4d(ctx, gc, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
                grad_add_or_set(graph, src0, r);
            }
        } break;
        case OP_VIEW: {
            if (need0) {
                grad_acc_or_set(graph, src0, grad, tensor);
            }
        } break;
        case OP_PERMUTE: {
            if (need0) {
                // 轴序的逆置换：axes[i] = k  =>  inv[k] = i
                int inv[MAX_DIMS] = {0, 1, 2, 3};
                for (int i = 0; i < MAX_DIMS; ++i) {
                    inv[tensor->op_params[i] & 0x3] = i;
                }
                grad_add_or_set(graph, src0, permute(ctx, grad, inv[0], inv[1], inv[2], inv[3]));
            }
        } break;
        case OP_TRANSPOSE: {
            if (need0) {
                grad_add_or_set(graph, src0, transpose(ctx, grad));
            }
        } break;
        case OP_CAST: {
            if (need0) {
                grad_add_or_set(graph, src0, cast(ctx, grad, src0->type));
            }
        } break;
        case OP_ACC:
            TRC_ABORT("反向尚未支持 acc 节点（将在 M1.3b 评估）");

        // ---------------- 逐元素二元 ----------------
        case OP_ADD: {
            if (need0) {
                grad_add_or_set(graph, src0, grad);
            }
            if (need1) {
                Tensor* c = tensor_are_same_shape(src0, src1) ? grad : repeat_back(ctx, grad, src1);
                grad_add_or_set(graph, src1, c);
            }
        } break;
        case OP_ADD1: {
            if (need0) {
                grad_add_or_set(graph, src0, grad);
            }
            if (need1) {
                // dL/db = sum(grad)（标量；ggml 现用 mean，本库取数学正确的 sum）
                grad_add1_or_set(graph, src1, sum(ctx, grad));
            }
        } break;
        case OP_SUB: {
            if (need0) {
                grad_add_or_set(graph, src0, grad);
            }
            if (need1) {
                Tensor* c = tensor_are_same_shape(src0, src1) ? grad : repeat_back(ctx, grad, src1);
                grad_sub_or_set(graph, src1, c);
            }
        } break;
        case OP_MUL: {
            if (need0) {
                grad_add_or_set(graph, src0, mul(ctx, grad, src1));
            }
            if (need1) {
                Tensor* c = mul(ctx, grad, src0);
                if (!tensor_are_same_shape(src0, src1)) {
                    c = repeat_back(ctx, c, src1);
                }
                grad_add_or_set(graph, src1, c);
            }
        } break;
        case OP_DIV: {
            if (need0) {
                grad_add_or_set(graph, src0, div(ctx, grad, src1));
            }
            if (need1) {
                // dL/db = -grad * a / b^2 = -grad * (y / b)
                Tensor* c = mul(ctx, grad, div(ctx, tensor, src1));
                if (!tensor_are_same_shape(src0, src1)) {
                    c = repeat_back(ctx, c, src1);
                }
                grad_sub_or_set(graph, src1, c);
            }
        } break;

        // ---------------- 逐元素一元 ----------------
        case OP_NEG:
            if (need0) {
                grad_sub_or_set(graph, src0, grad);
            }
            break;
        case OP_ABS:
            if (need0) {
                grad_add_or_set(graph, src0, mul(ctx, grad, sgn(ctx, src0)));
            }
            break;
        case OP_SGN:
        case OP_STEP:
            // 分段常量，导数为 0
            break;
        case OP_SQR:
            if (need0) {
                grad_add_or_set(graph, src0, scale(ctx, mul(ctx, src0, grad), 2.0f));
            }
            break;
        case OP_SQRT:
            if (need0) {
                grad_add_or_set(graph, src0, scale(ctx, div(ctx, grad, tensor), 0.5f));
            }
            break;
        case OP_EXP:
            if (need0) {
                grad_add_or_set(graph, src0, mul(ctx, tensor, grad));
            }
            break;
        case OP_LOG:
            if (need0) {
                grad_add_or_set(graph, src0, div(ctx, grad, src0));
            }
            break;
        case OP_SIN:
            if (need0) {
                grad_add_or_set(graph, src0, mul(ctx, grad, cos_op(ctx, src0)));
            }
            break;
        case OP_COS:
            if (need0) {
                grad_sub_or_set(graph, src0, mul(ctx, grad, sin_op(ctx, src0)));
            }
            break;
        case OP_CLAMP: {
            if (need0) {
                const float lo = param_f32(tensor, 0);
                const float hi = param_f32(tensor, 1);
                Tensor* c_hi = repeat(ctx, new_scalar_const(ctx, hi), src0);
                Tensor* mask = mul(ctx, step(ctx, sub(ctx, src0, new_scalar_const(ctx, lo))),
                                   step(ctx, sub(ctx, c_hi, src0)));
                grad_add_or_set(graph, src0, mul(ctx, grad, mask));
            }
        } break;
        case OP_SCALE:
            if (need0) {
                grad_add_or_set(graph, src0, scale(ctx, grad, param_f32(tensor, 0)));
            }
            break;

        // ---------------- 激活 ----------------
        case OP_RELU:
            if (need0) {
                grad_add_or_set(graph, src0, mul(ctx, grad, step(ctx, src0)));
            }
            break;
        case OP_LEAKY_RELU: {
            if (need0) {
                // PyTorch 语义：x > 0 → 1；x <= 0（含 ±0）→ slope。
                // 旧实现 step(x) + slope*step(-x) 在 x==±0 处得 0：
                //   1) 与 PyTorch 的 leaky_relu 反向不一致；
                //   2) 也不是该分段函数的合法次梯度（合法区间 [slope, 1]）。
                const float slope = param_f32(tensor, 0);
                Tensor* mask = add1(ctx, scale(ctx, step(ctx, src0), 1.0f - slope), new_scalar_const(ctx, slope));
                grad_add_or_set(graph, src0, mul(ctx, grad, mask));
            }
        } break;
        case OP_SIGMOID:
            if (need0) {
                // y' = y * (1 - y) = y * sigmoid(-x)
                grad_add_or_set(graph, src0, mul(ctx, grad, mul(ctx, tensor, sigmoid(ctx, neg(ctx, src0)))));
            }
            break;
        case OP_TANH:
            if (need0) {
                // y' = 1 - y^2
                Tensor* one = repeat(ctx, new_scalar_const(ctx, 1.0f), tensor);
                grad_add_or_set(graph, src0, mul(ctx, grad, sub(ctx, one, sqr(ctx, tensor))));
            }
            break;
        case OP_SILU:
            if (need0) {
                // y' = sig + x*sig*(1-sig) = sig + x*sig*sig(-x)
                Tensor* sig  = sigmoid(ctx, src0);
                Tensor* sneg = sigmoid(ctx, neg(ctx, src0));
                Tensor* dy   = add(ctx, sig, mul(ctx, src0, mul(ctx, sig, sneg)));
                grad_add_or_set(graph, src0, mul(ctx, grad, dy));
            }
            break;
        case OP_SOFTPLUS:
            if (need0) {
                grad_add_or_set(graph, src0, mul(ctx, grad, sigmoid(ctx, src0)));
            }
            break;
        case OP_HARDSWISH: {
            if (need0) {
                // h(x) = x*relu6(x+3)/6；h' = relu6(x+3)/6 + x*(step(x+3)-step(x-3))/6
                Tensor* c3  = new_scalar_const(ctx, 3.0f);
                Tensor* cm3 = new_scalar_const(ctx, -3.0f);
                Tensor* u   = add1(ctx, src0, c3);  // x + 3
                Tensor* t   = scale(ctx, clamp(ctx, u, 0.0f, 6.0f), 1.0f / 6.0f);
                Tensor* dt  = scale(ctx, sub(ctx, step(ctx, u), step(ctx, add1(ctx, src0, cm3))),
                                    1.0f / 6.0f);
                grad_add_or_set(graph, src0, mul(ctx, grad, add(ctx, t, mul(ctx, src0, dt))));
            }
        } break;
        case OP_GELU: {
            if (need0) {
                // tanh 近似：u = c0*(x + c1*x^3)，y = 0.5x(1+tanh u)
                const float c0 = 0.7978845608028654f;
                const float c1 = 0.044715f;
                Tensor* one = new_scalar_const(ctx, 1.0f);
                Tensor* x2  = sqr(ctx, src0);
                Tensor* x3  = mul(ctx, x2, src0);
                Tensor* u   = scale(ctx, add(ctx, src0, scale(ctx, x3, c1)), c0);
                Tensor* t   = tanh_op(ctx, u);
                Tensor* omt2 = sub(ctx, repeat(ctx, one, t), sqr(ctx, t));          // 1 - t^2
                Tensor* du   = scale(ctx, add1(ctx, scale(ctx, x2, 3.0f * c1), one), c0);  // du/dx
                Tensor* dy   = add(ctx, scale(ctx, add1(ctx, t, one), 0.5f),
                                   scale(ctx, mul(ctx, src0, mul(ctx, omt2, du)), 0.5f));
                grad_add_or_set(graph, src0, mul(ctx, grad, dy));
            }
        } break;
        case OP_GELU_ERF: {
            if (need0) {
                // y = 0.5x(1+erf(x/√2))；y' = 0.5(1+erf(x/√2)) + x*exp(-x²/2)/√(2π)
                const float c0 = 0.3989422804014327f;  // 1/sqrt(2*pi)
                Tensor* one   = new_scalar_const(ctx, 1.0f);
                Tensor* u     = scale(ctx, src0, 0.7071067811865476f);
                Tensor* term1 = scale(ctx, add1(ctx, erf_op(ctx, u), one), 0.5f);
                Tensor* term2 = scale(ctx, mul(ctx, src0, exp_op(ctx, scale(ctx, sqr(ctx, src0), -0.5f))), c0);
                grad_add_or_set(graph, src0, mul(ctx, grad, add(ctx, term1, term2)));
            }
        } break;

        // ---------------- 归约 ----------------
        case OP_SUM:
            if (need0) {
                grad_add1_or_set(graph, src0, grad);
            }
            break;
        case OP_SUM_ROWS:
            if (need0) {
                grad_add_or_set(graph, src0, repeat(ctx, grad, src0));
            }
            break;
        case OP_MEAN:
            if (need0) {
                Tensor* g = scale(ctx, grad, 1.0f / (float) src0->ne[0]);
                grad_add_or_set(graph, src0, repeat(ctx, g, src0));
            }
            break;

        // ---------------- 形状扩展 ----------------
        case OP_REPEAT:
            if (need0) {
                grad_add_or_set(graph, src0, repeat_back(ctx, grad, src0));
            }
            break;
        case OP_REPEAT_BACK:
            if (need0) {
                grad_add_or_set(graph, src0, repeat(ctx, grad, src0));
            }
            break;
        case OP_CONCAT: {
            const int dim = tensor->op_params[0];
            if (need0 || need1) {
                Tensor* gc = cont(ctx, grad);
                if (need0) {
                    grad_add_or_set(graph, src0,
                                    view_4d(ctx, gc, src0->ne[0], src0->ne[1], src0->ne[2],
                                            src0->ne[3], 0));
                }
                if (need1) {
                    const size_t offset = (size_t) src0->ne[dim] * gc->nb[dim];
                    grad_add_or_set(graph, src1,
                                    view_4d(ctx, gc, src1->ne[0], src1->ne[1], src1->ne[2],
                                            src1->ne[3], offset));
                }
            }
        } break;
        case OP_PAD: {
            if (need0) {
                const int32_t* p    = tensor->op_params;
                const int      mode = p[8];
                if (mode == (int) PAD_REFLECT) {
                    // 镜像反射反向：dX[j] = Σ_{i: reflect(i-lp)=j} dY[i]（专用内核）
                    grad_add_or_set(graph, src0,
                                    pad_back(ctx, grad, src0, p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                                             p[7]));
                } else {
                    // 补零反向：取中心区域（左填充偏移 = Σ lp[d]*nb[d]）
                    Tensor* gc = cont(ctx, grad);
                    const size_t offset = (size_t) p[0] * gc->nb[0] + (size_t) p[2] * gc->nb[1] +
                                          (size_t) p[4] * gc->nb[2] + (size_t) p[6] * gc->nb[3];
                    grad_add_or_set(graph, src0,
                                    view_4d(ctx, gc, src0->ne[0], src0->ne[1], src0->ne[2],
                                            src0->ne[3], offset));
                }
            }
        } break;

        // ---------------- 线性代数 ----------------
        case OP_MUL_MAT: {
            // grad: [m, n, b2, b3]；src0: [k, m, a2, a3]；src1: [k, n, b2, b3]
            // dA = (grad^T * B^T)^T = sum_j grad[i,j]*B[k,j]（每平面独立）
            // dB = A^T * grad（mul_mat 自动按 a 的平面广播）
            if (need0) {
                Tensor* dA = transpose(ctx, mul_mat(ctx, transpose(ctx, grad), transpose(ctx, src1)));
                // a 的 batch 被广播时，沿 b 的平面把 dA 归约回 a 的形状（取模分组求和）
                if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
                    dA = repeat_back(ctx, dA, src0);
                }
                grad_add_or_set(graph, src0, dA);
            }
            if (need1) {
                Tensor* dB = mul_mat(ctx, transpose(ctx, src0), grad);
                grad_add_or_set(graph, src1, dB);
            }
        } break;

        // ---------------- 注意力 ----------------
        case OP_SOFT_MAX: {
            if (need0) {
                // dx = scale * y * (grad - sum_rows(grad*y))
                const float s = param_f32(tensor, 0);
                Tensor* gy = mul(ctx, grad, tensor);
                Tensor* d  = sub(ctx, grad, sum_rows(ctx, gy));
                grad_add_or_set(graph, src0, scale(ctx, mul(ctx, tensor, d), s));
            }
        } break;

        // ---------------- 损失 ----------------
        case OP_CROSS_ENTROPY_LOSS: {
            if (need0) {
                // dA = (softmax(a) - target) * dloss / nr（专用 kernel，与 ggml 一致）
                grad_add_or_set(graph, src0, cross_entropy_loss_back(ctx, grad, src0, src1));
            }
        } break;

        // ---------------- 索引 ----------------
        case OP_GET_ROWS: {
            if (need0) {
                // 查表反向：按索引把梯度行 scatter-add 回表（非参数索引不参与）
                grad_add_or_set(graph, src0, get_rows_back(ctx, grad, src1, src0));
            }
        } break;

        // ---------------- 卷积 ----------------
        case OP_IM2COL: {
            if (need1) {
                // 对输入图像求梯度；卷积核的梯度经 reshape/mul_mat 路径回流（ignore_src 已屏蔽 src0）
                const int s0 = tensor->op_params[0];
                const int s1 = tensor->op_params[1];
                const int p0 = tensor->op_params[2];
                const int p1 = tensor->op_params[3];
                const int d0 = tensor->op_params[4];
                const int d1 = tensor->op_params[5];
                const bool is_2d = tensor->op_params[6] == 1;
                Tensor* g = cont(ctx, grad);
                grad_add_or_set(graph, src1,
                                im2col_back(ctx, g, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2d));
            }
        } break;

        // ---------------- 归一化 ----------------
        case OP_NORM:
            if (need0) {
                grad_add_or_set(graph, src0, norm_back(ctx, grad, src0, param_f32(tensor, 0)));
            }
            break;
        case OP_RMS_NORM:
            if (need0) {
                grad_add_or_set(graph, src0, rms_norm_back(ctx, grad, src0, param_f32(tensor, 0)));
            }
            break;
        case OP_GROUP_NORM:
            if (need0) {
                grad_add_or_set(graph, src0,
                                group_norm_back(ctx, grad, src0, tensor->op_params[0],
                                                param_f32(tensor, 1)));
            }
            break;

        // ---------------- 转置卷积（M2.3b）----------------
        case OP_CONV_TRANSPOSE_1D: {
            // 前向（p0=0、d0=1）：dst[t*s+k, co] += Σ_ci W[k,co,ci] * X[t,ci]
            // 令 col_dy[co*K+k, t] = dY[t*s+k, co]（im2col：kernel 仅取形状，值不参与）
            //   dW[k,co,ci] = Σ_t dY[t*s+k,co] * X[t,ci]
            //               = reshape_3d(mul_mat(transpose(col_dy), X), K, Cout, Cin)
            //   dX[t,ci]    = Σ_{k,co} dY[t*s+k,co] * W[k,co,ci]
            //               = mul_mat(col_dy, reshape_2d(cont(W), K*Cout, Cin))
            const int64_t K    = src0->ne[0];
            const int64_t Cout = src0->ne[1];
            const int64_t Cin  = src0->ne[2];
            const int     s0   = tensor->op_params[0];

            // 形状载体：W 的 [K, Cout, 1, 1] 视图（im2col 只用 kernel 形状）
            Tensor* kernel_shape = view_4d(ctx, src0, K, Cout, 1, 1, 0);
            // grad 形状 [T_out, Cout, 1, 1]；im2col 按 nb 寻址，无需 cont
            Tensor* col_dy =
                im2col(ctx, kernel_shape, grad, s0, 0, 0, 0, 1, 0, false, TYPE_F32);

            if (need0) {
                Tensor* dW = reshape_3d(ctx, mul_mat(ctx, transpose(ctx, col_dy), src1), K, Cout, Cin);
                grad_add_or_set(graph, src0, dW);
            }
            if (need1) {
                Tensor* w2 = reshape_2d(ctx, cont(ctx, src0), K * Cout, Cin);
                grad_add_or_set(graph, src1, mul_mat(ctx, col_dy, w2));
            }
        } break;

        // ---------------- 池化 / 2D 转置卷积（M2.3g）----------------
        case OP_POOL_2D: {
            // 专用反向 kernel（CPU）：AVG 均分核面积、MAX 归到首个严格最大位置
            if (need0) {
                const int32_t* p = tensor->op_params;
                grad_add_or_set(graph, src0,
                                pool_2d_back(ctx, grad, src0, (PoolMode) p[0], p[1], p[2], p[3],
                                             p[4], p[5], p[6]));
            }
        } break;
        case OP_CONV_TRANSPOSE_2D: {
            // 前向（p=0、d=1）：dst[ow,oh,co,n] = Σ W[kw,kh,co,ci]*X[iw,ih,ci,n]
            //   （iw=(ow-kw)/stride、ih=(oh-kh)/stride，需整除且在界内）
            // 令 col_dy[co*KH*KW+kh*KW+kw, j*W+i, n] = dY[i*stride+kw, j*stride+kh, co, n]
            //   （im2col 只用核形状；行序 ic*KH*KW+kh*KW+kw，与 reshape(W) 的行序一致）
            //   dW = reshape_4d(mul_mat(transpose(A), B), KW, KH, Cout, Cin)
            //   dX = permute(reshape_4d(mul_mat(A, W2), W, H, N, Cin), 0,1,3,2)
            // 其中 A=[R, P*N]（R=Cout*KH*KW，P=W*H，列 q=n*P + j*W + i）、
            //      B=[P*N, Cin]（由 X 的 [P,Cin,N] 置换后 reshape）、W2=[R, Cin]
            const int64_t KW   = src0->ne[0];
            const int64_t KH   = src0->ne[1];
            const int64_t Cout = src0->ne[2];
            const int64_t Cin  = src0->ne[3];
            const int     stride = tensor->op_params[0];
            const int64_t W = src1->ne[0];
            const int64_t H = src1->ne[1];
            const int64_t N = src1->ne[3];
            const int64_t P = W * H;
            const int64_t R = Cout * KH * KW;

            Tensor* kernel_shape = view_4d(ctx, src0, KW, KH, Cout, 1, 0);
            Tensor* col_dy =
                im2col(ctx, kernel_shape, grad, stride, stride, 0, 0, 1, 1, true, TYPE_F32);
            Tensor* A = reshape_2d(ctx, col_dy, R, P * N);

            if (need0) {
                Tensor* x3 = reshape_3d(ctx, cont(ctx, src1), P, Cin, N);  // [P, Cin, N]
                Tensor* b2 = reshape_2d(ctx, cont(ctx, permute(ctx, x3, 0, 2, 1, 3)), P * N, Cin);
                Tensor* dW = reshape_4d(ctx, mul_mat(ctx, transpose(ctx, A), b2), KW, KH, Cout, Cin);
                grad_add_or_set(graph, src0, dW);
            }
            if (need1) {
                Tensor* w2  = reshape_2d(ctx, cont(ctx, src0), R, Cin);
                Tensor* dx4 = reshape_4d(ctx, mul_mat(ctx, A, w2), W, H, N, Cin);
                grad_add_or_set(graph, src1, cont(ctx, permute(ctx, dx4, 0, 1, 3, 2)));
            }
        } break;

        // ---------------- 1.3b 及以后 ----------------
        // 设备端优化器步（M4.1）：非可微副作用算子（就地更新参数），不参与反向传播
        case OP_OPT_STEP_ADAMW:
        case OP_OPT_STEP_SGD:
        // 设备端梯度裁剪 / WeightNorm 同步（M4.5）：同为不可微副作用算子
        case OP_SUM_SQR_ACC:
        case OP_CLIP_SCALE_INPLACE:
        case OP_WEIGHTNORM_SYNC:
            break;

        case OP_COL2IM:
        case OP_PAD_BACK:  // 仅作为反向结果使用，不参与二次反向
        case OP_POOL_2D_BACK:  // 仅作为 pool_2d 反向结果使用，不参与二次反向
        case OP_OUT_PROD:
        case OP_CROSS_ENTROPY_LOSS_BACK:  // 仅作为反向结果使用，不参与二次反向
        case OP_CPY:
        case OP_SET_ROWS:
        default:
            TRC_ABORT("算子 %s 的反向尚未实现", op_name(tensor->op));
    }
}

} // namespace

// ---------------------------------------------------------------- 公共 API

void tensor_set_param(Tensor* t) {
    TRC_ASSERT(t != nullptr, "tensor_set_param: 张量为空");
    TRC_ASSERT(t->op == OP_NONE, "tensor_set_param: 只能标记无输入的叶子张量（当前 op=%s）",
               op_name(t->op));
    t->flags |= TENSOR_FLAG_PARAM;
}

void tensor_clear_param(Tensor* t) {
    TRC_ASSERT(t != nullptr, "tensor_clear_param: 张量为空");
    t->flags &= ~TENSOR_FLAG_PARAM;
}

void tensor_set_loss(Tensor* t) {
    TRC_ASSERT(t != nullptr, "tensor_set_loss: 张量为空");
    TRC_ASSERT(t->type == TYPE_F32, "tensor_set_loss: 损失必须是 f32");
    TRC_ASSERT(tensor_nelements(t) == 1, "tensor_set_loss: 损失必须是标量");
    t->flags |= TENSOR_FLAG_LOSS;
}

void graph_build_backward_expand(Graph* graph) {
    TRC_ASSERT(graph != nullptr, "graph_build_backward_expand: graph 为空");
    // R15：重复展开会叠加反向节点导致梯度静默翻倍
    TRC_ASSERT(!graph->backward_built,
               "graph_build_backward_expand: 同一 Graph 已构建过反向（重复展开会静默双倍梯度，R15）；"
               "请先 graph_clear 或新建 Graph");
    Context* ctx = graph->ctx;
    const int64_t n_fwd = (int64_t) graph->nodes.size();
    TRC_ASSERT(n_fwd > 0, "graph_build_backward_expand: 前向图为空");

    // 0) 清理上一次构建的梯度指针（grad_acc 保留复用）
    bool any_param = false;
    bool any_loss  = false;
    for (int64_t i = 0; i < n_fwd; ++i) {
        Tensor* node = graph->nodes[(size_t) i];
        node->grad = nullptr;
        any_param = any_param || has_flag(node, TENSOR_FLAG_PARAM);
        any_loss  = any_loss || has_flag(node, TENSOR_FLAG_LOSS);
    }
    TRC_ASSERT(any_param, "没有可训练参数：请先调用 tensor_set_param");
    TRC_ASSERT(any_loss, "没有损失：请先调用 tensor_set_loss");

    // 1) grads_needed：沿拓扑序传播（只对参数→损失路径上的节点求梯度）
    std::unordered_set<Tensor*> needed;
    needed.reserve((size_t) n_fwd * 2);
    for (int64_t i = 0; i < n_fwd; ++i) {
        Tensor* node = graph->nodes[(size_t) i];
        if (node->type == TYPE_I32 || node->type == TYPE_I64) {
            continue;
        }
        bool need = has_flag(node, TENSOR_FLAG_PARAM) || has_flag(node, TENSOR_FLAG_LOSS);
        if (!need) {
            for (int32_t j = 0; j < node->nsrc && !need; ++j) {
                Tensor* s = node->src[j];
                if (s == nullptr || ignore_src(node, j)) {
                    continue;
                }
                need = needed.count(s) != 0;
            }
        }
        if (need) {
            needed.insert(node);
        }
    }

    // 1.5) 多图共享守卫（R4）：反向通过 t->grad / grad_acc 单指针写回，任何将参与本图
    //      反向的张量（PARAM / LOSS / 梯度路径上的中间量）不得已在其他图构建过反向
    for (int64_t i = 0; i < n_fwd; ++i) {
        Tensor* node = graph->nodes[(size_t) i];
        if (!has_flag(node, TENSOR_FLAG_PARAM) && !has_flag(node, TENSOR_FLAG_LOSS) &&
            needed.count(node) == 0) {
            continue;
        }
        if (node->backward_graph != nullptr && node->backward_graph != graph) {
            TRC_ABORT("检测到多图共享张量 %s（op=%s）：禁止多图共享参数/梯度路径中间量"
                      "（梯度会静默串扰，R4）；双图训练请使用冻结/解冻 + detach 输入"
                      "（见 docs/开发进度-二期.md §5.1 R4）",
                      node->name[0] != '\0' ? node->name : "<未命名>", op_name(node->op));
        }
        node->backward_graph = graph;
    }

    // 2) 为 PARAM / LOSS 创建梯度累加器
    for (int64_t i = 0; i < n_fwd; ++i) {
        Tensor* node = graph->nodes[(size_t) i];
        if (!has_flag(node, TENSOR_FLAG_PARAM) && !has_flag(node, TENSOR_FLAG_LOSS)) {
            continue;
        }
        if (node->grad_acc == nullptr) {
            node->grad_acc = new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, node->ne);
            tensor_set_name(node->grad_acc, "grad_acc for %s", node->name);
        }
        node->grad = node->grad_acc;
    }

    // 3) 逆拓扑遍历构建反向图
    for (int64_t i = n_fwd - 1; i >= 0; --i) {
        backward_op(graph, graph->nodes[(size_t) i], needed);
    }

    graph->backward_built = true;
}

Tensor* graph_get_grad(const Graph* graph, const Tensor* t) {
    (void) graph;
    return t == nullptr ? nullptr : t->grad;
}

Tensor* graph_get_grad_acc(const Graph* graph, const Tensor* t) {
    (void) graph;
    return t == nullptr ? nullptr : t->grad_acc;
}

void graph_reset(Graph* graph) {
    TRC_ASSERT(graph != nullptr, "graph_reset: graph 为空");
    // M4.2：设备张量的梯度清零走设备 fill（单次提交），不再主机逐元素扫描
    std::vector<TensorFill> fills;
    fills.reserve(graph->nodes.size());
    for (Tensor* node : graph->nodes) {
        if (node->grad_acc == nullptr) {
            continue;
        }
        // 损失的初始梯度为 1，其余为 0（与 ggml_graph_reset 一致）
        fills.push_back({node->grad_acc, has_flag(node, TENSOR_FLAG_LOSS) ? 1.0f : 0.0f});
    }
    if (!fills.empty()) {
        tensor_fill_values(fills.data(), fills.size());
    }
}

void graph_reset_accumulate(Graph* graph) {
    TRC_ASSERT(graph != nullptr, "graph_reset_accumulate: graph 为空");
    std::vector<TensorFill> fills;
    fills.reserve(graph->nodes.size());
    for (Tensor* node : graph->nodes) {
        if (node->grad_acc != nullptr && has_flag(node, TENSOR_FLAG_LOSS)) {
            // 只播种损失的初始梯度，保留已累积的参数梯度（micro-batch 累积）
            fills.push_back({node->grad_acc, 1.0f});
        }
    }
    if (!fills.empty()) {
        tensor_fill_values(fills.data(), fills.size());
    }
}

} // namespace traincpp
