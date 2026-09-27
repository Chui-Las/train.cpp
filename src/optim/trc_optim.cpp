// train.cpp - 优化器与学习率调度实现（主机端，后端无关）
//
// 数值语义：与 PyTorch torch.optim.SGD / torch.optim.AdamW 逐条对齐（见 trc_optim.h）；
// 状态张量为 F32、与参数同形状，由优化器在首次 step 时分配（优先参数所在后端的
// BufferType，从而 CPU/Vulkan 通用；参数尚未分配时退回主机内存）。
#include "traincpp/trc_optim.h"

#include "core/trc_impl.h"
#include "traincpp/trc_backend.h"
#include "traincpp/trc_context.h"

#include <cmath>
#include <cstring>
#include <new>
#include <vector>

namespace traincpp {

namespace {

enum class OptimKind {
    SGD,
    ADAMW,
};

struct OptimState {
    Tensor* param = nullptr;
    Tensor* mom   = nullptr;  // SGD momentum / AdamW 一阶矩 m
    Tensor* vel   = nullptr;  // AdamW 二阶矩 v
    int64_t step  = 0;        // 本参数已更新次数（AdamW bias correction；SGD 首步判断）
};

struct ParamGroup {
    OptimKind    kind = OptimKind::SGD;
    SgdOptions   sgd{};
    AdamwOptions adamw{};
    std::vector<OptimState> states;
};

// R13：直接访问主机数据指针（CPU/Vulkan 的 t->data 都是主机指针，M1.4 语义），
// 不再为每个参数每次 step 分配全尺寸临时 vector。
float* data_mut(Tensor* t) {
    TRC_ASSERT(t->data != nullptr, "优化器：张量 %s 无数据（状态未分配？）", t->name);
    return tensor_data_f32(t);
}

const float* data_const(const Tensor* t) {
    TRC_ASSERT(t->data != nullptr, "优化器：张量 %s 无数据（状态未分配？）", t->name);
    return tensor_data_f32(t);
}

// 主机 scratch 扩容（仅 SGD 的 weight_decay/Nesterov 需要中间量）：
// bad_alloc 时给出"需要 X 字节"的可读报错（M2.2a-2）
void resize_or_abort(std::vector<float>& v, size_t n, const char* what) {
    try {
        v.resize(n);
    } catch (const std::bad_alloc&) {
        TRC_ABORT("主机内存不足：%s 需要约 %zu 字节（%.2f MiB）；请减小模型或换设备", what,
                  n * sizeof(float), (double) n * sizeof(float) / (1024.0 * 1024.0));
    }
}

void fill_f32(Tensor* t, float value) {
    const size_t n = (size_t) tensor_nelements(t);
    float*       p = data_mut(t);
    std::fill(p, p + n, value);
}

// 参数梯度（反向展开后 t->grad 指向梯度累加器）
const Tensor* param_grad(const Tensor* p) {
    return p->grad != nullptr ? p->grad : p->grad_acc;
}

} // namespace

// ---------------------------------------------------------------- 句柄定义

struct Optimizer {
    Context* ctx = nullptr;

    std::vector<ParamGroup> groups;
    std::vector<Tensor*>    state_tensors;  // 组→参数 顺序展开（序列化用）

    std::vector<Buffer*> state_buffers;  // 设备状态块（参数已由后端分配时；动态加组可多块）
    std::vector<void*>   host_blocks;    // 主机状态块（参数尚未分配时；动态加组可多块）

    // R13：热路径可复用主机 scratch（按最大参数长度增长一次，之后不再分配）
    std::vector<float> scratch_grad;  // SGD weight_decay 临时梯度
    std::vector<float> scratch_dir;   // SGD Nesterov 临时方向

    int64_t step_count = 0;
};

namespace {

Tensor* make_state_tensor(Optimizer* opt, const Tensor* p, const char* suffix) {
    Tensor* t = new_tensor_nd(opt->ctx, TYPE_F32, MAX_DIMS, p->ne);
    const char* base = p->name[0] != '\0' ? p->name : "param";
    tensor_set_name(t, "%s.%s", base, suffix);
    return t;
}

// 状态块使用的 BufferType：取第一个已由后端分配的参数；都未分配返回 nullptr（主机内存）
BufferType* state_buffer_type(const Optimizer* opt) {
    for (const ParamGroup& g : opt->groups) {
        for (const OptimState& st : g.states) {
            if (st.param->buffer != nullptr) {
                return st.param->buffer->buffer_type();
            }
        }
    }
    return nullptr;
}

// 为 [from, state_tensors.size()) 的状态张量分配一块状态存储并绑定 data/buffer
// （R11：可在动态加组时多次调用，每次一块独立缓冲）
void alloc_states_block(Optimizer* opt, size_t from) {
    TRC_ASSERT(from < opt->state_tensors.size(), "状态分配：起始下标越界");

    size_t total = 0;
    for (size_t i = from; i < opt->state_tensors.size(); ++i) {
        total = align_up(total, MEM_ALIGN) + tensor_nbytes(opt->state_tensors[i]);
    }

    BufferType* buft = state_buffer_type(opt);
    Buffer*     buf  = nullptr;
    uint8_t*    base = nullptr;
    if (buft != nullptr) {
        buf = buft->alloc_buffer(total);
        if (buf == nullptr) {
            // M2.2a-2：失败信息统一为"需要 X / 可用或上限 Y"
            const DeviceProps& props = buft->device()->props();
            if (props.memory_total != 0 || props.memory_free != 0) {
                TRC_ABORT("优化器状态分配失败：需要 %zu 字节（%.2f MiB），设备 '%s' 可用 %zu 字节"
                          "（%.2f MiB）/ 总 %zu 字节（%.2f MiB）；请减小模型或换设备",
                          total, (double) total / (1024.0 * 1024.0), props.name.c_str(),
                          props.memory_free, (double) props.memory_free / (1024.0 * 1024.0),
                          props.memory_total, (double) props.memory_total / (1024.0 * 1024.0));
            }
            TRC_ABORT("优化器状态分配失败：需要 %zu 字节（%.2f MiB）（设备 '%s' 未报告内存量）；"
                      "请减小模型或换设备",
                      total, (double) total / (1024.0 * 1024.0), props.name.c_str());
        }
        base = (uint8_t*) buf->base();
        TRC_ASSERT(base != nullptr, "优化器状态缓冲区无主机指针（M1.4 语义要求 host-visible）");
        std::memset(base, 0, total);
        opt->state_buffers.push_back(buf);
        backend_note_buffer_alloc(buf, buf->size());  // M2.2a：核心分配统计（optim_free 经 buffer_free 扣减）
    } else {
        // 参数尚未由 buffer 分配：状态先用主机内存，随后 buffer_alloc_ctx_tensors
        // 会跳过已有数据的张量（见 trc_backend_reg.cpp）
        base = (uint8_t*) alloc_aligned(total, MEM_ALIGN);
        TRC_ASSERT(base != nullptr,
                   "优化器状态主机内存分配失败：需要 %zu 字节（%.2f MiB）；请减小模型或检查可用物理内存",
                   total, (double) total / (1024.0 * 1024.0));
        std::memset(base, 0, total);
        opt->host_blocks.push_back(base);
    }

    size_t offset = 0;
    for (size_t i = from; i < opt->state_tensors.size(); ++i) {
        Tensor* t = opt->state_tensors[i];
        offset    = align_up(offset, MEM_ALIGN);
        t->buffer = buf;  // 主机内存情形保持 nullptr
        t->data   = base + offset;
        offset += tensor_nbytes(t);
    }
}

void add_group(Optimizer* opt, OptimKind kind, const SgdOptions& sgd, const AdamwOptions& adamw,
               Tensor* const* params, int64_t n_params) {
    TRC_ASSERT(opt != nullptr, "优化器为空");
    TRC_ASSERT(params != nullptr && n_params > 0, "优化器参数列表为空");

    const size_t first_new_state = opt->state_tensors.size();

    ParamGroup group;
    group.kind  = kind;
    group.sgd   = sgd;
    group.adamw = adamw;

    for (int64_t i = 0; i < n_params; ++i) {
        Tensor* p = params[i];
        TRC_ASSERT(p != nullptr, "优化器参数[%lld]为空", (long long) i);
        TRC_ASSERT(p->type == TYPE_F32, "优化器参数 %s 必须是 F32（当前 %s）", p->name,
                   type_name(p->type));
        TRC_ASSERT(tensor_is_contiguous(p), "优化器参数 %s 必须连续", p->name);
        TRC_ASSERT(!tensor_is_view(p), "优化器参数 %s 不能是视图", p->name);

        OptimState st;
        st.param = p;
        if (kind == OptimKind::SGD) {
            st.mom = make_state_tensor(opt, p, "momentum");
            opt->state_tensors.push_back(st.mom);
        } else {
            st.mom = make_state_tensor(opt, p, "m");
            st.vel = make_state_tensor(opt, p, "v");
            opt->state_tensors.push_back(st.mom);
            opt->state_tensors.push_back(st.vel);
        }
        group.states.push_back(st);
    }

    opt->groups.push_back(std::move(group));

    // 动态加组（M2.1h / R11）：若已有状态张量已分配数据（已 step / 已 alloc / 已被 ctx 分配），
    // 立即为新组分配独立状态块；否则留给首次 step 的 ensure_states 统一分配。
    if (first_new_state > 0 && opt->state_tensors[first_new_state - 1]->data != nullptr) {
        alloc_states_block(opt, first_new_state);
    }
}

// 首次 step 时分配全部状态（优先使用参数所在后端的 BufferType，CPU/Vulkan 通用）
void ensure_states(Optimizer* opt) {
    if (opt->state_tensors.empty() || opt->state_tensors[0]->data != nullptr) {
        return;
    }
    alloc_states_block(opt, 0);
}

void sgd_update(ParamGroup& group, Optimizer* opt) {
    const SgdOptions& o = group.sgd;

    for (OptimState& st : group.states) {
        const Tensor* gt = param_grad(st.param);
        TRC_ASSERT(gt != nullptr,
                   "优化器：参数 %s 没有梯度（需先 graph_build_backward_expand + 计算）",
                   st.param->name);

        const size_t n    = (size_t) tensor_nelements(st.param);
        float*       p    = data_mut(st.param);
        const float* grad = data_const(gt);

        // d_p = g + weight_decay·p（只读 grad，不修改梯度本身）
        const float* dir = grad;
        if (o.weight_decay != 0.0f) {
            resize_or_abort(opt->scratch_grad, n, "SGD weight_decay 临时量");
            float* work = opt->scratch_grad.data();
            for (size_t i = 0; i < n; ++i) {
                work[i] = grad[i] + o.weight_decay * p[i];
            }
            dir = work;
        }

        if (o.momentum != 0.0f) {
            float* m = data_mut(st.mom);
            if (st.step == 0) {
                // PyTorch：首个动量步直接克隆 d_p（不应用 dampening）
                for (size_t i = 0; i < n; ++i) {
                    m[i] = dir[i];
                }
            } else {
                const float alpha = 1.0f - o.dampening;
                for (size_t i = 0; i < n; ++i) {
                    m[i] = o.momentum * m[i] + alpha * dir[i];
                }
            }
            if (o.nesterov) {
                resize_or_abort(opt->scratch_dir, n, "SGD Nesterov 临时量");
                float* nd = opt->scratch_dir.data();
                for (size_t i = 0; i < n; ++i) {
                    nd[i] = dir[i] + o.momentum * m[i];
                }
                dir = nd;
            } else {
                dir = m;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            p[i] -= o.lr * dir[i];
        }
        ++st.step;
    }
}

void adamw_update(ParamGroup& group, Optimizer* opt) {
    (void) opt;
    const AdamwOptions& o = group.adamw;

    for (OptimState& st : group.states) {
        const Tensor* gt = param_grad(st.param);
        TRC_ASSERT(gt != nullptr,
                   "优化器：参数 %s 没有梯度（需先 graph_build_backward_expand + 计算）",
                   st.param->name);

        ++st.step;
        const double bc1      = 1.0 - std::pow((double) o.beta1, (double) st.step);
        const double bc2      = 1.0 - std::pow((double) o.beta2, (double) st.step);
        const float  step_sz  = (float) ((double) o.lr / bc1);
        const float  bc2_sqrt = (float) std::sqrt(bc2);
        const float  decay    = 1.0f - o.lr * o.weight_decay;

        // R13：原地读写参数与 m/v 状态（逐元素运算顺序不变，数值逐位一致）
        const size_t n = (size_t) tensor_nelements(st.param);
        float*       p = data_mut(st.param);
        const float* g = data_const(gt);
        float*       m = data_mut(st.mom);
        float*       v = data_mut(st.vel);

        for (size_t i = 0; i < n; ++i) {
            const float gi = g[i];
            // PyTorch：exp_avg.lerp_(grad, 1-beta1)；exp_avg_sq.mul_(beta2).addcmul_(g,g,1-beta2)
            m[i] += (gi - m[i]) * (1.0f - o.beta1);
            v[i] = v[i] * o.beta2 + gi * gi * (1.0f - o.beta2);

            p[i] *= decay;

            const float denom = std::sqrt(v[i]) / bc2_sqrt + o.eps;
            p[i] -= step_sz * (m[i] / denom);
        }
    }
}

} // namespace

enum class LrScheduleKind {
    STEP,
    EXPONENTIAL,
    COSINE,
    WARMUP_COSINE,
};

struct LrScheduler {
    Optimizer*      opt = nullptr;
    LrScheduleKind  kind = LrScheduleKind::STEP;
    float           base_lr = 0.0f;
    float           lr      = 0.0f;
    int64_t         epoch   = 0;

    int64_t step_size = 1;
    float   gamma     = 1.0f;

    int64_t t_max   = 1;
    float   eta_min = 0.0f;

    int64_t warmup_steps = 0;
    int64_t total_steps  = 0;
};

// ---------------------------------------------------------------- 公共 API

Optimizer* optim_sgd_new(Context* ctx, Tensor* const* params, int64_t n_params,
                         const SgdOptions& opts) {
    TRC_ASSERT(ctx != nullptr, "optim_sgd_new: ctx 为空");
    Optimizer* opt = new Optimizer();
    opt->ctx = ctx;
    add_group(opt, OptimKind::SGD, opts, AdamwOptions{}, params, n_params);
    return opt;
}

Optimizer* optim_adamw_new(Context* ctx, Tensor* const* params, int64_t n_params,
                           const AdamwOptions& opts) {
    TRC_ASSERT(ctx != nullptr, "optim_adamw_new: ctx 为空");
    Optimizer* opt = new Optimizer();
    opt->ctx = ctx;
    add_group(opt, OptimKind::ADAMW, SgdOptions{}, opts, params, n_params);
    return opt;
}

void optim_free(Optimizer* opt) {
    if (opt == nullptr) {
        return;
    }
    // 状态张量存活于 ctx：清空其数据指针，避免 free 后悬挂
    for (Tensor* t : opt->state_tensors) {
        t->buffer = nullptr;
        t->data   = nullptr;
    }
    for (Buffer* b : opt->state_buffers) {
        buffer_free(b);
    }
    for (void* p : opt->host_blocks) {
        free_aligned(p);
    }
    delete opt;
}

void optim_add_param_group_sgd(Optimizer* opt, Tensor* const* params, int64_t n_params,
                               const SgdOptions& opts) {
    add_group(opt, OptimKind::SGD, opts, AdamwOptions{}, params, n_params);
}

void optim_add_param_group_adamw(Optimizer* opt, Tensor* const* params, int64_t n_params,
                                 const AdamwOptions& opts) {
    add_group(opt, OptimKind::ADAMW, SgdOptions{}, opts, params, n_params);
}

int64_t optim_group_count(const Optimizer* opt) {
    return (int64_t) opt->groups.size();
}

int64_t optim_param_count(const Optimizer* opt) {
    int64_t n = 0;
    for (const ParamGroup& g : opt->groups) {
        n += (int64_t) g.states.size();
    }
    return n;
}

Tensor* optim_param(const Optimizer* opt, int64_t index) {
    TRC_ASSERT(index >= 0 && index < optim_param_count(opt), "optim_param: 下标越界 %lld",
               (long long) index);
    int64_t i = index;
    for (const ParamGroup& g : opt->groups) {
        if (i < (int64_t) g.states.size()) {
            return g.states[(size_t) i].param;
        }
        i -= (int64_t) g.states.size();
    }
    return nullptr;
}

int64_t optim_state_count(const Optimizer* opt) {
    return (int64_t) opt->state_tensors.size();
}

Tensor* optim_state(const Optimizer* opt, int64_t index) {
    TRC_ASSERT(index >= 0 && index < optim_state_count(opt), "optim_state: 下标越界 %lld",
               (long long) index);
    return opt->state_tensors[(size_t) index];
}

float optim_get_lr(const Optimizer* opt) {
    TRC_ASSERT(!opt->groups.empty(), "optim_get_lr: 优化器没有参数组");
    const ParamGroup& g = opt->groups[0];
    return g.kind == OptimKind::SGD ? g.sgd.lr : g.adamw.lr;
}

OptimType optim_group_type(const Optimizer* opt, int64_t group) {
    TRC_ASSERT(opt != nullptr, "optim_group_type: 优化器为空");
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(), "optim_group_type: 组下标越界 %lld",
               (long long) group);
    return opt->groups[(size_t) group].kind == OptimKind::SGD ? OptimType::SGD : OptimType::ADAMW;
}

int64_t optim_group_param_count(const Optimizer* opt, int64_t group) {
    TRC_ASSERT(opt != nullptr, "optim_group_param_count: 优化器为空");
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(),
               "optim_group_param_count: 组下标越界 %lld", (long long) group);
    return (int64_t) opt->groups[(size_t) group].states.size();
}

int64_t optim_group_state_count(const Optimizer* opt, int64_t group) {
    TRC_ASSERT(opt != nullptr, "optim_group_state_count: 优化器为空");
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(),
               "optim_group_state_count: 组下标越界 %lld", (long long) group);
    const ParamGroup& g = opt->groups[(size_t) group];
    return (int64_t) g.states.size() * (g.kind == OptimKind::SGD ? 1 : 2);
}

float optim_get_lr_group(const Optimizer* opt, int64_t group) {
    TRC_ASSERT(opt != nullptr, "optim_get_lr_group: 优化器为空");
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(), "optim_get_lr_group: 组下标越界 %lld",
               (long long) group);
    const ParamGroup& g = opt->groups[(size_t) group];
    return g.kind == OptimKind::SGD ? g.sgd.lr : g.adamw.lr;
}

SgdOptions optim_group_sgd_options(const Optimizer* opt, int64_t group) {
    TRC_ASSERT(opt != nullptr, "optim_group_sgd_options: 优化器为空");
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(),
               "optim_group_sgd_options: 组下标越界 %lld", (long long) group);
    const ParamGroup& g = opt->groups[(size_t) group];
    TRC_ASSERT(g.kind == OptimKind::SGD, "optim_group_sgd_options: 第 %lld 组不是 SGD", (long long) group);
    return g.sgd;
}

AdamwOptions optim_group_adamw_options(const Optimizer* opt, int64_t group) {
    TRC_ASSERT(opt != nullptr, "optim_group_adamw_options: 优化器为空");
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(),
               "optim_group_adamw_options: 组下标越界 %lld", (long long) group);
    const ParamGroup& g = opt->groups[(size_t) group];
    TRC_ASSERT(g.kind == OptimKind::ADAMW, "optim_group_adamw_options: 第 %lld 组不是 AdamW",
               (long long) group);
    return g.adamw;
}

void optim_alloc_state(Optimizer* opt) {
    TRC_ASSERT(opt != nullptr, "optim_alloc_state: 优化器为空");
    ensure_states(opt);
}

void optim_set_step_count(Optimizer* opt, int64_t step) {
    TRC_ASSERT(opt != nullptr, "optim_set_step_count: 优化器为空");
    TRC_ASSERT(step >= 0, "optim_set_step_count: step 不能为负 %lld", (long long) step);
    opt->step_count = step;
}

int64_t optim_state_step(const Optimizer* opt, int64_t param_index) {
    TRC_ASSERT(opt != nullptr, "optim_state_step: 优化器为空");
    TRC_ASSERT(param_index >= 0 && param_index < optim_param_count(opt),
               "optim_state_step: 参数下标越界 %lld", (long long) param_index);
    int64_t i = param_index;
    for (const ParamGroup& g : opt->groups) {
        if (i < (int64_t) g.states.size()) {
            return g.states[(size_t) i].step;
        }
        i -= (int64_t) g.states.size();
    }
    return 0;
}

void optim_set_state_step(Optimizer* opt, int64_t param_index, int64_t step) {
    TRC_ASSERT(opt != nullptr, "optim_set_state_step: 优化器为空");
    TRC_ASSERT(param_index >= 0 && param_index < optim_param_count(opt),
               "optim_set_state_step: 参数下标越界 %lld", (long long) param_index);
    TRC_ASSERT(step >= 0, "optim_set_state_step: step 不能为负 %lld", (long long) step);
    int64_t i = param_index;
    for (ParamGroup& g : opt->groups) {
        if (i < (int64_t) g.states.size()) {
            g.states[(size_t) i].step = step;
            return;
        }
        i -= (int64_t) g.states.size();
    }
}

void optim_set_lr(Optimizer* opt, float lr) {
    for (ParamGroup& g : opt->groups) {
        if (g.kind == OptimKind::SGD) {
            g.sgd.lr = lr;
        } else {
            g.adamw.lr = lr;
        }
    }
}

void optim_set_lr_group(Optimizer* opt, int64_t group, float lr) {
    TRC_ASSERT(group >= 0 && group < (int64_t) opt->groups.size(), "optim_set_lr_group: 组下标越界");
    ParamGroup& g = opt->groups[(size_t) group];
    if (g.kind == OptimKind::SGD) {
        g.sgd.lr = lr;
    } else {
        g.adamw.lr = lr;
    }
}

int64_t optim_step_count(const Optimizer* opt) {
    return opt->step_count;
}

void optim_zero_grad(Optimizer* opt) {
    for (const ParamGroup& g : opt->groups) {
        for (const OptimState& st : g.states) {
            if (st.param->grad_acc != nullptr) {
                fill_f32(st.param->grad_acc, 0.0f);
            } else if (st.param->grad != nullptr) {
                fill_f32(st.param->grad, 0.0f);
            }
        }
    }
}

void optim_step(Optimizer* opt) {
    TRC_ASSERT(opt != nullptr, "optim_step: 优化器为空");
    ensure_states(opt);
    for (ParamGroup& g : opt->groups) {
        if (g.kind == OptimKind::SGD) {
            sgd_update(g, opt);
        } else {
            adamw_update(g, opt);
        }
    }
    ++opt->step_count;
}

float optim_clip_grad_norm(Optimizer* opt, float max_norm) {
    TRC_ASSERT(opt != nullptr, "optim_clip_grad_norm: 优化器为空");

    // R13：直接读/写主机数据（逐元素运算顺序不变）
    double sum_sq = 0.0;
    for (const ParamGroup& g : opt->groups) {
        for (const OptimState& st : g.states) {
            const Tensor* gt = param_grad(st.param);
            if (gt == nullptr) {
                continue;
            }
            const size_t n = (size_t) tensor_nelements(gt);
            const float* p = data_const(gt);
            for (size_t i = 0; i < n; ++i) {
                sum_sq += (double) p[i] * (double) p[i];
            }
        }
    }

    const float total_norm = (float) std::sqrt(sum_sq);
    if (total_norm > max_norm) {
        const float scale = max_norm / (total_norm + 1e-6f);
        for (ParamGroup& g : opt->groups) {
            for (const OptimState& st : g.states) {
                const Tensor* gt = param_grad(st.param);
                if (gt == nullptr) {
                    continue;
                }
                const size_t n = (size_t) tensor_nelements(gt);
                float*       p = data_mut(const_cast<Tensor*>(gt));  // 梯度可能来自梯度累加器（可写）
                for (size_t i = 0; i < n; ++i) {
                    p[i] *= scale;
                }
            }
        }
    }
    return total_norm;
}

// ---------------------------------------------------------------- 调度器

namespace {

float compute_lr(const LrScheduler* s) {
    const double base = (double) s->base_lr;
    const double emin = (double) s->eta_min;
    switch (s->kind) {
        case LrScheduleKind::STEP: {
            const int64_t k = s->step_size > 0 ? s->epoch / s->step_size : 0;
            return (float) (base * std::pow((double) s->gamma, (double) k));
        }
        case LrScheduleKind::EXPONENTIAL:
            return (float) (base * std::pow((double) s->gamma, (double) s->epoch));
        case LrScheduleKind::COSINE: {
            // 与 PyTorch CosineAnnealingLR 一致：不把 t 截断在 T_max（t > T_max 后 lr 会回升）
            const double t = s->t_max > 0 ? (double) s->epoch / (double) s->t_max : 1.0;
            return (float) (emin + (base - emin) * (1.0 + std::cos(3.14159265358979323846 * t)) / 2.0);
        }
        case LrScheduleKind::WARMUP_COSINE: {
            if (s->epoch <= s->warmup_steps) {
                if (s->warmup_steps <= 0) {
                    return s->base_lr;
                }
                return (float) (base * (double) s->epoch / (double) s->warmup_steps);
            }
            const int64_t remain = s->total_steps - s->warmup_steps;
            if (remain <= 0) {
                return s->base_lr;
            }
            const double t = (double) (s->epoch - s->warmup_steps) / (double) remain;
            const double tc = t > 1.0 ? 1.0 : t;
            return (float) (emin + (base - emin) * (1.0 + std::cos(3.14159265358979323846 * tc)) / 2.0);
        }
    }
    return s->base_lr;
}

LrScheduler* scheduler_new(Optimizer* opt, LrScheduleKind kind) {
    TRC_ASSERT(opt != nullptr, "学习率调度：优化器为空");
    LrScheduler* s = new LrScheduler();
    s->opt     = opt;
    s->kind    = kind;
    s->base_lr = optim_get_lr(opt);
    s->lr      = s->base_lr;
    return s;
}

} // namespace

LrScheduler* lr_scheduler_step_new(Optimizer* opt, int64_t step_size, float gamma) {
    LrScheduler* s = scheduler_new(opt, LrScheduleKind::STEP);
    s->step_size = step_size;
    s->gamma     = gamma;
    return s;
}

LrScheduler* lr_scheduler_exponential_new(Optimizer* opt, float gamma) {
    LrScheduler* s = scheduler_new(opt, LrScheduleKind::EXPONENTIAL);
    s->gamma = gamma;
    return s;
}

LrScheduler* lr_scheduler_cosine_new(Optimizer* opt, int64_t t_max, float eta_min) {
    LrScheduler* s = scheduler_new(opt, LrScheduleKind::COSINE);
    s->t_max   = t_max;
    s->eta_min = eta_min;
    return s;
}

LrScheduler* lr_scheduler_warmup_cosine_new(Optimizer* opt, int64_t warmup_steps, int64_t total_steps,
                                            float eta_min) {
    LrScheduler* s = scheduler_new(opt, LrScheduleKind::WARMUP_COSINE);
    s->warmup_steps = warmup_steps;
    s->total_steps  = total_steps;
    s->eta_min      = eta_min;
    return s;
}

void lr_scheduler_free(LrScheduler* sched) {
    delete sched;
}

float lr_scheduler_step(LrScheduler* sched) {
    TRC_ASSERT(sched != nullptr, "lr_scheduler_step: 调度器为空");
    ++sched->epoch;
    sched->lr = compute_lr(sched);
    optim_set_lr(sched->opt, sched->lr);
    return sched->lr;
}

float lr_scheduler_get_lr(const LrScheduler* sched) {
    TRC_ASSERT(sched != nullptr, "lr_scheduler_get_lr: 调度器为空");
    return sched->lr;
}

int64_t lr_scheduler_epoch(const LrScheduler* sched) {
    TRC_ASSERT(sched != nullptr, "lr_scheduler_epoch: 调度器为空");
    return sched->epoch;
}

} // namespace traincpp
