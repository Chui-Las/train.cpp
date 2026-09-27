// train.cpp - 训练检查点实现（M1.6b；M2.3f schema v2 多优化器，兼容读 v1；复用 GGUF 容器，见 trc_gguf.h）
//
// 文件结构：GGUF v3，general.architecture = "traincpp.checkpoint"
//   写出恒为 schema v2：
//     traincpp.checkpoint.version = 2 / traincpp.model.arch（可选）
//     traincpp.optimizer.count = N
//     traincpp.optimizer.{i}.name / .type / .groups / .param_count / .step / .has_state
//     traincpp.optimizer.{i}.param_names / .state_names / .state_steps
//     traincpp.optimizer.{i}.group{j}.{lr,+超参}
//     traincpp.optimizer.{i}.scheduler.epoch（可选，每分区独立）
//     traincpp.rng.state（可选；多优化器共享一份）
//   读兼容 v1（单优化器旧键）：
//     traincpp.optimizer.{type,groups,param_count,step,has_state,param_names,state_names,state_steps}
//     traincpp.optimizer.group{i}.{lr,+超参} / traincpp.scheduler.epoch
//   张量：参数（名字 = Tensor::name）+ 状态（<param>.momentum 或 <param>.m/.v）
#define _CRT_SECURE_NO_WARNINGS

#include "traincpp/trc_checkpoint.h"

#include "core/trc_impl.h"
#include "traincpp/trc_gguf.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace traincpp {

namespace {

constexpr const char* CKPT_ARCH           = "traincpp.checkpoint";
constexpr uint32_t    CKPT_SCHEMA_V1      = 1;
constexpr uint32_t    CKPT_SCHEMA_V2      = 2;
constexpr uint32_t    CKPT_SCHEMA_VERSION = CKPT_SCHEMA_V2;  // 写出恒为 v2

const char* optim_type_str(OptimType t) {
    return t == OptimType::SGD ? "sgd" : "adamw";
}

// ---------------- 优化器分区（v2：名字 + 优化器 + 可选调度器） ----------------

struct Partition {
    std::string  name;
    bool         named = false;  // true：来自 entries（v2 按名匹配）；false：单优化器快捷字段
    Optimizer*   opt   = nullptr;
    LrScheduler* sched = nullptr;
};

// v2 分区键：traincpp.optimizer.{i}.{field}
std::string v2_key(int64_t i, const char* field) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "traincpp.optimizer.%lld.%s", (long long) i, field);
    return buf;
}

// 分区内键方案：v1 旧键（traincpp.optimizer.{field} / .group{g}.{field}）或 v2 分区键
struct KeyScheme {
    bool    v1  = false;
    int64_t idx = 0;

    std::string k(const char* field) const {
        char buf[192];
        if (v1) {
            std::snprintf(buf, sizeof(buf), "traincpp.optimizer.%s", field);
        } else {
            std::snprintf(buf, sizeof(buf), "traincpp.optimizer.%lld.%s", (long long) idx, field);
        }
        return buf;
    }

    std::string g(int64_t group, const char* field) const {
        char buf[224];
        if (v1) {
            std::snprintf(buf, sizeof(buf), "traincpp.optimizer.group%lld.%s", (long long) group, field);
        } else {
            std::snprintf(buf, sizeof(buf), "traincpp.optimizer.%lld.group%lld.%s",
                          (long long) idx, (long long) group, field);
        }
        return buf;
    }

    // 调度器 epoch：v1 在顶层（traincpp.scheduler.epoch），v2 在分区内
    std::string s() const {
        char buf[192];
        if (v1) {
            std::snprintf(buf, sizeof(buf), "traincpp.scheduler.epoch");
        } else {
            std::snprintf(buf, sizeof(buf), "traincpp.optimizer.%lld.scheduler.epoch",
                          (long long) idx);
        }
        return buf;
    }
};

using FailFn = std::function<bool(const char*)>;

// 从 CheckpointItems 构建请求分区列表：entries 优先；否则单优化器快捷字段（名字 "optimizer"）
bool collect_partitions(const CheckpointItems& items, std::vector<Partition>* out, std::string* err) {
    out->clear();
    const bool multi = items.optimizers != nullptr || items.optimizer_count != 0;
    if (multi) {
        if (items.optimizers == nullptr || items.optimizer_count <= 0) {
            *err = "optimizers 与 optimizer_count 不一致";
            return false;
        }
        for (int64_t i = 0; i < items.optimizer_count; ++i) {
            const CheckpointOptimizer& e = items.optimizers[(size_t) i];
            if (e.optimizer == nullptr) {
                *err = "第 " + std::to_string((long long) i) + " 个分区没有 optimizer";
                return false;
            }
            if (e.name == nullptr || *e.name == '\0') {
                *err = "第 " + std::to_string((long long) i) + " 个分区名字为空";
                return false;
            }
            for (const Partition& p : *out) {
                if (p.name == e.name) {
                    *err = std::string("分区名重复：") + e.name;
                    return false;
                }
            }
            out->push_back(Partition{e.name, true, e.optimizer, e.scheduler});
        }
        return true;
    }
    if (items.optimizer == nullptr) {
        *err = "缺少优化器（optimizer 或 optimizers 必填）";
        return false;
    }
    out->push_back(Partition{"optimizer", false, items.optimizer, items.scheduler});
    return true;
}

// 校验字符串数组与给定张量名一一对应（顺序敏感）
bool names_match(const GgufFile* f, const char* key, const std::vector<Tensor*>& tensors) {
    const int64_t idx = gguf_find_key(f, key);
    if (idx < 0 || gguf_arr_type(f, idx) != GgufType::STRING ||
        gguf_arr_count(f, idx) != (int64_t) tensors.size()) {
        TRC_LOG_ERROR("checkpoint: 元数据 %s 缺失或数量不符", key);
        return false;
    }
    for (int64_t i = 0; i < (int64_t) tensors.size(); ++i) {
        const char* name = gguf_get_arr_str(f, key, i);
        if (name == nullptr || std::strcmp(name, tensors[(size_t) i]->name) != 0) {
            TRC_LOG_ERROR("checkpoint: 张量名不匹配（第 %lld 个：文件 %s，当前 %s）", (long long) i,
                          name != nullptr ? name : "(null)", tensors[(size_t) i]->name);
            return false;
        }
    }
    return true;
}

// 校验非 lr 超参与当前优化器一致（防止用错配置续训）
bool hyperparams_match(const GgufFile* f, const KeyScheme& ks, const Optimizer* opt) {
    const int64_t n_groups = optim_group_count(opt);
    for (int64_t gi = 0; gi < n_groups; ++gi) {
        if (optim_group_type(opt, gi) == OptimType::SGD) {
            const SgdOptions o = optim_group_sgd_options(opt, gi);
            const float momentum     = gguf_get_f32(f, ks.g(gi, "momentum").c_str());
            const float dampening    = gguf_get_f32(f, ks.g(gi, "dampening").c_str());
            const float weight_decay = gguf_get_f32(f, ks.g(gi, "weight_decay").c_str());
            const bool  nesterov     = gguf_get_bool(f, ks.g(gi, "nesterov").c_str());
            if (momentum != o.momentum || dampening != o.dampening ||
                weight_decay != o.weight_decay || nesterov != o.nesterov) {
                TRC_LOG_ERROR("checkpoint: SGD 第 %lld 组超参不一致（momentum/dampening/wd/nesterov）",
                              (long long) gi);
                return false;
            }
        } else {
            const AdamwOptions o = optim_group_adamw_options(opt, gi);
            const float beta1        = gguf_get_f32(f, ks.g(gi, "beta1").c_str());
            const float beta2        = gguf_get_f32(f, ks.g(gi, "beta2").c_str());
            const float eps          = gguf_get_f32(f, ks.g(gi, "eps").c_str());
            const float weight_decay = gguf_get_f32(f, ks.g(gi, "weight_decay").c_str());
            if (beta1 != o.beta1 || beta2 != o.beta2 || eps != o.eps ||
                weight_decay != o.weight_decay) {
                TRC_LOG_ERROR("checkpoint: AdamW 第 %lld 组超参不一致（beta1/beta2/eps/wd）",
                              (long long) gi);
                return false;
            }
        }
    }
    return true;
}

void write_group_options(GgufWriter* w, const KeyScheme& ks, const Optimizer* opt, int64_t gi) {
    if (optim_group_type(opt, gi) == OptimType::SGD) {
        const SgdOptions o = optim_group_sgd_options(opt, gi);
        gguf_writer_set_f32(w, ks.g(gi, "lr").c_str(), o.lr);
        gguf_writer_set_f32(w, ks.g(gi, "momentum").c_str(), o.momentum);
        gguf_writer_set_f32(w, ks.g(gi, "dampening").c_str(), o.dampening);
        gguf_writer_set_f32(w, ks.g(gi, "weight_decay").c_str(), o.weight_decay);
        gguf_writer_set_bool(w, ks.g(gi, "nesterov").c_str(), o.nesterov);
    } else {
        const AdamwOptions o = optim_group_adamw_options(opt, gi);
        gguf_writer_set_f32(w, ks.g(gi, "lr").c_str(), o.lr);
        gguf_writer_set_f32(w, ks.g(gi, "beta1").c_str(), o.beta1);
        gguf_writer_set_f32(w, ks.g(gi, "beta2").c_str(), o.beta2);
        gguf_writer_set_f32(w, ks.g(gi, "eps").c_str(), o.eps);
        gguf_writer_set_f32(w, ks.g(gi, "weight_decay").c_str(), o.weight_decay);
    }
}

// 张量数据逐元素比对（F32，精确相等）
bool same_tensor(const GgufFile* f, Tensor* t) {
    const int64_t n = tensor_nelements(t);
    std::vector<float> expect((size_t) n);
    tensor_get(t, expect.data(), 0, expect.size() * sizeof(float));
    std::vector<float> got((size_t) n, 0.0f);
    if (!gguf_tensor_to_f32(f, t->name, got.data(), n)) {
        return false;
    }
    for (size_t i = 0; i < expect.size(); ++i) {
        if (got[i] != expect[i]) {
            TRC_LOG_ERROR("checkpoint 自检失败：张量 %s 第 %zu 个元素不一致（%.9g vs %.9g）",
                          t->name, i, (double) got[i], (double) expect[i]);
            return false;
        }
    }
    return true;
}

// 保存期的跨分区张量名唯一校验（GGUF 要求全局唯一；重名会导致状态张量互相覆盖）
void check_global_name(std::vector<std::pair<std::string, std::string>>* seen, const char* name,
                       const char* part_name) {
    for (const auto& e : *seen) {
        if (e.first == name) {
            TRC_ABORT("checkpoint_save: 张量名跨优化器分区重复 %s（分区 %s 与 %s）", name,
                      e.second.c_str(), part_name);
        }
    }
    seen->push_back({name, part_name});
}

// 按 version 匹配请求分区与文件分区：v1 仅单分区；v2 按名字（entries 模式）或单分区（快捷字段）
bool match_partitions(const GgufFile* f, uint32_t version, const std::vector<Partition>& req,
                      std::vector<int64_t>* file_idx, std::string* err) {
    file_idx->assign(req.size(), -1);
    if (version == CKPT_SCHEMA_V1) {
        if (req.size() != 1) {
            *err = "v1 检查点仅支持单优化器";
            return false;
        }
        (*file_idx)[0] = 0;
        return true;
    }
    const int64_t n_file = (int64_t) gguf_get_u32(f, "traincpp.optimizer.count");
    if (n_file != (int64_t) req.size()) {
        *err = "优化器分区数量不匹配";
        return false;
    }
    if (!req.empty() && req[0].named) {
        for (size_t i = 0; i < req.size(); ++i) {
            int64_t found = -1;
            for (int64_t j = 0; j < n_file; ++j) {
                const std::string nk = v2_key(j, "name");
                const char*       fn = gguf_get_str(f, nk.c_str());
                if (fn != nullptr && req[i].name == fn) {
                    found = j;
                    break;
                }
            }
            if (found < 0) {
                *err = std::string("找不到优化器分区：") + req[i].name;
                return false;
            }
            (*file_idx)[i] = found;
        }
    } else {
        (*file_idx)[0] = 0;
    }
    return true;
}

// 载入一个分区（v1/v2 共用；成功返回 true，失败调用 fail 并返回 false）
bool load_partition(const GgufFile* f, const KeyScheme& ks, const Partition& part, const FailFn& fail) {
    Optimizer* opt = part.opt;

    const int64_t n_groups = (int64_t) gguf_get_u32(f, ks.k("groups").c_str());
    const int64_t n_params = (int64_t) gguf_get_u32(f, ks.k("param_count").c_str());
    if (n_groups != optim_group_count(opt) || n_params != optim_param_count(opt)) {
        return fail("优化器分组/参数数量不匹配");
    }

    const char* type_str = gguf_get_str(f, ks.k("type").c_str());
    if (type_str == nullptr) {
        return fail("缺少优化器类型");
    }
    for (int64_t gi = 0; gi < n_groups; ++gi) {
        if (std::strcmp(type_str, optim_type_str(optim_group_type(opt, gi))) != 0) {
            return fail("优化器类型不匹配");
        }
    }

    if (!hyperparams_match(f, ks, opt)) {
        return fail("优化器超参不匹配");
    }

    std::vector<Tensor*> params((size_t) n_params);
    for (int64_t i = 0; i < n_params; ++i) {
        params[(size_t) i] = optim_param(opt, i);
    }
    if (!names_match(f, ks.k("param_names").c_str(), params)) {
        return fail("参数名列表不匹配");
    }

    const bool has_state = gguf_get_bool(f, ks.k("has_state").c_str());
    std::vector<Tensor*> states;
    if (has_state) {
        const int64_t n_states = optim_state_count(opt);
        states.reserve((size_t) n_states);
        for (int64_t i = 0; i < n_states; ++i) {
            Tensor* s = optim_state(opt, i);
            if (s->data == nullptr) {
                return fail("优化器状态未分配（请先调用 optim_alloc_state）");
            }
            states.push_back(s);
        }
        if (!names_match(f, ks.k("state_names").c_str(), states)) {
            return fail("状态张量名列表不匹配");
        }
    }

    // 载入张量
    for (Tensor* p : params) {
        if (!gguf_load_tensor(f, p->name, p)) {
            return fail("参数张量载入失败");
        }
    }
    if (has_state) {
        for (Tensor* s : states) {
            if (!gguf_load_tensor(f, s->name, s)) {
                return fail("状态张量载入失败");
            }
        }
        // 每个参数的独立步数（缺失会导致 AdamW bias correction / SGD 首步判断错误）
        int64_t        n_steps = 0;
        const int64_t* steps   = gguf_get_arr_i64(f, ks.k("state_steps").c_str(), &n_steps);
        if (steps == nullptr || n_steps != n_params) {
            return fail("状态步数数组缺失或数量不符");
        }
        for (int64_t i = 0; i < n_params; ++i) {
            optim_set_state_step(opt, i, steps[i]);
        }
    }

    // step 与每组 lr
    optim_set_step_count(opt, gguf_get_i64(f, ks.k("step").c_str()));
    for (int64_t gi = 0; gi < n_groups; ++gi) {
        optim_set_lr_group(opt, gi, gguf_get_f32(f, ks.g(gi, "lr").c_str()));
    }

    // 调度器：epoch 只能前进，用 lr_scheduler_step 回放
    if (part.sched != nullptr) {
        const std::string sk = ks.s();
        if (gguf_has_key(f, sk.c_str())) {
            const int64_t target = gguf_get_i64(f, sk.c_str());
            int64_t       cur    = lr_scheduler_epoch(part.sched);
            if (cur > target) {
                return fail("调度器 epoch 大于存档，无法回退");
            }
            while (cur < target) {
                lr_scheduler_step(part.sched);
                ++cur;
            }
        }
    }
    return true;
}

// 自检一个分区（与 load_partition 同口径；成功返回 true，失败调用 fail 并返回 false）
bool verify_partition(const GgufFile* f, const KeyScheme& ks, const Partition& part, const FailFn& fail) {
    Optimizer* opt = part.opt;

    const int64_t n_groups = (int64_t) gguf_get_u32(f, ks.k("groups").c_str());
    const int64_t n_params = (int64_t) gguf_get_u32(f, ks.k("param_count").c_str());
    if (n_groups != optim_group_count(opt) || n_params != optim_param_count(opt)) {
        return fail("优化器分组/参数数量不匹配");
    }
    const char* type_str = gguf_get_str(f, ks.k("type").c_str());
    if (type_str == nullptr) {
        return fail("缺少优化器类型");
    }
    for (int64_t gi = 0; gi < n_groups; ++gi) {
        if (std::strcmp(type_str, optim_type_str(optim_group_type(opt, gi))) != 0) {
            return fail("优化器类型不匹配");
        }
    }
    if (!hyperparams_match(f, ks, opt)) {
        return fail("优化器超参不匹配");
    }

    std::vector<Tensor*> params((size_t) n_params);
    for (int64_t i = 0; i < n_params; ++i) {
        params[(size_t) i] = optim_param(opt, i);
    }
    if (!names_match(f, ks.k("param_names").c_str(), params)) {
        return fail("参数名列表不匹配");
    }

    const bool mem_has_state = optim_state_count(opt) > 0 && optim_state(opt, 0)->data != nullptr;
    const bool has_state     = gguf_get_bool(f, ks.k("has_state").c_str());
    if (mem_has_state != has_state) {
        return fail("优化器状态存在性不一致");
    }

    std::vector<Tensor*> states;
    if (has_state) {
        const int64_t n_states = optim_state_count(opt);
        states.reserve((size_t) n_states);
        for (int64_t i = 0; i < n_states; ++i) {
            states.push_back(optim_state(opt, i));
        }
        if (!names_match(f, ks.k("state_names").c_str(), states)) {
            return fail("状态张量名列表不匹配");
        }
    }

    for (Tensor* p : params) {
        if (!same_tensor(f, p)) {
            return fail("参数数据不一致");
        }
    }
    if (has_state) {
        for (Tensor* s : states) {
            if (!same_tensor(f, s)) {
                return fail("状态数据不一致");
            }
        }
        int64_t        n_steps = 0;
        const int64_t* steps   = gguf_get_arr_i64(f, ks.k("state_steps").c_str(), &n_steps);
        if (steps == nullptr || n_steps != n_params) {
            return fail("状态步数数组缺失或数量不符");
        }
        for (int64_t i = 0; i < n_params; ++i) {
            if (optim_state_step(opt, i) != steps[i]) {
                return fail("每参数状态步数不一致");
            }
        }
    }

    if (gguf_get_i64(f, ks.k("step").c_str()) != optim_step_count(opt)) {
        return fail("全局 step 不一致");
    }
    for (int64_t gi = 0; gi < n_groups; ++gi) {
        if (gguf_get_f32(f, ks.g(gi, "lr").c_str()) != optim_get_lr_group(opt, gi)) {
            return fail("每组学习率不一致");
        }
    }
    if (part.sched != nullptr) {
        const std::string sk = ks.s();
        if (!gguf_has_key(f, sk.c_str()) ||
            lr_scheduler_epoch(part.sched) != gguf_get_i64(f, sk.c_str())) {
            return fail("调度器 epoch 不一致");
        }
    }
    return true;
}

} // namespace

bool checkpoint_save(const char* path, const CheckpointItems& items, const char* model_arch) {
    TRC_ASSERT(path != nullptr, "checkpoint_save: path 为空");
    std::vector<Partition> parts;
    std::string            err;
    TRC_ASSERT(collect_partitions(items, &parts, &err), "checkpoint_save: %s", err.c_str());

    GgufWriter* w = gguf_writer_new();
    gguf_writer_set_arch(w, CKPT_ARCH);
    gguf_writer_set_u32(w, "traincpp.checkpoint.version", CKPT_SCHEMA_VERSION);
    if (model_arch != nullptr && *model_arch != '\0') {
        gguf_writer_set_str(w, "traincpp.model.arch", model_arch);
    }
    gguf_writer_set_u32(w, "traincpp.optimizer.count", (uint32_t) parts.size());

    std::vector<std::pair<std::string, std::string>> seen_names;  // (张量名, 分区名)

    for (size_t pi = 0; pi < parts.size(); ++pi) {
        Partition&  part = parts[pi];
        Optimizer*  opt  = part.opt;
        const KeyScheme ks{/*v1=*/false, (int64_t) pi};

        const int64_t n_groups = optim_group_count(opt);
        const int64_t n_params = optim_param_count(opt);
        TRC_ASSERT(n_groups > 0 && n_params > 0, "checkpoint_save: 优化器 %s 没有参数",
                   part.name.c_str());

        const OptimType t0 = optim_group_type(opt, 0);
        for (int64_t gi = 0; gi < n_groups; ++gi) {
            TRC_ASSERT(optim_group_type(opt, gi) == t0,
                       "checkpoint_save: 优化器 %s 暂不支持组间混合优化器类型（第 %lld 组）",
                       part.name.c_str(), (long long) gi);
        }

        // 参数：名字非空、分区内唯一、跨分区全局唯一
        std::vector<Tensor*>     params((size_t) n_params);
        std::vector<const char*> pnames((size_t) n_params);
        for (int64_t i = 0; i < n_params; ++i) {
            Tensor* p = optim_param(opt, i);
            TRC_ASSERT(p->name[0] != '\0', "checkpoint_save: 优化器 %s 参数[%lld] 没有名字",
                       part.name.c_str(), (long long) i);
            for (int64_t j = 0; j < i; ++j) {
                TRC_ASSERT(std::strcmp(pnames[(size_t) j], p->name) != 0,
                           "checkpoint_save: 优化器 %s 参数名重复 %s", part.name.c_str(), p->name);
            }
            check_global_name(&seen_names, p->name, part.name.c_str());
            params[(size_t) i] = p;
            pnames[(size_t) i] = p->name;
        }

        // 状态：要么全部分配、要么全未分配
        bool has_state = optim_state_count(opt) > 0;
        if (optim_state_count(opt) > 0) {
            const bool allocated = optim_state(opt, 0)->data != nullptr;
            for (int64_t i = 0; i < optim_state_count(opt); ++i) {
                TRC_ASSERT((optim_state(opt, i)->data != nullptr) == allocated,
                           "checkpoint_save: 优化器 %s 状态分配不一致", part.name.c_str());
            }
            has_state = allocated;
        }
        TRC_ASSERT(!(optim_step_count(opt) > 0 && !has_state),
                   "checkpoint_save: 优化器 %s 已训练 %lld 步但状态未分配", part.name.c_str(),
                   (long long) optim_step_count(opt));

        std::vector<Tensor*>     states;
        std::vector<const char*> snames;
        if (has_state) {
            const int64_t n_states = optim_state_count(opt);
            states.resize((size_t) n_states);
            snames.resize((size_t) n_states);
            for (int64_t i = 0; i < n_states; ++i) {
                Tensor* s = optim_state(opt, i);
                TRC_ASSERT(s->name[0] != '\0', "checkpoint_save: 优化器 %s 状态张量[%lld] 没有名字",
                           part.name.c_str(), (long long) i);
                for (int64_t j = 0; j < i; ++j) {
                    TRC_ASSERT(std::strcmp(snames[(size_t) j], s->name) != 0,
                               "checkpoint_save: 优化器 %s 状态张量名重复 %s（参数名过长导致截断？）",
                               part.name.c_str(), s->name);
                }
                check_global_name(&seen_names, s->name, part.name.c_str());
                states[(size_t) i] = s;
                snames[(size_t) i] = s->name;
            }
        }

        // 元数据
        gguf_writer_set_str(w, ks.k("name").c_str(), part.name.c_str());
        gguf_writer_set_str(w, ks.k("type").c_str(), optim_type_str(t0));
        gguf_writer_set_u32(w, ks.k("groups").c_str(), (uint32_t) n_groups);
        gguf_writer_set_u32(w, ks.k("param_count").c_str(), (uint32_t) n_params);
        gguf_writer_set_i64(w, ks.k("step").c_str(), optim_step_count(opt));
        gguf_writer_set_bool(w, ks.k("has_state").c_str(), has_state);
        gguf_writer_set_arr_str(w, ks.k("param_names").c_str(), pnames.data(), n_params);
        if (has_state) {
            gguf_writer_set_arr_str(w, ks.k("state_names").c_str(), snames.data(),
                                    optim_state_count(opt));
            // 每个参数各自的优化器状态步数（AdamW bias correction / SGD 首步判断）
            std::vector<int64_t> state_steps((size_t) n_params, 0);
            for (int64_t i = 0; i < n_params; ++i) {
                state_steps[(size_t) i] = optim_state_step(opt, i);
            }
            gguf_writer_set_arr_i64(w, ks.k("state_steps").c_str(), state_steps.data(), n_params);
        }
        if (part.sched != nullptr) {
            gguf_writer_set_i64(w, ks.s().c_str(),
                                lr_scheduler_epoch(part.sched));
        }
        for (int64_t gi = 0; gi < n_groups; ++gi) {
            write_group_options(w, ks, opt, gi);
        }

        for (Tensor* p : params) {
            gguf_writer_add_tensor(w, p->name, p, TYPE_F32);
        }
        if (has_state) {
            for (Tensor* s : states) {
                gguf_writer_add_tensor(w, s->name, s, TYPE_F32);
            }
        }
    }

    if (items.rng != nullptr) {
        gguf_writer_set_u64(w, "traincpp.rng.state", items.rng->state);
    }

    const bool ok = gguf_writer_write(w, path);
    gguf_writer_free(w);
    return ok;
}

bool checkpoint_load(const char* path, const CheckpointItems& items) {
    TRC_ASSERT(path != nullptr, "checkpoint_load: path 为空");
    std::vector<Partition> req;
    std::string            err;
    if (!collect_partitions(items, &req, &err)) {
        TRC_LOG_ERROR("checkpoint: %s（%s）", err.c_str(), path);
        return false;
    }

    GgufFile* f = gguf_open(path);
    if (f == nullptr) {
        return false;
    }
    const auto fail = [&](const char* why) {
        TRC_LOG_ERROR("checkpoint: %s（%s）", why, path);
        gguf_close(f);
        return false;
    };

    const char* arch = gguf_get_str(f, "general.architecture");
    if (arch == nullptr || std::strcmp(arch, CKPT_ARCH) != 0) {
        return fail("不是 checkpoint 文件（general.architecture 不符）");
    }
    const uint32_t version = gguf_get_u32(f, "traincpp.checkpoint.version");
    if (version != CKPT_SCHEMA_V1 && version != CKPT_SCHEMA_V2) {
        return fail("schema 版本不支持");
    }

    std::vector<int64_t> file_idx;
    if (!match_partitions(f, version, req, &file_idx, &err)) {
        return fail(err.c_str());
    }

    for (size_t i = 0; i < req.size(); ++i) {
        const KeyScheme ks{/*v1=*/version == CKPT_SCHEMA_V1, file_idx[i]};
        if (!load_partition(f, ks, req[i], fail)) {
            return false;
        }
    }

    // RNG 状态（共享一份）
    if (items.rng != nullptr && gguf_has_key(f, "traincpp.rng.state")) {
        items.rng->state = gguf_get_u64(f, "traincpp.rng.state");
    }

    gguf_close(f);
    return true;
}

bool checkpoint_probe(const char* path, CheckpointInfo* out) {
    TRC_ASSERT(path != nullptr && out != nullptr, "checkpoint_probe: 参数为空");

    GgufFile* f = gguf_open(path, /*load_data=*/false);
    if (f == nullptr) {
        return false;
    }
    const char* arch = gguf_get_str(f, "general.architecture");
    if (arch == nullptr || std::strcmp(arch, CKPT_ARCH) != 0) {
        gguf_close(f);
        return false;
    }
    const uint32_t version = gguf_get_u32(f, "traincpp.checkpoint.version");
    if (version != CKPT_SCHEMA_V1 && version != CKPT_SCHEMA_V2) {
        gguf_close(f);
        return false;
    }

    *out = CheckpointInfo{};
    out->version  = (int64_t) version;
    out->has_rng  = gguf_has_key(f, "traincpp.rng.state");
    if (version == CKPT_SCHEMA_V1) {
        out->optimizer_count = 1;
        out->step            = gguf_get_i64(f, "traincpp.optimizer.step");
        out->param_count     = (int64_t) gguf_get_u32(f, "traincpp.optimizer.param_count");
        out->group_count     = (int64_t) gguf_get_u32(f, "traincpp.optimizer.groups");
        const char* tstr     = gguf_get_str(f, "traincpp.optimizer.type");
        std::snprintf(out->optimizer_type, sizeof(out->optimizer_type), "%s",
                      tstr != nullptr ? tstr : "");
        out->has_scheduler   = gguf_has_key(f, "traincpp.scheduler.epoch");
        out->scheduler_epoch = out->has_scheduler ? gguf_get_i64(f, "traincpp.scheduler.epoch") : 0;
    } else {
        out->optimizer_count = (int64_t) gguf_get_u32(f, "traincpp.optimizer.count");
        const KeyScheme ks{/*v1=*/false, 0};
        out->step        = gguf_get_i64(f, ks.k("step").c_str());
        out->param_count = (int64_t) gguf_get_u32(f, ks.k("param_count").c_str());
        out->group_count = (int64_t) gguf_get_u32(f, ks.k("groups").c_str());
        const char* tstr = gguf_get_str(f, ks.k("type").c_str());
        std::snprintf(out->optimizer_type, sizeof(out->optimizer_type), "%s",
                      tstr != nullptr ? tstr : "");
        const std::string sk = ks.s();
        out->has_scheduler   = gguf_has_key(f, sk.c_str());
        out->scheduler_epoch = out->has_scheduler ? gguf_get_i64(f, sk.c_str()) : 0;
    }
    const char* march = gguf_has_key(f, "traincpp.model.arch") ? gguf_get_str(f, "traincpp.model.arch")
                                                                : nullptr;
    std::snprintf(out->model_arch, sizeof(out->model_arch), "%s", march != nullptr ? march : "");

    gguf_close(f);
    return true;
}

bool checkpoint_verify(const char* path, const CheckpointItems& items) {
    TRC_ASSERT(path != nullptr, "checkpoint_verify: path 为空");
    std::vector<Partition> req;
    std::string            err;
    if (!collect_partitions(items, &req, &err)) {
        TRC_LOG_ERROR("checkpoint 自检失败：%s（%s）", err.c_str(), path);
        return false;
    }

    GgufFile* f = gguf_open(path);
    if (f == nullptr) {
        return false;
    }
    const auto fail = [&](const char* why) {
        TRC_LOG_ERROR("checkpoint 自检失败：%s（%s）", why, path);
        gguf_close(f);
        return false;
    };

    const char* arch = gguf_get_str(f, "general.architecture");
    if (arch == nullptr || std::strcmp(arch, CKPT_ARCH) != 0) {
        return fail("不是 checkpoint 文件（general.architecture 不符）");
    }
    const uint32_t version = gguf_get_u32(f, "traincpp.checkpoint.version");
    if (version != CKPT_SCHEMA_V1 && version != CKPT_SCHEMA_V2) {
        return fail("schema 版本不支持");
    }

    std::vector<int64_t> file_idx;
    if (!match_partitions(f, version, req, &file_idx, &err)) {
        return fail(err.c_str());
    }

    int64_t total_params = 0;
    int64_t total_groups = 0;
    for (size_t i = 0; i < req.size(); ++i) {
        const KeyScheme ks{/*v1=*/version == CKPT_SCHEMA_V1, file_idx[i]};
        if (!verify_partition(f, ks, req[i], fail)) {
            return false;
        }
        total_params += optim_param_count(req[i].opt);
        total_groups += optim_group_count(req[i].opt);
    }

    if (items.rng != nullptr) {
        if (!gguf_has_key(f, "traincpp.rng.state") ||
            items.rng->state != gguf_get_u64(f, "traincpp.rng.state")) {
            return fail("RNG 状态不一致");
        }
    }

    TRC_LOG_INFO("checkpoint 自检通过：%s（%zu 优化器，%lld 参数，%lld 组，step=%lld）", path,
                 req.size(), (long long) total_params, (long long) total_groups,
                 (long long) optim_step_count(req[0].opt));
    gguf_close(f);
    return true;
}

} // namespace traincpp
