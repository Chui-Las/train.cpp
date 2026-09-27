// train.cpp - ggml 交叉验证工具（M1.6c）
//
// 目的：证明 train.cpp 导出的 GGUF（模型样例 + 训练 checkpoint）能被 **ggml 本体**正确读取：
//   1. 用 traincpp 的 GGUF writer / checkpoint_save 写出样例文件；
//   2. 用 ggml 的 gguf_init_from_file 重新打开，校验元数据、张量名/类型/形状/字节数与数据值；
//   3. 打印检查统计，失败时返回非 0。
//
// 该工具是可选的外部验证（不参与主构建、不成为库依赖、不修改 ggml 参考项目源码）。
#define _CRT_SECURE_NO_WARNINGS

#include "ggml.h"
#include "gguf.h"

#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace traincpp;

namespace {

int g_checks   = 0;
int g_failures = 0;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        ++g_checks;                                            \
        if (!(cond)) {                                         \
            ++g_failures;                                      \
            std::printf("  [FAIL] ");                          \
            std::printf(__VA_ARGS__);                          \
            std::printf("\n");                                 \
        }                                                      \
    } while (0)

// ---------------------------------------------------------------- 导出样例（traincpp）

std::vector<float> sample_w() {
    std::vector<float> v(20);
    for (int i = 0; i < 20; ++i) {
        v[(size_t) i] = (float) i * 0.5f - 2.0f;  // 均可被 F16 精确表示
    }
    return v;
}

std::vector<float> sample_h() {
    return {0.5f, -1.5f, 2.0f, 3.25f, 0.0f, -0.125f};
}

std::vector<int32_t> sample_idx() {
    return {-3, -1, 0, 2, 7, 100};
}

bool export_sample_model(const std::string& path) {
    ContextParams params;
    params.mem_size = 1 << 20;
    params.no_alloc = false;
    Context* ctx = context_new(params);

    Tensor* w   = new_tensor_2d(ctx, TYPE_F32, 5, 4);
    Tensor* h   = new_tensor_1d(ctx, TYPE_F32, 6);
    Tensor* idx = new_tensor_1d(ctx, TYPE_I32, 6);
    tensor_set_name(w, "w");
    tensor_set_name(h, "h");
    tensor_set_name(idx, "idx");

    const std::vector<float>   wv = sample_w();
    const std::vector<float>   hv = sample_h();
    const std::vector<int32_t> iv = sample_idx();
    tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
    tensor_set(h, hv.data(), 0, hv.size() * sizeof(float));
    tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int32_t));

    GgufWriter* gw = gguf_writer_new();
    gguf_writer_set_arch(gw, "trc.crosscheck");
    gguf_writer_set_u32(gw, "test.u32", 123u);
    gguf_writer_set_i32(gw, "test.i32", -7);
    gguf_writer_set_f32(gw, "test.f32", 0.25f);
    gguf_writer_set_bool(gw, "test.bool", true);
    gguf_writer_set_i64(gw, "test.i64", -5);
    gguf_writer_set_str(gw, "test.str", "中文 ok");
    const int32_t arr_i32[3] = {1, 2, 3};
    gguf_writer_set_arr_i32(gw, "test.arr_i32", arr_i32, 3);
    const int64_t arr_i64[3] = {1, -5, 1099511627776ll};
    gguf_writer_set_arr_i64(gw, "test.arr_i64", arr_i64, 3);
    const char* arr_str[2] = {"a", "贝塔"};
    gguf_writer_set_arr_str(gw, "test.arr_str", arr_str, 2);
    gguf_writer_add_tensor(gw, "w", w, TYPE_F32);
    gguf_writer_add_tensor(gw, "h", h, TYPE_F16);  // F32 -> F16 转换
    gguf_writer_add_tensor(gw, "idx", idx, TYPE_I32);

    const bool ok = gguf_writer_write(gw, path.c_str());
    CHECK(ok, "写出模型样例失败: %s", path.c_str());
    gguf_writer_free(gw);
    context_free(ctx);
    return ok;
}

bool export_sample_checkpoint(const std::string& path) {
    Context* ctx = context_new(1 << 20);  // no_alloc=true

    Tensor* p1 = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* p2 = new_tensor_2d(ctx, TYPE_F32, 2, 3);
    Tensor* p3 = new_tensor_1d(ctx, TYPE_F32, 3);
    tensor_set_name(p1, "p1");
    tensor_set_name(p2, "p2");
    tensor_set_name(p3, "p3");
    tensor_set_param(p1);
    tensor_set_param(p2);
    tensor_set_param(p3);

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, device_cpu()->default_buffer_type());
    CHECK(buf != nullptr, "checkpoint 样例：buffer 分配失败");

    const std::vector<float> p1v = {0.1f, -0.2f, 0.3f, -0.4f};
    const std::vector<float> p2v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> p3v = {7.0f, 8.0f, 9.0f};
    tensor_set(p1, p1v.data(), 0, p1v.size() * sizeof(float));
    tensor_set(p2, p2v.data(), 0, p2v.size() * sizeof(float));
    tensor_set(p3, p3v.data(), 0, p3v.size() * sizeof(float));

    // ---- 分区 G：AdamW（p1/p2，m/v 状态 + 指数调度器） ----
    Tensor*    g_params[2] = {p1, p2};
    Optimizer* g_opt       = optim_adamw_new(ctx, g_params, 2, AdamwOptions{});
    optim_alloc_state(g_opt);

    const std::vector<float> m1 = {0.01f, 0.02f, 0.03f, 0.04f};
    const std::vector<float> v1 = {0.11f, 0.12f, 0.13f, 0.14f};
    const std::vector<float> m2 = {0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f};
    const std::vector<float> v2 = {0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f};
    tensor_set(optim_state(g_opt, 0), m1.data(), 0, m1.size() * sizeof(float));  // p1.m
    tensor_set(optim_state(g_opt, 1), v1.data(), 0, v1.size() * sizeof(float));  // p1.v
    tensor_set(optim_state(g_opt, 2), m2.data(), 0, m2.size() * sizeof(float));  // p2.m
    tensor_set(optim_state(g_opt, 3), v2.data(), 0, v2.size() * sizeof(float));  // p2.v
    optim_set_step_count(g_opt, 3);
    optim_set_state_step(g_opt, 0, 3);
    optim_set_state_step(g_opt, 1, 2);
    optim_set_lr(g_opt, 0.002f);
    LrScheduler* g_sched = lr_scheduler_exponential_new(g_opt, 0.5f);
    lr_scheduler_step(g_sched);  // epoch=1，lr=0.001

    // ---- 分区 D：SGD momentum（p3） ----
    SgdOptions so;
    so.lr       = 0.01f;
    so.momentum = 0.9f;
    Tensor*    d_params[1] = {p3};
    Optimizer* d_opt       = optim_sgd_new(ctx, d_params, 1, so);
    optim_alloc_state(d_opt);
    const std::vector<float> mom3 = {0.7f, 0.8f, 0.9f};
    tensor_set(optim_state(d_opt, 0), mom3.data(), 0, mom3.size() * sizeof(float));
    optim_set_step_count(d_opt, 2);
    optim_set_state_step(d_opt, 0, 2);

    Rng rng;
    rng_seed(&rng, 42);

    CheckpointOptimizer parts[2] = {{"G", g_opt, g_sched}, {"D", d_opt, nullptr}};
    CheckpointItems     items{};
    items.rng             = &rng;
    items.optimizers      = parts;
    items.optimizer_count = 2;
    const bool ok = checkpoint_save(path.c_str(), items, "crosscheck");
    CHECK(ok, "写出 checkpoint 样例失败: %s", path.c_str());

    lr_scheduler_free(g_sched);
    optim_free(d_opt);
    optim_free(g_opt);
    buffer_free(buf);
    context_free(ctx);
    return ok;
}

// ---------------------------------------------------------------- ggml 读取校验

struct GgmlReader {
    struct ggml_context* ctx = nullptr;
    struct gguf_context* gf = nullptr;

    ~GgmlReader() {
        if (gf != nullptr) {
            gguf_free(gf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t require_key(struct gguf_context* gf, const char* key) {
    const int64_t id = gguf_find_key(gf, key);
    CHECK(id >= 0, "ggml 找不到元数据 key %s", key);
    return id;
}

void check_f32_tensor(struct ggml_context* gctx, struct gguf_context* gf, const char* name,
                      int64_t ne0, int64_t ne1, const std::vector<float>& expect) {
    const int64_t tid = gguf_find_tensor(gf, name);
    CHECK(tid >= 0, "ggml 找不到张量 %s", name);
    if (tid < 0) {
        return;
    }
    CHECK(gguf_get_tensor_type(gf, tid) == GGML_TYPE_F32, "%s 类型不是 F32", name);
    const int64_t* ne = gguf_get_tensor_ne(gf, tid);
    CHECK(ne[0] == ne0 && ne[1] == ne1, "%s 形状不符（[%lld,%lld] vs [%lld,%lld]）", name,
          (long long) ne[0], (long long) ne[1], (long long) ne0, (long long) ne1);

    struct ggml_tensor* t = ggml_get_tensor(gctx, name);
    CHECK(t != nullptr, "ggml 上下文缺少张量 %s", name);
    if (t == nullptr) {
        return;
    }
    CHECK(ggml_nbytes(t) == gguf_get_tensor_size(gf, tid), "%s 字节数不符", name);
    CHECK(ggml_nelements(t) == (int64_t) expect.size(), "%s 元素数不符", name);
    const float* got = (const float*) t->data;
    for (size_t i = 0; i < expect.size(); ++i) {
        CHECK(got[i] == expect[i], "%s[%zu] = %.9g != %.9g", name, i, (double) got[i],
              (double) expect[i]);
    }
}

void check_f16_tensor(struct ggml_context* gctx, struct gguf_context* gf, const char* name,
                      const std::vector<float>& expect) {
    const int64_t tid = gguf_find_tensor(gf, name);
    CHECK(tid >= 0, "ggml 找不到张量 %s", name);
    if (tid < 0) {
        return;
    }
    CHECK(gguf_get_tensor_type(gf, tid) == GGML_TYPE_F16, "%s 类型不是 F16", name);
    struct ggml_tensor* t = ggml_get_tensor(gctx, name);
    CHECK(t != nullptr, "ggml 上下文缺少张量 %s", name);
    if (t == nullptr) {
        return;
    }
    const ggml_fp16_t* got = (const ggml_fp16_t*) t->data;
    for (size_t i = 0; i < expect.size(); ++i) {
        CHECK(ggml_fp16_to_fp32(got[i]) == expect[i], "%s[%zu] 的 F16 解码值不符", name, i);
    }
}

void check_i32_tensor(struct ggml_context* gctx, struct gguf_context* gf, const char* name,
                      const std::vector<int32_t>& expect) {
    const int64_t tid = gguf_find_tensor(gf, name);
    CHECK(tid >= 0, "ggml 找不到张量 %s", name);
    if (tid < 0) {
        return;
    }
    CHECK(gguf_get_tensor_type(gf, tid) == GGML_TYPE_I32, "%s 类型不是 I32", name);
    struct ggml_tensor* t = ggml_get_tensor(gctx, name);
    CHECK(t != nullptr, "ggml 上下文缺少张量 %s", name);
    if (t == nullptr) {
        return;
    }
    const int32_t* got = (const int32_t*) t->data;
    for (size_t i = 0; i < expect.size(); ++i) {
        CHECK(got[i] == expect[i], "%s[%zu] = %d != %d", name, i, got[i], expect[i]);
    }
}

void crosscheck_model(const std::string& path) {
    std::printf("-- 模型样例：%s\n", path.c_str());
    GgmlReader r;
    struct gguf_init_params ip = { /*no_alloc=*/false, /*ctx=*/&r.ctx };
    r.gf = gguf_init_from_file(path.c_str(), ip);
    CHECK(r.gf != nullptr, "ggml 无法打开 %s", path.c_str());
    if (r.gf == nullptr) {
        return;
    }

    CHECK(gguf_get_version(r.gf) == 3, "版本 %u != 3", gguf_get_version(r.gf));
    CHECK(gguf_get_alignment(r.gf) == GGUF_DEFAULT_ALIGNMENT, "对齐 %zu != 32",
          gguf_get_alignment(r.gf));
    CHECK(gguf_get_n_tensors(r.gf) == 3, "张量数 %lld != 3", (long long) gguf_get_n_tensors(r.gf));

    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "general.architecture")),
                      "trc.crosscheck") == 0,
          "general.architecture 不符");
    CHECK(gguf_get_val_u32(r.gf, require_key(r.gf, "test.u32")) == 123u, "u32 元数据不符");
    CHECK(gguf_get_val_i32(r.gf, require_key(r.gf, "test.i32")) == -7, "i32 元数据不符");
    CHECK(gguf_get_val_f32(r.gf, require_key(r.gf, "test.f32")) == 0.25f, "f32 元数据不符");
    CHECK(gguf_get_val_bool(r.gf, require_key(r.gf, "test.bool")), "bool 元数据不符");
    CHECK(gguf_get_val_i64(r.gf, require_key(r.gf, "test.i64")) == -5, "i64 元数据不符");
    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "test.str")), "中文 ok") == 0,
          "字符串元数据不符");

    const int64_t a32 = require_key(r.gf, "test.arr_i32");
    CHECK(gguf_get_arr_type(r.gf, a32) == GGUF_TYPE_INT32, "arr_i32 元素类型不符");
    CHECK(gguf_get_arr_n(r.gf, a32) == 3, "arr_i32 元素数不符");
    const int32_t* v32 = (const int32_t*) gguf_get_arr_data(r.gf, a32);
    CHECK(v32[0] == 1 && v32[1] == 2 && v32[2] == 3, "arr_i32 值不符");

    const int64_t a64 = require_key(r.gf, "test.arr_i64");
    CHECK(gguf_get_arr_type(r.gf, a64) == GGUF_TYPE_INT64, "arr_i64 元素类型不符");
    const int64_t* v64 = (const int64_t*) gguf_get_arr_data(r.gf, a64);
    CHECK(v64[2] == 1099511627776ll, "arr_i64 值不符");

    const int64_t as = require_key(r.gf, "test.arr_str");
    CHECK(gguf_get_arr_type(r.gf, as) == GGUF_TYPE_STRING, "arr_str 元素类型不符");
    CHECK(std::strcmp(gguf_get_arr_str(r.gf, as, 1), "贝塔") == 0, "arr_str 值不符");

    check_f32_tensor(r.ctx, r.gf, "w", 5, 4, sample_w());
    check_f16_tensor(r.ctx, r.gf, "h", sample_h());
    check_i32_tensor(r.ctx, r.gf, "idx", sample_idx());
}

void crosscheck_checkpoint(const std::string& path) {
    std::printf("-- checkpoint：%s\n", path.c_str());
    GgmlReader r;
    struct gguf_init_params ip = { /*no_alloc=*/false, /*ctx=*/&r.ctx };
    r.gf = gguf_init_from_file(path.c_str(), ip);
    CHECK(r.gf != nullptr, "ggml 无法打开 %s", path.c_str());
    if (r.gf == nullptr) {
        return;
    }

    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "general.architecture")),
                      "traincpp.checkpoint") == 0,
          "checkpoint arch 不符");
    CHECK(gguf_get_val_u32(r.gf, require_key(r.gf, "traincpp.checkpoint.version")) == 2,
          "checkpoint 版本不符");
    CHECK(gguf_get_val_u32(r.gf, require_key(r.gf, "traincpp.optimizer.count")) == 2,
          "优化器分区数不符");
    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "traincpp.model.arch")),
                      "crosscheck") == 0,
          "model_arch 不符");
    CHECK(gguf_get_val_u64(r.gf, require_key(r.gf, "traincpp.rng.state")) == 42u, "rng state 不符");

    // ---- 分区 0：G = AdamW（带指数调度器） ----
    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "traincpp.optimizer.0.name")), "G") == 0,
          "分区 0 名字不符");
    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "traincpp.optimizer.0.type")),
                      "adamw") == 0,
          "分区 0 类型不符");
    CHECK(gguf_get_val_u32(r.gf, require_key(r.gf, "traincpp.optimizer.0.param_count")) == 2,
          "分区 0 参数数不符");
    CHECK(gguf_get_val_i64(r.gf, require_key(r.gf, "traincpp.optimizer.0.step")) == 3,
          "分区 0 step 不符");
    CHECK(gguf_get_val_bool(r.gf, require_key(r.gf, "traincpp.optimizer.0.has_state")),
          "分区 0 has_state 不符");
    CHECK(gguf_get_val_i64(r.gf, require_key(r.gf, "traincpp.optimizer.0.scheduler.epoch")) == 1,
          "分区 0 调度器 epoch 不符");
    CHECK(gguf_get_val_f32(r.gf, require_key(r.gf, "traincpp.optimizer.0.group0.lr")) == 0.001f,
          "分区 0 lr 不符");

    const int64_t gnames = require_key(r.gf, "traincpp.optimizer.0.param_names");
    CHECK(gguf_get_arr_n(r.gf, gnames) == 2, "分区 0 param_names 数量不符");
    CHECK(std::strcmp(gguf_get_arr_str(r.gf, gnames, 0), "p1") == 0 &&
              std::strcmp(gguf_get_arr_str(r.gf, gnames, 1), "p2") == 0,
          "分区 0 param_names 内容不符");

    const int64_t gsnames = require_key(r.gf, "traincpp.optimizer.0.state_names");
    CHECK(gguf_get_arr_n(r.gf, gsnames) == 4, "分区 0 state_names 数量不符");
    CHECK(std::strcmp(gguf_get_arr_str(r.gf, gsnames, 2), "p2.m") == 0,
          "分区 0 state_names 内容不符");

    const int64_t gsteps = require_key(r.gf, "traincpp.optimizer.0.state_steps");
    CHECK(gguf_get_arr_type(r.gf, gsteps) == GGUF_TYPE_INT64, "分区 0 state_steps 元素类型不符");
    const int64_t* gsv = (const int64_t*) gguf_get_arr_data(r.gf, gsteps);
    CHECK(gguf_get_arr_n(r.gf, gsteps) == 2 && gsv[0] == 3 && gsv[1] == 2,
          "分区 0 state_steps 内容不符");

    // ---- 分区 1：D = SGD momentum（无调度器） ----
    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "traincpp.optimizer.1.name")), "D") == 0,
          "分区 1 名字不符");
    CHECK(std::strcmp(gguf_get_val_str(r.gf, require_key(r.gf, "traincpp.optimizer.1.type")),
                      "sgd") == 0,
          "分区 1 类型不符");
    CHECK(gguf_get_val_u32(r.gf, require_key(r.gf, "traincpp.optimizer.1.param_count")) == 1,
          "分区 1 参数数不符");
    CHECK(gguf_get_val_i64(r.gf, require_key(r.gf, "traincpp.optimizer.1.step")) == 2,
          "分区 1 step 不符");
    CHECK(gguf_get_val_f32(r.gf, require_key(r.gf, "traincpp.optimizer.1.group0.lr")) == 0.01f,
          "分区 1 lr 不符");
    CHECK(gguf_get_val_f32(r.gf, require_key(r.gf, "traincpp.optimizer.1.group0.momentum")) == 0.9f,
          "分区 1 momentum 不符");
    CHECK(gguf_find_key(r.gf, "traincpp.optimizer.1.scheduler.epoch") < 0, "分区 1 不应有调度器");

    const int64_t dnames = require_key(r.gf, "traincpp.optimizer.1.param_names");
    CHECK(gguf_get_arr_n(r.gf, dnames) == 1 &&
              std::strcmp(gguf_get_arr_str(r.gf, dnames, 0), "p3") == 0,
          "分区 1 param_names 内容不符");
    const int64_t dsnames = require_key(r.gf, "traincpp.optimizer.1.state_names");
    CHECK(gguf_get_arr_n(r.gf, dsnames) == 1 &&
              std::strcmp(gguf_get_arr_str(r.gf, dsnames, 0), "p3.momentum") == 0,
          "分区 1 state_names 内容不符");

    CHECK(gguf_get_n_tensors(r.gf) == 8, "checkpoint 张量数 %lld != 8",
          (long long) gguf_get_n_tensors(r.gf));
    check_f32_tensor(r.ctx, r.gf, "p1", 4, 1, {0.1f, -0.2f, 0.3f, -0.4f});
    check_f32_tensor(r.ctx, r.gf, "p2", 2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    check_f32_tensor(r.ctx, r.gf, "p3", 3, 1, {7.0f, 8.0f, 9.0f});
    check_f32_tensor(r.ctx, r.gf, "p1.m", 4, 1, {0.01f, 0.02f, 0.03f, 0.04f});
    check_f32_tensor(r.ctx, r.gf, "p1.v", 4, 1, {0.11f, 0.12f, 0.13f, 0.14f});
    check_f32_tensor(r.ctx, r.gf, "p2.m", 2, 3, {0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f});
    check_f32_tensor(r.ctx, r.gf, "p2.v", 2, 3, {0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f});
    check_f32_tensor(r.ctx, r.gf, "p3.momentum", 3, 1, {0.7f, 0.8f, 0.9f});
}

} // namespace

int main() {
    const std::filesystem::path out_dir = std::filesystem::current_path() / "crosscheck_out";
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    const std::string model_path   = (out_dir / "trc_sample_model.gguf").string();
    const std::string ckpt_path    = (out_dir / "trc_sample_checkpoint.gguf").string();

    std::printf("== train.cpp 导出 ==\n");
    if (!export_sample_model(model_path) || !export_sample_checkpoint(ckpt_path)) {
        std::printf("导出失败\n");
        return 1;
    }
    std::printf("  %s\n  %s\n\n", model_path.c_str(), ckpt_path.c_str());

    std::printf("== ggml 读取校验 ==\n");
    crosscheck_model(model_path);
    crosscheck_checkpoint(ckpt_path);

    std::printf("\n交叉验证：%d 项检查，%d 项失败\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
