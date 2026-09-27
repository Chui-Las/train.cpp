// train.cpp - GGUF 文件读写（与 ggml v0.23.0 对齐，M1.6a）
//
// 目的：
//   - 训练产物导出为 GGUF，可被 ggml 生态程序直接加载推理（首要兼容目标是 ggml）；
//   - 可读取 ggml 生态导出的 GGUF（含常见量化类型，反量化为 F32），用于热启动/微调。
//
// 格式要点（与 ggml src/gguf.cpp 逐条一致）：
//   - 文件头 24 字节：magic "GGUF"(4) + version(u32) + tensor_count(u64) + metadata_kv_count(u64)
//   - KV 区：key(u64 长度 + UTF-8 无 NUL) + 类型(int32) + 值；
//     ARRAY 额外带元素类型(int32)与元素个数(u64)；字符串元素同样为 u64 长度 + 字节
//   - tensor info 区：name(u64+UTF-8) + n_dims(u32) + ne[0..n_dims-1](int64，**不反转**)
//     + type(int32，值即 ggml_type) + offset(u64，相对数据段起点)
//   - 数据段：起始对齐到 alignment（默认 32），每个张量数据按 alignment 补齐，offset 连续无洞
//   - 写端恒为 v3；读端支持 v2/v3（v1 不再支持，与 ggml 一致）；仅支持小端主机
//
// 与 ggml 的差异（已记入 docs/兼容性.md）：
//   - 读端允许数据段末尾缺少填充（ggml 严格要求按对齐累计读取）
//   - 写端不做量化（仅原样 / F32<->F16）；量化导出留后续阶段
//   - 反量化读取支持 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0（K-quant 等留后续）
#pragma once

#include "trc_context.h"
#include "trc_tensor.h"
#include "trc_types.h"

#include <cstddef>
#include <cstdint>

namespace traincpp {

// GGUF 元数据类型（与 ggml enum gguf_type 逐值一致；落盘为 int32）
enum class GgufType : int32_t {
    U8     = 0,
    I8     = 1,
    U16    = 2,
    I16    = 3,
    U32    = 4,
    I32    = 5,
    F32    = 6,
    BOOL   = 7,
    STRING = 8,
    ARRAY  = 9,
    U64    = 10,
    I64    = 11,
    F64    = 12,
};

struct GgufFile;
struct GgufWriter;

// 张量元信息（值语义；name 指针随 gguf_close 失效）
struct GgufTensorInfo {
    const char* name    = nullptr;
    Type        type    = TYPE_F32;
    int64_t     ne[MAX_DIMS] = {1, 1, 1, 1};  // 顺序即 ggml ne（不反转）
    size_t      offset  = 0;                  // 相对数据段起点
    size_t      nbytes  = 0;
};

// ---------------------------------------------------------------- 读

// 打开 GGUF。load_data=false 时只解析元数据（张量数据访问返回失败）。
// 文件不存在/损坏/版本不支持时返回 nullptr 并打印错误。
GgufFile* gguf_open(const char* path, bool load_data = true);
void      gguf_close(GgufFile* f);

int64_t gguf_version(const GgufFile* f);      // 2 或 3
size_t  gguf_alignment(const GgufFile* f);    // 数据段对齐（默认 32）
size_t  gguf_data_offset(const GgufFile* f);  // 数据段文件偏移

int64_t     gguf_kv_count(const GgufFile* f);
int64_t     gguf_find_key(const GgufFile* f, const char* key);  // 未找到返回 -1
const char* gguf_key(const GgufFile* f, int64_t index);
GgufType    gguf_kv_type(const GgufFile* f, int64_t index);
GgufType    gguf_arr_type(const GgufFile* f, int64_t index);
int64_t     gguf_arr_count(const GgufFile* f, int64_t index);
bool        gguf_has_key(const GgufFile* f, const char* key);

// 标量取值：键缺失/类型不符时打印错误并返回 0（字符串返回 nullptr）。
// BOOL 落盘为 1 字节（与 ggml 一致）。
uint8_t     gguf_get_u8(const GgufFile* f, const char* key);
int8_t      gguf_get_i8(const GgufFile* f, const char* key);
uint16_t    gguf_get_u16(const GgufFile* f, const char* key);
int16_t     gguf_get_i16(const GgufFile* f, const char* key);
uint32_t    gguf_get_u32(const GgufFile* f, const char* key);
int32_t     gguf_get_i32(const GgufFile* f, const char* key);
float       gguf_get_f32(const GgufFile* f, const char* key);
bool        gguf_get_bool(const GgufFile* f, const char* key);
uint64_t    gguf_get_u64(const GgufFile* f, const char* key);
int64_t     gguf_get_i64(const GgufFile* f, const char* key);
double      gguf_get_f64(const GgufFile* f, const char* key);
const char* gguf_get_str(const GgufFile* f, const char* key);

// 数组取值（键缺失/类型不符/下标越界返回 nullptr，并打印错误）
const int32_t* gguf_get_arr_i32(const GgufFile* f, const char* key, int64_t* n);
const int64_t* gguf_get_arr_i64(const GgufFile* f, const char* key, int64_t* n);
const float*   gguf_get_arr_f32(const GgufFile* f, const char* key, int64_t* n);
const char*    gguf_get_arr_str(const GgufFile* f, const char* key, int64_t index);

int64_t               gguf_tensor_count(const GgufFile* f);
int64_t               gguf_find_tensor(const GgufFile* f, const char* name);  // 未找到返回 -1
const GgufTensorInfo* gguf_tensor_info(const GgufFile* f, int64_t index);
int64_t               gguf_tensor_nelements(const GgufFile* f, int64_t index);

// 把张量数据逐元素展开为 F32（支持 F32/F16/BF16/F64/I8/I16/I32/I64 与 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0）。
// 要求 load_data=true 且 n_elems >= 张量元素数。
bool gguf_tensor_to_f32(const GgufFile* f, const char* name, float* out, int64_t n_elems);

// 按名把张量数据载入已创建的张量：
//   - dst 为 F32：任意支持类型均可（自动转换/反量化）；
//   - 否则要求类型一致（原样拷贝）。
// dst 必须连续、元素数一致；无数据/名字不存在时返回 false 并打印错误。
bool gguf_load_tensor(const GgufFile* f, const char* name, Tensor* dst);

// 按文件中的形状在 ctx 中创建张量（默认 F32，量化源类型可覆盖为 F32）。
// 创建后需分配 buffer（buffer_alloc_ctx_tensors）再调用 gguf_load_tensor。
Tensor* gguf_new_tensor(Context* ctx, const GgufFile* f, const char* name, Type type = TYPE_F32);

// 结构自检（写后校验/外部文件检查）：重新解析并验证头/版本/KV/张量信息/对齐/offset 连续
// 以及每个张量数据都在文件范围内（GGUF 无校验和，仅结构与范围校验）。成功打印摘要并返回 true。
bool gguf_validate(const char* path);

// ---------------------------------------------------------------- 写

GgufWriter* gguf_writer_new();
void        gguf_writer_free(GgufWriter* w);

// 元数据设置（同名 key 覆盖；key 必须非空）
void gguf_writer_set_arch(GgufWriter* w, const char* arch);  // general.architecture
void gguf_writer_set_u8(GgufWriter* w, const char* key, uint8_t v);
void gguf_writer_set_i8(GgufWriter* w, const char* key, int8_t v);
void gguf_writer_set_u16(GgufWriter* w, const char* key, uint16_t v);
void gguf_writer_set_i16(GgufWriter* w, const char* key, int16_t v);
void gguf_writer_set_u32(GgufWriter* w, const char* key, uint32_t v);
void gguf_writer_set_i32(GgufWriter* w, const char* key, int32_t v);
void gguf_writer_set_f32(GgufWriter* w, const char* key, float v);
void gguf_writer_set_bool(GgufWriter* w, const char* key, bool v);
void gguf_writer_set_u64(GgufWriter* w, const char* key, uint64_t v);
void gguf_writer_set_i64(GgufWriter* w, const char* key, int64_t v);
void gguf_writer_set_f64(GgufWriter* w, const char* key, double v);
void gguf_writer_set_str(GgufWriter* w, const char* key, const char* value);
void gguf_writer_set_arr_i32(GgufWriter* w, const char* key, const int32_t* v, int64_t n);
void gguf_writer_set_arr_i64(GgufWriter* w, const char* key, const int64_t* v, int64_t n);
void gguf_writer_set_arr_f32(GgufWriter* w, const char* key, const float* v, int64_t n);
void gguf_writer_set_arr_str(GgufWriter* w, const char* key, const char* const* v, int64_t n);

// 添加张量（只记录指针，写出前必须保持存活）：
//   - name 非空、唯一、长度 < MAX_NAME（63 字符 + NUL）；
//   - t 必须连续且数据已分配（buffer 或 host 内存均可）；
//   - out_type 允许：t->type（原样）、TYPE_F16（源 F32 时转换）；其余组合报错。
void gguf_writer_add_tensor(GgufWriter* w, const char* name, const Tensor* t,
                            Type out_type = TYPE_F32);
int64_t gguf_writer_tensor_count(const GgufWriter* w);

// 写出文件（恒为 v3、32 字节对齐）。失败返回 false 并打印错误。
bool gguf_writer_write(GgufWriter* w, const char* path);

} // namespace traincpp
