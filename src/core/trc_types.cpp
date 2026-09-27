// train.cpp - 数据类型表
#include "traincpp/trc_types.h"

#include "core/trc_impl.h"

#include <array>

namespace traincpp {

namespace {

struct TypeInfo {
    int64_t     blck  = 0;      // 0 表示未实现
    size_t      size  = 0;
    const char* name  = "unknown";
    bool        quant = false;
};

using TypeTable = std::array<TypeInfo, TYPE_COUNT>;

// 块大小/字节数与 ggml 保持一致（参考 ggml-common.h 与 ggml.c 的类型表）
const TypeTable& type_table() {
    static const TypeTable table = [] {
        TypeTable t{};
        t[TYPE_F32]  = {1, 4, "f32", false};
        t[TYPE_F16]  = {1, 2, "f16", false};
        t[TYPE_Q4_0] = {32, 18, "q4_0", true};
        t[TYPE_Q4_1] = {32, 20, "q4_1", true};
        t[TYPE_Q5_0] = {32, 22, "q5_0", true};
        t[TYPE_Q5_1] = {32, 24, "q5_1", true};
        t[TYPE_Q8_0] = {32, 34, "q8_0", true};
        t[TYPE_Q8_1] = {32, 36, "q8_1", true};
        t[TYPE_Q2_K] = {256, 84, "q2_K", true};
        t[TYPE_Q3_K] = {256, 110, "q3_K", true};
        t[TYPE_Q4_K] = {256, 144, "q4_K", true};
        t[TYPE_Q5_K] = {256, 176, "q5_K", true};
        t[TYPE_Q6_K] = {256, 210, "q6_K", true};
        t[TYPE_Q8_K] = {256, 292, "q8_K", true};
        t[TYPE_I8]   = {1, 1, "i8", false};
        t[TYPE_I16]  = {1, 2, "i16", false};
        t[TYPE_I32]  = {1, 4, "i32", false};
        t[TYPE_I64]  = {1, 8, "i64", false};
        t[TYPE_F64]  = {1, 8, "f64", false};
        t[TYPE_BF16] = {1, 2, "bf16", false};
        return t;
    }();
    return table;
}

const TypeInfo& get_type_info(Type type) {
    const int32_t v = (int32_t) type;
    TRC_ASSERT(v >= 0 && v < TYPE_COUNT, "非法的数据类型: %d", v);
    const TypeInfo& info = type_table()[v];
    TRC_ASSERT(info.blck != 0, "数据类型 %d 尚未实现（plan: M1.2/M1.6 补充）", v);
    return info;
}

} // namespace

int64_t type_blck_size(Type type) {
    const int32_t v = (int32_t) type;
    if (v < 0 || v >= TYPE_COUNT) {
        return 0;
    }
    return type_table()[v].blck;
}

size_t type_size(Type type) {
    const int32_t v = (int32_t) type;
    if (v < 0 || v >= TYPE_COUNT) {
        return 0;
    }
    return type_table()[v].size;
}

double type_sizef(Type type) {
    const TypeInfo& info = get_type_info(type);
    return (double) info.size / (double) info.blck;
}

size_t type_row_size(Type type, int64_t ne) {
    const TypeInfo& info = get_type_info(type);
    TRC_ASSERT(ne % info.blck == 0, "类型 %s 的行元素数 %lld 不是块大小 %lld 的整数倍", info.name,
               (long long) ne, (long long) info.blck);
    return (size_t) (ne / info.blck) * info.size;
}

const char* type_name(Type type) {
    const int32_t v = (int32_t) type;
    if (v < 0 || v >= TYPE_COUNT) {
        return "unknown";
    }
    return type_table()[v].name;
}

bool type_is_quantized(Type type) {
    const int32_t v = (int32_t) type;
    if (v < 0 || v >= TYPE_COUNT) {
        return false;
    }
    return type_table()[v].quant;
}

bool type_is_supported(Type type) {
    switch (type) {
        case TYPE_F32:
        case TYPE_F16:
        case TYPE_BF16:
        case TYPE_F64:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_I64:
            return true;
        default:
            return false;
    }
}

} // namespace traincpp
