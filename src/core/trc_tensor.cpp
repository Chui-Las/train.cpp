// train.cpp - 张量实现
#include "traincpp/trc_tensor.h"

#include "core/trc_impl.h"
#include "traincpp/trc_context.h"

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <new>

namespace traincpp {

int tensor_n_dims(const Tensor* t) {
    int n_dims = 1;
    for (int i = 1; i < MAX_DIMS; ++i) {
        if (t->ne[i] > 1) {
            n_dims = i + 1;
        }
    }
    return n_dims;
}

int64_t tensor_nelements(const Tensor* t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

int64_t tensor_nrows(const Tensor* t) {
    return t->ne[1] * t->ne[2] * t->ne[3];
}

size_t tensor_nbytes(const Tensor* t) {
    const int64_t blck = type_blck_size(t->type);
    TRC_ASSERT(blck > 0, "tensor_nbytes: 数据类型 %d 未实现", (int) t->type);
    TRC_ASSERT(t->ne[0] % blck == 0, "tensor_nbytes: ne[0]=%lld 不是块大小 %lld 的整数倍",
               (long long) t->ne[0], (long long) blck);

    size_t nbytes = (size_t) (t->ne[0] / blck) * type_size(t->type);
    if (t->ne[1] > 1) {
        nbytes += (size_t) (t->ne[1] - 1) * t->nb[1];
    }
    if (t->ne[2] > 1) {
        nbytes += (size_t) (t->ne[2] - 1) * t->nb[2];
    }
    if (t->ne[3] > 1) {
        nbytes += (size_t) (t->ne[3] - 1) * t->nb[3];
    }
    return nbytes;
}

size_t tensor_element_size(const Tensor* t) {
    return type_size(t->type);
}

size_t tensor_row_size(const Tensor* t) {
    return type_row_size(t->type, t->ne[0]);
}

bool tensor_is_contiguous(const Tensor* t) {
    const int64_t blck = type_blck_size(t->type);
    TRC_ASSERT(blck > 0, "tensor_is_contiguous: 数据类型 %d 未实现", (int) t->type);

    size_t nb_expected[MAX_DIMS] = {0};
    nb_expected[0] = type_size(t->type);
    for (int i = 1; i < MAX_DIMS; ++i) {
        TRC_ASSERT(t->ne[i - 1] % blck == 0, "tensor_is_contiguous: 非法形状");
        nb_expected[i] = nb_expected[i - 1] * (size_t) (t->ne[i - 1] / blck);
    }
    for (int i = 0; i < MAX_DIMS; ++i) {
        if (t->nb[i] != nb_expected[i]) {
            return false;
        }
    }
    return true;
}

bool tensor_are_same_shape(const Tensor* a, const Tensor* b) {
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

bool tensor_can_repeat(const Tensor* b, const Tensor* a) {
    for (int i = 0; i < MAX_DIMS; ++i) {
        if (b->ne[i] != a->ne[i] && b->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

bool tensor_is_view(const Tensor* t) {
    return t->view_src != nullptr;
}

size_t tensor_offset_linear(const Tensor* t, int64_t i) {
    size_t offset = 0;
    for (int d = 0; d < MAX_DIMS; ++d) {
        const int64_t idx = i % t->ne[d];
        i /= t->ne[d];
        offset += (size_t) idx * t->nb[d];
    }
    return offset;
}

namespace {

// 上游偏移：线性下标 i 在张量中的字节偏移（不取模，用于越界检查）
size_t tensor_last_element_offset(const Tensor* t) {
    return (size_t) (t->ne[0] - 1) * t->nb[0] + (size_t) (t->ne[1] - 1) * t->nb[1] +
           (size_t) (t->ne[2] - 1) * t->nb[2] + (size_t) (t->ne[3] - 1) * t->nb[3];
}

// 张量元数据创建。nb 为 nullptr 时按连续行主序计算；alloc_data 为 false 时不分配数据。
Tensor* tensor_new_raw(Context* ctx, Type type, int n_dims, const int64_t* ne, const size_t* nb,
                       bool alloc_data) {
    TRC_ASSERT(ctx != nullptr, "new_tensor: ctx 为空");
    TRC_ASSERT(n_dims >= 1 && n_dims <= MAX_DIMS, "new_tensor: 维度数非法 %d", n_dims);
    TRC_ASSERT(type_is_supported(type), "new_tensor: 数据类型 %s 不支持分配（一期仅支持基本类型）",
               type_name(type));

    const int64_t blck = type_blck_size(type);
    TRC_ASSERT(blck > 0, "new_tensor: 数据类型 %d 未实现", (int) type);

    for (int i = 0; i < n_dims; ++i) {
        TRC_ASSERT(ne[i] > 0, "new_tensor: 第 %d 维大小必须大于 0", i);
    }

    Tensor* t = (Tensor*) context_alloc(ctx, sizeof(Tensor), MEM_ALIGN);
    // arena 内存未经过构造函数，使用 placement new 初始化默认值
    new (t) Tensor();

    t->type = type;
    for (int i = 0; i < MAX_DIMS; ++i) {
        t->ne[i] = (i < n_dims) ? ne[i] : 1;
    }

    TRC_ASSERT(t->ne[0] % blck == 0, "new_tensor: ne[0]=%lld 不是块大小 %lld 的整数倍",
               (long long) t->ne[0], (long long) blck);

    if (nb != nullptr) {
        for (int i = 0; i < MAX_DIMS; ++i) {
            t->nb[i] = nb[i];
        }
    } else {
        t->nb[0] = type_size(type);
        for (int i = 1; i < MAX_DIMS; ++i) {
            t->nb[i] = t->nb[i - 1] * (size_t) (t->ne[i - 1] / blck);
        }
    }

    if (alloc_data && !context_no_alloc(ctx)) {
        t->data = context_alloc(ctx, tensor_nbytes(t), MEM_ALIGN);
    }

    ctx->tensors.push_back(t);
    return t;
}

// 创建视图：沿用指定 nb，校验不越界；no_alloc=false 时直接解析数据指针
Tensor* tensor_new_view(Context* ctx, Tensor* a, Op op, int n_dims, const int64_t* ne, const size_t* nb,
                        size_t offset) {
    TRC_ASSERT(a != nullptr, "视图源张量为空");
    TRC_ASSERT(!type_is_quantized(a->type), "视图暂不支持量化类型 (%s)", type_name(a->type));

    Tensor* v = tensor_new_raw(ctx, a->type, n_dims, ne, nb, /*alloc_data=*/false);
    v->op          = op;
    v->view_src    = a;
    v->view_offs   = offset;
    // 与 ggml 一致：视图也把源张量登记为 src[0]，保证图遍历能访问到源计算节点
    v->src[0]      = a;
    v->nsrc        = 1;

    // 越界检查：与源张量的实际元素跨度比较（不能用 tensor_nbytes，非连续视图的跨度会被低估）
    const size_t src_extent = tensor_last_element_offset(a) + type_size(a->type);
    const size_t end = offset + tensor_last_element_offset(v) + type_size(v->type);
    TRC_ASSERT(end <= src_extent, "视图越界：offset=%zu end=%zu 源跨度=%zu", offset, end, src_extent);

    if (!context_no_alloc(ctx)) {
        TRC_ASSERT(a->data != nullptr, "视图源张量数据为空");
        v->data = (uint8_t*) a->data + offset;
    }
    return v;
}

} // namespace

Tensor* new_tensor_nd(Context* ctx, Type type, int n_dims, const int64_t* ne) {
    return tensor_new_raw(ctx, type, n_dims, ne, nullptr, /*alloc_data=*/true);
}

Tensor* new_tensor_1d(Context* ctx, Type type, int64_t ne0) {
    const int64_t ne[MAX_DIMS] = {ne0, 1, 1, 1};
    return new_tensor_nd(ctx, type, 1, ne);
}

Tensor* new_tensor_2d(Context* ctx, Type type, int64_t ne0, int64_t ne1) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, 1, 1};
    return new_tensor_nd(ctx, type, 2, ne);
}

Tensor* new_tensor_3d(Context* ctx, Type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, ne2, 1};
    return new_tensor_nd(ctx, type, 3, ne);
}

Tensor* new_tensor_4d(Context* ctx, Type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, ne2, ne3};
    return new_tensor_nd(ctx, type, 4, ne);
}

Tensor* tensor_dup_meta(Context* ctx, const Tensor* src) {
    Tensor* t = new_tensor_nd(ctx, src->type, MAX_DIMS, src->ne);
    std::memcpy(t->name, src->name, MAX_NAME);
    // 注意：不复制 flags —— 参数标志不能传染给算子结果
    return t;
}

Tensor* new_scalar_const(Context* ctx, float value) {
    Tensor* t = new_tensor_1d(ctx, TYPE_F32, 1);
    tensor_set_name(t, "const %.6g", (double) value);
    if (t->data != nullptr) {
        *(float*) t->data = value;  // no_alloc=false：数据已在 arena
    } else {
        Context::PendingFill fill;
        fill.tensor = t;
        fill.kind   = Context::PendingFill::CONST;
        fill.a      = value;
        ctx->pending_fills.push_back(fill);
    }
    return t;
}

// ---------------- 视图与形状 ----------------

Tensor* view_1d(Context* ctx, Tensor* a, int64_t ne0, size_t offset) {
    const int64_t ne[MAX_DIMS] = {ne0, 1, 1, 1};
    return tensor_new_view(ctx, a, OP_VIEW, 1, ne, a->nb, offset);
}

Tensor* view_2d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, size_t offset) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, 1, 1};
    return tensor_new_view(ctx, a, OP_VIEW, 2, ne, a->nb, offset);
}

Tensor* view_3d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t offset) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, ne2, 1};
    return tensor_new_view(ctx, a, OP_VIEW, 3, ne, a->nb, offset);
}

Tensor* view_4d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t offset) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, ne2, ne3};
    return tensor_new_view(ctx, a, OP_VIEW, 4, ne, a->nb, offset);
}

namespace {

Tensor* reshape_impl(Context* ctx, Tensor* a, int n_dims, const int64_t* ne) {
    TRC_ASSERT(a != nullptr, "reshape: 输入为空");
    TRC_ASSERT(tensor_is_contiguous(a), "reshape 要求源张量连续（与 ggml 一致）");

    int64_t total = 1;
    for (int i = 0; i < n_dims; ++i) {
        total *= ne[i];
    }
    TRC_ASSERT(total == tensor_nelements(a), "reshape: 元素总数不一致 (%lld vs %lld)",
               (long long) total, (long long) tensor_nelements(a));

    return tensor_new_view(ctx, a, OP_RESHAPE, n_dims, ne, nullptr, 0);
}

} // namespace

Tensor* reshape_1d(Context* ctx, Tensor* a, int64_t ne0) {
    const int64_t ne[MAX_DIMS] = {ne0, 1, 1, 1};
    return reshape_impl(ctx, a, 1, ne);
}

Tensor* reshape_2d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, 1, 1};
    return reshape_impl(ctx, a, 2, ne);
}

Tensor* reshape_3d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, ne2, 1};
    return reshape_impl(ctx, a, 3, ne);
}

Tensor* reshape_4d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const int64_t ne[MAX_DIMS] = {ne0, ne1, ne2, ne3};
    return reshape_impl(ctx, a, 4, ne);
}

Tensor* permute(Context* ctx, Tensor* a, int axis0, int axis1, int axis2, int axis3) {
    TRC_ASSERT(a != nullptr, "permute: 输入为空");

    const int axes[MAX_DIMS] = {axis0, axis1, axis2, axis3};
    bool seen[MAX_DIMS] = {false, false, false, false};
    for (int i = 0; i < MAX_DIMS; ++i) {
        TRC_ASSERT(axes[i] >= 0 && axes[i] < MAX_DIMS, "permute: 轴下标非法 %d", axes[i]);
        TRC_ASSERT(!seen[axes[i]], "permute: 轴下标重复 %d", axes[i]);
        seen[axes[i]] = true;
    }

    int64_t ne[MAX_DIMS];
    size_t  nb[MAX_DIMS];
    for (int i = 0; i < MAX_DIMS; ++i) {
        ne[i] = a->ne[axes[i]];
        nb[i] = a->nb[axes[i]];
    }
    Tensor* result = tensor_new_view(ctx, a, OP_PERMUTE, MAX_DIMS, ne, nb, 0);
    // 保存轴序（反向需要求逆置换，与 ggml 一致存于 op_params）
    for (int i = 0; i < MAX_DIMS; ++i) {
        result->op_params[i] = axes[i];
    }
    return result;
}

Tensor* transpose(Context* ctx, Tensor* a) {
    return permute(ctx, a, 1, 0, 2, 3);
}

Tensor* acc(Context* ctx, Tensor* a, Tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset,
            bool inplace) {
    TRC_ASSERT(a != nullptr && b != nullptr, "acc: 输入为空");
    TRC_ASSERT(a->type == b->type, "acc: 输入类型不一致");
    TRC_ASSERT(tensor_is_contiguous(a), "acc: 目标张量必须连续");
    TRC_ASSERT(tensor_is_contiguous(b), "acc: 源张量必须连续");
    TRC_ASSERT(type_is_supported(a->type), "acc: 类型不支持");

    // 写入区域越界检查（nb0 = 类型元素大小）
    const size_t nb0 = type_size(a->type);
    size_t last = (size_t) (b->ne[0] - 1) * nb0;
    last += b->ne[1] > 1 ? (size_t) (b->ne[1] - 1) * nb1 : 0;
    last += b->ne[2] > 1 ? (size_t) (b->ne[2] - 1) * nb2 : 0;
    last += b->ne[3] > 1 ? (size_t) (b->ne[3] - 1) * nb3 : 0;
    TRC_ASSERT(offset + last + nb0 <= tensor_nbytes(a), "acc: 写入区域越界 (offset=%zu last=%zu nbytes=%zu)",
               offset, last, tensor_nbytes(a));

    Tensor* result = nullptr;
    if (inplace) {
        // 与 ggml 一致：结果为 a 的完整视图（形状/步长同 a），写入区域由 op_params 指定
        result = tensor_new_view(ctx, a, OP_ACC, MAX_DIMS, a->ne, a->nb, 0);
    } else {
        result = tensor_dup_meta(ctx, a);
        result->op = OP_ACC;
    }
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;

    // 与 ggml 一致（ggml_acc_impl）：op_params 依次为 int32 的 nb1、nb2、nb3、offset、inplace
    // ⚠️ 不得改回 size_t 布局：inplace 标志曾写在字节 8，覆盖 nb2 低字节（M1.5d 修复）
    TRC_ASSERT(nb1 <= (size_t) INT32_MAX && nb2 <= (size_t) INT32_MAX && nb3 <= (size_t) INT32_MAX &&
                   offset <= (size_t) INT32_MAX,
               "acc: nb/offset 超出 int32（与 ggml 一致）");
    const int32_t params[5] = {(int32_t) nb1, (int32_t) nb2, (int32_t) nb3, (int32_t) offset,
                               inplace ? 1 : 0};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

void tensor_set_name(Tensor* t, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(t->name, MAX_NAME, fmt, args);
    va_end(args);
}

float* tensor_data_f32(Tensor* t) {
    TRC_ASSERT(t->type == TYPE_F32, "tensor_data_f32: 张量类型为 %s，不是 f32", type_name(t->type));
    return (float*) t->data;
}

const float* tensor_data_f32(const Tensor* t) {
    TRC_ASSERT(t->type == TYPE_F32, "tensor_data_f32: 张量类型为 %s，不是 f32", type_name(t->type));
    return (const float*) t->data;
}

} // namespace traincpp
