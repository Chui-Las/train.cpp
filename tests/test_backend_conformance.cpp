// train.cpp - 后端 conformance：同一计算图在 CPU 与 Vulkan 上结果对照（M1.4b）
//
// 每个用例只写一遍构图代码，分别在参考后端（CPU）与目标后端（Vulkan）执行并逐元素比较。
// 无 Vulkan 设备（或未启用 TRC_VULKAN）时自动跳过。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

using namespace traincpp;

namespace {

struct BuiltGraph {
    Tensor* out = nullptr;
    std::vector<std::pair<Tensor*, std::vector<float>>>   inputs;
    std::vector<std::pair<Tensor*, std::vector<int32_t>>> int_inputs;
};

using BuildFn = std::function<BuiltGraph(Context*, Graph*)>;

Device* find_vulkan_device() {
    Device* dev = device_by_type(DeviceType::GPU);
    if (dev == nullptr) {
        dev = device_by_type(DeviceType::IGPU);
    }
    return dev;
}

// 在指定后端上构图、分配、喂数据、计算，返回输出拷贝；目标后端不支持整图时标记跳过
struct RunResult {
    std::vector<float> out;
    bool               skipped = false;
    const char*        missing = nullptr;
};

RunResult run_on(Device* dev, const BuildFn& build) {
    Context* ctx = context_new(1 << 24);
    Graph*   g   = graph_new(ctx);

    BuiltGraph bg  = build(ctx, g);
    Buffer*    buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);

    for (const auto& in : bg.inputs) {
        tensor_set(in.first, in.second.data(), 0, in.second.size() * sizeof(float));
    }
    for (const auto& in : bg.int_inputs) {
        tensor_set(in.first, in.second.data(), 0, in.second.size() * sizeof(int32_t));
    }

    graph_build_forward_expand(ctx, g, bg.out);

    // 目标后端不支持整图时跳过（例如某算子只在 CPU 实现；不静默通过）
    if (const Tensor* bad = trc_test::first_unsupported_node(g, dev)) {
        RunResult r;
        r.skipped = true;
        r.missing = op_name(bad->op);
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
        return r;
    }

    dev->graph_compute(g);

    RunResult r;
    r.out.resize((size_t) tensor_nelements(bg.out));
    tensor_get(bg.out, r.out.data(), 0, r.out.size() * sizeof(float));

    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return r;
}

void compare(Device* ref, Device* dev, const char* name, const BuildFn& build) {
    const std::vector<float> want = run_on(ref, build).out;
    const RunResult          got  = run_on(dev, build);
    if (got.skipped) {
        std::printf("    [跳过] %-24s 后端不支持 %s\n", name, got.missing);
        return;
    }
    if (want.size() != got.out.size()) {
        trc_test::add_failure(std::string("conformance ") + name + ": 输出元素数不一致");
        return;
    }
    double  max_abs = 0;
    int64_t max_idx = 0;
    int64_t n_bad   = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double diff = std::fabs((double) got.out[i] - (double) want[i]);
        const double tol  = 1e-5 + 2e-4 * std::fabs((double) want[i]);
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = (int64_t) i;
        }
        if (diff > tol) {
            ++n_bad;
        }
    }
    if (n_bad != 0) {
        char msg[512];
        std::snprintf(msg, sizeof(msg),
                      "conformance %s 失败: 超差 %lld/%zu 元素, 最大绝对差 %.3g (下标 %lld: Vulkan %.6g, "
                      "CPU %.6g)",
                      name, (long long) n_bad, want.size(), max_abs, (long long) max_idx,
                      (double) got.out[(size_t) max_idx], (double) want[(size_t) max_idx]);
        trc_test::add_failure(msg);
        int shown = 0;
        for (size_t i = 0; i < want.size() && shown < 8; ++i) {
            const double diff = std::fabs((double) got.out[i] - (double) want[i]);
            if (diff > 1e-5 + 2e-4 * std::fabs((double) want[i])) {
                std::printf("        [%zu] Vulkan=%.6g CPU=%.6g\n", i, (double) got.out[i],
                            (double) want[i]);
                ++shown;
            }
        }
    } else {
        std::printf("    [ OK ] %-24s max_abs=%.3g\n", name, max_abs);
    }
}

std::vector<float> ramp(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = (float) ((int) (i % 17) - 8) * 0.25f;
    }
    return v;
}

std::vector<float> ramp_pos(size_t n) {
    std::vector<float> v = ramp(n);
    for (float& x : v) {
        x = std::fabs(x) + 0.1f;
    }
    return v;
}

} // namespace

TRC_TEST(conformance_binary) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // 同形状 + 广播（b 沿 ne1 广播）
    compare(cpu, dev, "add_broadcast", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 3, 4);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 3, 1);
        Tensor* out = add(ctx, a, b);
        return BuiltGraph{out, {{a, ramp(12)}, {b, ramp(3)}}};
    });

    // 二元链：sub(mul(a,b), div(a,b2))，b2 恒正
    compare(cpu, dev, "binary_chain", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 4, 3);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 4, 3);
        Tensor* b2  = new_tensor_2d(ctx, TYPE_F32, 4, 1);
        Tensor* out = sub(ctx, mul(ctx, a, b), div(ctx, a, b2));
        return BuiltGraph{out, {{a, ramp(12)}, {b, ramp(12)}, {b2, ramp_pos(4)}}};
    });

    // add1：b 为 1 元素标量（沿所有维广播）
    compare(cpu, dev, "add1_scalar", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 5, 3);
        Tensor* b   = new_tensor_1d(ctx, TYPE_F32, 1);
        Tensor* out = add1(ctx, a, b);
        return BuiltGraph{out, {{a, ramp(15)}, {b, {0.75f}}}};
    });

    // 非连续视图输入：base[5,3] 转置为 [3,5]
    compare(cpu, dev, "add_transposed_view", [](Context* ctx, Graph*) {
        Tensor* base = new_tensor_2d(ctx, TYPE_F32, 5, 3);
        Tensor* b    = new_tensor_2d(ctx, TYPE_F32, 3, 5);
        Tensor* v    = transpose(ctx, base);
        Tensor* out  = add(ctx, v, b);
        return BuiltGraph{out, {{base, ramp(15)}, {b, ramp(15)}}};
    });
}

TRC_TEST(conformance_unary) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // 激活一组：relu / sigmoid / tanh / silu
    compare(cpu, dev, "unary_activations", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 6, 5);
        Tensor* out = add(ctx, add(ctx, relu(ctx, x), sigmoid(ctx, x)),
                          add(ctx, tanh_op(ctx, x), silu(ctx, x)));
        return BuiltGraph{out, {{x, ramp(30)}}};
    });

    // 超越函数：sqrt/log（正值输入）+ sin/cos
    compare(cpu, dev, "unary_transcendental", [](Context* ctx, Graph*) {
        Tensor* p   = new_tensor_2d(ctx, TYPE_F32, 7, 3);
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 7, 3);
        Tensor* out = add(ctx, add(ctx, sqrt_op(ctx, p), log_op(ctx, p)),
                          add(ctx, sin_op(ctx, x), cos_op(ctx, x)));
        return BuiltGraph{out, {{p, ramp_pos(21)}, {x, ramp(21)}}};
    });

    // gelu 家族
    compare(cpu, dev, "unary_gelu_family", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, 32);
        Tensor* out = add(ctx, add(ctx, gelu(ctx, x), gelu_erf(ctx, x)), erf_op(ctx, x));
        return BuiltGraph{out, {{x, ramp(32)}}};
    });

    // 带参数一元 + 分段函数
    compare(cpu, dev, "unary_params_piecewise", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, 48);
        Tensor* out = add(ctx,
                          add(ctx, leaky_relu(ctx, x, 0.1f), clamp(ctx, x, -0.5f, 0.5f)),
                          add(ctx, scale(ctx, hardswish(ctx, x), 2.5f), softplus(ctx, x)));
        return BuiltGraph{out, {{x, ramp(48)}}};
    });

    // sgn / step / neg / abs / sqr / exp
    compare(cpu, dev, "unary_misc", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, 40);
        Tensor* out = add(ctx, add(ctx, sgn(ctx, x), step(ctx, x)),
                          add(ctx, add(ctx, neg(ctx, abs_op(ctx, x)), sqr(ctx, x)), exp_op(ctx, x)));
        return BuiltGraph{out, {{x, ramp(40)}}};
    });
}

TRC_TEST(conformance_reductions_norm_softmax) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // sum / sum_rows / mean
    compare(cpu, dev, "sum_all", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 4, 3, 2);
        Tensor* out = sum(ctx, x);
        return BuiltGraph{out, {{x, ramp(24)}}};
    });
    compare(cpu, dev, "sum_rows", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 5, 3, 2);
        Tensor* out = sum_rows(ctx, x);
        return BuiltGraph{out, {{x, ramp(30)}}};
    });
    compare(cpu, dev, "mean", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 7, 4);
        Tensor* out = mean(ctx, x);
        return BuiltGraph{out, {{x, ramp(28)}}};
    });
    // 转置视图上的行归约（nb 非连续）
    compare(cpu, dev, "sum_rows_transposed", [](Context* ctx, Graph*) {
        Tensor* base = new_tensor_2d(ctx, TYPE_F32, 5, 4);
        Tensor* v    = transpose(ctx, base);      // [4,5]
        Tensor* out  = sum_rows(ctx, v);
        return BuiltGraph{out, {{base, ramp(20)}}};
    });

    // norm / rms_norm（行长为 ne0）
    compare(cpu, dev, "norm", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 6, 4, 3);
        Tensor* out = norm(ctx, x, 1e-5f);
        return BuiltGraph{out, {{x, ramp(72)}}};
    });
    compare(cpu, dev, "rms_norm", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 6, 4, 3);
        Tensor* out = rms_norm(ctx, x, 1e-5f);
        return BuiltGraph{out, {{x, ramp(72)}}};
    });
    compare(cpu, dev, "group_norm", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_4d(ctx, TYPE_F32, 3, 2, 4, 2);
        Tensor* out = group_norm(ctx, x, 2, 1e-5f);
        return BuiltGraph{out, {{x, ramp(48)}}};
    });

    // soft_max / soft_max_ext（mask + scale + ALiBi）
    compare(cpu, dev, "soft_max", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 8, 3, 2);
        Tensor* out = soft_max(ctx, x);
        return BuiltGraph{out, {{x, ramp(48)}}};
    });
    compare(cpu, dev, "soft_max_ext_mask_scale", [](Context* ctx, Graph*) {
        Tensor* x    = new_tensor_4d(ctx, TYPE_F32, 8, 2, 2, 1);
        Tensor* mask = new_tensor_4d(ctx, TYPE_F32, 8, 2, 1, 1);
        Tensor* out  = soft_max_ext(ctx, x, mask, 0.5f, 0.0f);
        return BuiltGraph{out, {{x, ramp(32)}, {mask, ramp(16)}}};
    });
    compare(cpu, dev, "soft_max_ext_alibi", [](Context* ctx, Graph*) {
        Tensor* x    = new_tensor_4d(ctx, TYPE_F32, 4, 2, 2, 1);
        Tensor* mask = new_tensor_4d(ctx, TYPE_F32, 4, 2, 1, 1);
        Tensor* out  = soft_max_ext(ctx, x, mask, 1.0f, 8.0f);
        return BuiltGraph{out, {{x, ramp(16)}, {mask, ramp(8)}}};
    });
}

TRC_TEST(conformance_loss) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // cross_entropy_loss 前向（软标签，多行）
    compare(cpu, dev, "cross_entropy_loss", [](Context* ctx, Graph*) {
        Tensor* logits = new_tensor_3d(ctx, TYPE_F32, 5, 3, 2);  // nc=5, nr=6
        Tensor* target = new_tensor_3d(ctx, TYPE_F32, 5, 3, 2);
        Tensor* out    = cross_entropy_loss(ctx, logits, target);
        return BuiltGraph{out, {{logits, ramp(30)}, {target, ramp_pos(30)}}};
    });

    // one-hot 目标 + 较大的 nc（检查行内循环）
    compare(cpu, dev, "cross_entropy_loss_onehot", [](Context* ctx, Graph*) {
        Tensor* logits = new_tensor_2d(ctx, TYPE_F32, 33, 4);  // nc=33, nr=4
        Tensor* target = new_tensor_2d(ctx, TYPE_F32, 33, 4);
        std::vector<float> t(132, 0.0f);
        for (int r = 0; r < 4; ++r) {
            t[(size_t) (r * 33 + (r * 7) % 33)] = 1.0f;
        }
        Tensor* out = cross_entropy_loss(ctx, logits, target);
        return BuiltGraph{out, {{logits, ramp(132)}, {target, t}}};
    });

    // cross_entropy_loss_back：dA = (softmax(logits) - target) * grad / nr
    compare(cpu, dev, "cross_entropy_loss_back", [](Context* ctx, Graph*) {
        Tensor* grad   = new_tensor_1d(ctx, TYPE_F32, 1);
        Tensor* logits = new_tensor_3d(ctx, TYPE_F32, 6, 2, 2);  // nc=6, nr=4
        Tensor* target = new_tensor_3d(ctx, TYPE_F32, 6, 2, 2);
        Tensor* out    = cross_entropy_loss_back(ctx, grad, logits, target);
        return BuiltGraph{out, {{grad, {2.5f}}, {logits, ramp(24)}, {target, ramp_pos(24)}}};
    });
}

TRC_TEST(conformance_copy_and_shape) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // cont：物化转置视图
    compare(cpu, dev, "cont_transposed", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 5, 3);
        Tensor* out = cont(ctx, transpose(ctx, x));
        return BuiltGraph{out, {{x, ramp(15)}}};
    });

    // dup
    compare(cpu, dev, "dup", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 4, 3);
        Tensor* out = dup(ctx, x);
        return BuiltGraph{out, {{x, ramp(12)}}};
    });

    // pad（每维末尾补零）
    compare(cpu, dev, "pad", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 3, 2, 2);
        Tensor* out = pad(ctx, x, 2, 1, 1, 0);
        return BuiltGraph{out, {{x, ramp(12)}}};
    });

    // concat（dim=0 与 dim=1）
    compare(cpu, dev, "concat_dim0", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 2, 3);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 3, 3);
        Tensor* out = concat(ctx, a, b, 0);
        return BuiltGraph{out, {{a, ramp(6)}, {b, ramp(9)}}};
    });
    compare(cpu, dev, "concat_dim1", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 4, 1);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 4, 2);
        Tensor* out = concat(ctx, a, b, 1);
        return BuiltGraph{out, {{a, ramp(4)}, {b, ramp(8)}}};
    });
}

TRC_TEST(conformance_indexing) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // get_rows：表 [4,5,2,1]，索引 [3,2,1,1]
    compare(cpu, dev, "get_rows", [](Context* ctx, Graph*) {
        Tensor* table = new_tensor_4d(ctx, TYPE_F32, 4, 5, 2, 1);
        Tensor* idx   = new_tensor_4d(ctx, TYPE_I32, 3, 2, 1, 1);
        Tensor* out   = get_rows(ctx, table, idx);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({table, ramp(40)});
        bg.int_inputs.push_back({idx, {0, 4, 2, 1, 3, 0}});
        return bg;
    });

    // get_rows 表为转置视图（M2.1b：CPU 曾忽略表步长 nb[0] 静默算错；两后端必须一致）
    compare(cpu, dev, "get_rows_transposed_table", [](Context* ctx, Graph*) {
        Tensor* base  = new_tensor_2d(ctx, TYPE_F32, 4, 3);
        Tensor* table = transpose(ctx, base);  // 表 [3,4]：dim=3、vocab=4
        Tensor* idx   = new_tensor_4d(ctx, TYPE_I32, 2, 1, 1, 1);
        Tensor* out   = get_rows(ctx, table, idx);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({base, ramp(12)});
        bg.int_inputs.push_back({idx, {2, 0}});
        return bg;
    });

    // get_rows_back：梯度 [4,3]，索引 [3,1]，表 [4,5]（仅取形状）
    compare(cpu, dev, "get_rows_back", [](Context* ctx, Graph*) {
        Tensor* grad  = new_tensor_4d(ctx, TYPE_F32, 4, 3, 1, 1);
        Tensor* idx   = new_tensor_4d(ctx, TYPE_I32, 3, 1, 1, 1);
        Tensor* table = new_tensor_4d(ctx, TYPE_F32, 4, 5, 1, 1);
        Tensor* out   = get_rows_back(ctx, grad, idx, table);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({grad, ramp(12)});
        bg.int_inputs.push_back({idx, {2, 0, 4}});
        return bg;
    });

    // set_rows：目标 [4,6,1,1]，值 [4,2,1,1]，索引 [2,1]
    compare(cpu, dev, "set_rows", [](Context* ctx, Graph*) {
        Tensor* dest = new_tensor_4d(ctx, TYPE_F32, 4, 6, 1, 1);
        Tensor* vals = new_tensor_4d(ctx, TYPE_F32, 4, 2, 1, 1);
        Tensor* idx  = new_tensor_4d(ctx, TYPE_I32, 2, 1, 1, 1);
        Tensor* out  = set_rows(ctx, dest, vals, idx);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({dest, ramp(24)});
        // 两行取值刻意不同，避免与 dest 原值巧合掩盖问题
        bg.inputs.push_back({vals, {-2.0f, -1.75f, -1.5f, -1.25f, 9.0f, 8.0f, 7.0f, 6.0f}});
        bg.int_inputs.push_back({idx, {3, 1}});
        return bg;
    });
}

TRC_TEST(conformance_matmul) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // 基础：a=[k=5,m=3] * b=[k=5,n=4] -> [3,4]
    compare(cpu, dev, "mul_mat_basic", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 5, 3);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 5, 4);
        Tensor* out = mul_mat(ctx, a, b);
        return BuiltGraph{out, {{a, ramp(15)}, {b, ramp(20)}}};
    });

    // a 为转置视图：base [m=3,k=5] 转置为 [k=5,m=3]（非连续 nb）
    compare(cpu, dev, "mul_mat_transposed_a", [](Context* ctx, Graph*) {
        Tensor* base = new_tensor_2d(ctx, TYPE_F32, 3, 5);
        Tensor* b    = new_tensor_2d(ctx, TYPE_F32, 5, 4);
        Tensor* a    = transpose(ctx, base);  // [5,3]
        Tensor* out  = mul_mat(ctx, a, b);
        return BuiltGraph{out, {{base, ramp(15)}, {b, ramp(20)}}};
    });

    // b 为转置视图：base [n=4,k=5] 转置为 [k=5,n=4]
    compare(cpu, dev, "mul_mat_transposed_b", [](Context* ctx, Graph*) {
        Tensor* a    = new_tensor_2d(ctx, TYPE_F32, 5, 3);
        Tensor* base = new_tensor_2d(ctx, TYPE_F32, 4, 5);
        Tensor* b    = transpose(ctx, base);  // [5,4]
        Tensor* out  = mul_mat(ctx, a, b);
        return BuiltGraph{out, {{a, ramp(15)}, {base, ramp(20)}}};
    });

    // 跨分块边界（shader 分块 BM=BN=64、BK=16）：k=33、m=70、n=66
    compare(cpu, dev, "mul_mat_tile_boundary", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 33, 70);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 33, 66);
        Tensor* out = mul_mat(ctx, a, b);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(33 * 70)});
        bg.inputs.push_back({b, ramp(33 * 66)});
        return bg;
    });

    // batch：b 有 2x3 个平面，a 单平面广播；输出 [m,n,2,3]
    compare(cpu, dev, "mul_mat_batch", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_4d(ctx, TYPE_F32, 5, 3, 1, 1);
        Tensor* b   = new_tensor_4d(ctx, TYPE_F32, 5, 4, 2, 3);
        Tensor* out = mul_mat(ctx, a, b);
        TRC_EXPECT_EQ(out->ne[2], 2);
        TRC_EXPECT_EQ(out->ne[3], 3);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(15)});
        bg.inputs.push_back({b, ramp(5 * 4 * 6)});
        return bg;
    });

    // batch 整除广播：a2=2、b2=4 → 平面 i2 用 a 的平面 (i2 % 2)
    compare(cpu, dev, "mul_mat_batch_bcast_div", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_4d(ctx, TYPE_F32, 5, 3, 2, 1);
        Tensor* b   = new_tensor_4d(ctx, TYPE_F32, 5, 4, 4, 1);
        Tensor* out = mul_mat(ctx, a, b);
        TRC_EXPECT_EQ(out->ne[2], 4);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(5 * 3 * 2)});
        bg.inputs.push_back({b, ramp(5 * 4 * 4)});
        return bg;
    });

    // batch + 非连续视图：a 为 [m,k,a2,a3] 的转置，b 为 [n,k,b2,b3] 的转置
    compare(cpu, dev, "mul_mat_batch_transposed", [](Context* ctx, Graph*) {
        Tensor* abase = new_tensor_4d(ctx, TYPE_F32, 3, 5, 2, 2);  // [m=3,k=5,2,2]
        Tensor* bbase = new_tensor_4d(ctx, TYPE_F32, 4, 5, 2, 2);  // [n=4,k=5,2,2]
        Tensor* a     = transpose(ctx, abase);                    // [k=5,m=3,2,2]
        Tensor* b     = transpose(ctx, bbase);                    // [k=5,n=4,2,2]
        Tensor* out   = mul_mat(ctx, a, b);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({abase, ramp(3 * 5 * 4)});
        bg.inputs.push_back({bbase, ramp(4 * 5 * 4)});
        return bg;
    });

    // batch + 跨分块边界：k=33、m=70、n=66、b2=2、b3=2
    compare(cpu, dev, "mul_mat_batch_tile_boundary", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_4d(ctx, TYPE_F32, 33, 70, 1, 1);
        Tensor* b   = new_tensor_4d(ctx, TYPE_F32, 33, 66, 2, 2);
        Tensor* out = mul_mat(ctx, a, b);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(33 * 70)});
        bg.inputs.push_back({b, ramp(33 * 66 * 4)});
        return bg;
    });
}

TRC_TEST(conformance_convolution) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // im2col 1D（图像为转置视图，检验 nb 寻址）：kernel [KW=3,IC=2]、img [W=5,IC=2]
    compare(cpu, dev, "im2col_1d_view", [](Context* ctx, Graph*) {
        Tensor* k    = new_tensor_3d(ctx, TYPE_F32, 3, 2, 1);
        Tensor* base = new_tensor_2d(ctx, TYPE_F32, 2, 5);   // [IC=2, W=5]
        Tensor* img  = transpose(ctx, base);                 // [W=5, IC=2]
        Tensor* out  = im2col(ctx, k, img, 1, 0, 1, 0, 1, 0, false, TYPE_F32);
        return BuiltGraph{out, {{k, ramp(6)}, {base, ramp(10)}}};
    });

    // im2col 2D：kernel [KW=3,KH=3,IC=2]、img [W=6,H=5,IC=2]（N=1）
    compare(cpu, dev, "im2col_2d", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_4d(ctx, TYPE_F32, 3, 3, 2, 1);
        Tensor* img = new_tensor_4d(ctx, TYPE_F32, 6, 5, 2, 1);
        Tensor* out = im2col(ctx, k, img, 1, 1, 1, 1, 1, 1, true, TYPE_F32);
        return BuiltGraph{out, {{k, ramp(18)}, {img, ramp(60)}}};
    });

    // conv_1d（组合图：im2col + reshape + mul_mat + reshape）：kernel [3,2,2]、img [6,2,1]
    compare(cpu, dev, "conv_1d", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_3d(ctx, TYPE_F32, 3, 2, 2);
        Tensor* img = new_tensor_3d(ctx, TYPE_F32, 6, 2, 1);
        Tensor* out = conv_1d(ctx, k, img, 2, 1, 1);
        return BuiltGraph{out, {{k, ramp(12)}, {img, ramp(12)}}};
    });

    // conv_2d（组合图含 permute/cont）：kernel [3,3,2,2]、img [6,5,2,1]
    compare(cpu, dev, "conv_2d", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_4d(ctx, TYPE_F32, 3, 3, 2, 2);
        Tensor* img = new_tensor_4d(ctx, TYPE_F32, 6, 5, 2, 1);
        Tensor* out = conv_2d(ctx, k, img, 1, 1, 1, 1, 1, 1);
        return BuiltGraph{out, {{k, ramp(36)}, {img, ramp(60)}}};
    });

    // im2col_back 1D：grad [feat=6, OW=5]，图像 ne = {W=5, IC=2, N=1, 1}
    compare(cpu, dev, "im2col_back_1d", [](Context* ctx, Graph*) {
        Tensor*        grad   = new_tensor_2d(ctx, TYPE_F32, 6, 5);
        Tensor*        kernel = new_tensor_2d(ctx, TYPE_F32, 3, 2);
        const int64_t  ne[4]  = {5, 2, 1, 1};
        Tensor*        out    = im2col_back(ctx, grad, kernel, ne, 1, 0, 1, 0, 1, 0, false);
        return BuiltGraph{out, {{grad, ramp(30)}, {kernel, ramp(6)}}};
    });

    // im2col_back 2D：grad [feat=IC*KH*KW=18, OW=6, OH=5]，图像 ne = {6, 5, 2, 1}
    compare(cpu, dev, "im2col_back_2d", [](Context* ctx, Graph*) {
        Tensor*        grad   = new_tensor_4d(ctx, TYPE_F32, 18, 6, 5, 1);
        Tensor*        kernel = new_tensor_4d(ctx, TYPE_F32, 3, 3, 2, 1);
        const int64_t  ne[4]  = {6, 5, 2, 1};
        Tensor*        out    = im2col_back(ctx, grad, kernel, ne, 1, 1, 1, 1, 1, 1, true);
        return BuiltGraph{out, {{grad, ramp(540)}, {kernel, ramp(18)}}};
    });

    // col2im_1d：src [K*OC=4, T_in=5] -> [T_out=6, OC=2]
    compare(cpu, dev, "col2im_1d", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 4, 5);
        Tensor* out = col2im_1d(ctx, a, 1, 2, 0);
        return BuiltGraph{out, {{a, ramp(20)}}};
    });

    // conv_transpose_1d：kernel [K=3, Cout=2, Cin=2]、input [T_in=5, Cin=2]、s0=2
    compare(cpu, dev, "conv_transpose_1d", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_3d(ctx, TYPE_F32, 3, 2, 2);
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 5, 2);
        Tensor* out = conv_transpose_1d(ctx, k, x, 2, 0, 1);
        return BuiltGraph{out, {{k, ramp(12)}, {x, ramp(10)}}};
    });
}

// WeightNorm × 分组卷积（M3.1）：nn 层组合（weightnorm_forward + conv1d_forward_weight groups>1）
// CPU vs Vulkan；覆盖 w=g·v/‖v‖ 与逐组 view/cont/conv_1d/concat 的整条链
TRC_TEST(conformance_weightnorm_grouped) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    compare(cpu, dev, "weightnorm_grouped_conv1d", [](Context* ctx, Graph*) {
        Tensor* v = new_tensor_3d(ctx, TYPE_F32, 3, 2, 4);  // [K,IC/g,OC]（IC=4、groups=2）
        Tensor* g = new_tensor_3d(ctx, TYPE_F32, 1, 1, 4);  // [1,1,OC]
        WeightNorm wn{};
        weightnorm_init(ctx, &wn, v, g, "wn");
        Tensor* x = new_tensor_3d(ctx, TYPE_F32, 6, 4, 2);  // [W,IC,N]
        Conv1d cw{};
        cw.has_bias = false;
        cw.groups   = 2;
        cw.stride   = 1;
        cw.padding  = 1;
        cw.dilation = 1;
        BuiltGraph bg;
        bg.out = conv1d_forward_weight(ctx, &cw, weightnorm_forward(ctx, &wn), x);
        bg.inputs.push_back({v, ramp(24)});
        bg.inputs.push_back({g, ramp(4)});
        bg.inputs.push_back({x, ramp(48)});
        return bg;
    });
}

// pool_2d / conv_transpose_2d（M2.3g）：CPU vs Vulkan
// 注：pool_2d_back 仅有 CPU 实现（scatter-add 竞态），不在此对照
TRC_TEST(conformance_pool_transpose2d) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // pool_2d AVG + 填充（非对称核 3x2、步长 2）：a [6,5,C=2,N=1]
    compare(cpu, dev, "pool_2d_avg_pad", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_4d(ctx, TYPE_F32, 6, 5, 2, 1);
        Tensor* out = pool_2d(ctx, a, POOL_AVG, 3, 2, 2, 2, 1, 1);
        return BuiltGraph{out, {{a, ramp(60)}}};
    });

    // pool_2d MAX（步长 1 重叠 + 填充）
    compare(cpu, dev, "pool_2d_max", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_4d(ctx, TYPE_F32, 5, 4, 1, 1);
        Tensor* out = pool_2d(ctx, a, POOL_MAX, 2, 2, 1, 1, 1, 1);
        return BuiltGraph{out, {{a, ramp(20)}}};
    });

    // conv_transpose_2d stride=1：kernel [2,2,2,2]、input [4,3,2,1]
    compare(cpu, dev, "conv_transpose_2d_s1", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_4d(ctx, TYPE_F32, 2, 2, 2, 2);
        Tensor* x   = new_tensor_4d(ctx, TYPE_F32, 4, 3, 2, 1);
        Tensor* out = conv_transpose_2d(ctx, k, x, 1);
        return BuiltGraph{out, {{k, ramp(16)}, {x, ramp(24)}}};
    });

    // conv_transpose_2d stride=2（含 (ow-kw)%stride 过滤）：kernel [3,2,2,2]、input [3,2,2,1]
    compare(cpu, dev, "conv_transpose_2d_s2", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_4d(ctx, TYPE_F32, 3, 2, 2, 2);
        Tensor* x   = new_tensor_4d(ctx, TYPE_F32, 3, 2, 2, 1);
        Tensor* out = conv_transpose_2d(ctx, k, x, 2);
        return BuiltGraph{out, {{k, ramp(24)}, {x, ramp(12)}}};
    });

    // 批 N=2（覆盖 nb[3] 与逐 batch 寻址）
    compare(cpu, dev, "conv_transpose_2d_batch", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_4d(ctx, TYPE_F32, 2, 2, 2, 2);
        Tensor* x   = new_tensor_4d(ctx, TYPE_F32, 3, 2, 2, 2);
        Tensor* out = conv_transpose_2d(ctx, k, x, 2);
        return BuiltGraph{out, {{k, ramp(16)}, {x, ramp(24)}}};
    });
}

TRC_TEST(conformance_cast) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // F32 -> F16 -> F32 往返（Vulkan 仅实现 F32↔F16；设备无 16 位存储时本用例自动跳过）
    compare(cpu, dev, "cast_f16_roundtrip", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 7, 3);
        Tensor* out = cast(ctx, cast(ctx, x, TYPE_F16), TYPE_F32);
        return BuiltGraph{out, {{x, ramp(21)}}};
    });
}

TRC_TEST(conformance_norm_back) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // norm_back（LayerNorm 反向）：dy/x [6,4,3]
    compare(cpu, dev, "norm_back", [](Context* ctx, Graph*) {
        Tensor* dy  = new_tensor_3d(ctx, TYPE_F32, 6, 4, 3);
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 6, 4, 3);
        Tensor* out = norm_back(ctx, dy, x, 1e-5f);
        return BuiltGraph{out, {{dy, ramp(72)}, {x, ramp_pos(72)}}};
    });

    // rms_norm_back
    compare(cpu, dev, "rms_norm_back", [](Context* ctx, Graph*) {
        Tensor* dy  = new_tensor_3d(ctx, TYPE_F32, 6, 4, 3);
        Tensor* x   = new_tensor_3d(ctx, TYPE_F32, 6, 4, 3);
        Tensor* out = rms_norm_back(ctx, dy, x, 1e-5f);
        return BuiltGraph{out, {{dy, ramp(72)}, {x, ramp_pos(72)}}};
    });

    // group_norm_back：x [3,2,4,2]、2 组（通道维 ne2=4）
    compare(cpu, dev, "group_norm_back", [](Context* ctx, Graph*) {
        Tensor* dy  = new_tensor_4d(ctx, TYPE_F32, 3, 2, 4, 2);
        Tensor* x   = new_tensor_4d(ctx, TYPE_F32, 3, 2, 4, 2);
        Tensor* out = group_norm_back(ctx, dy, x, 2, 1e-5f);
        return BuiltGraph{out, {{dy, ramp(48)}, {x, ramp_pos(48)}}};
    });
}

TRC_TEST(conformance_shape_extra) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // repeat：a=[2,1] -> target [2,3]
    compare(cpu, dev, "repeat", [](Context* ctx, Graph*) {
        Tensor* a      = new_tensor_2d(ctx, TYPE_F32, 2, 1);
        Tensor* target = new_tensor_2d(ctx, TYPE_F32, 2, 3);
        Tensor* out    = repeat(ctx, a, target);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(2)});
        return bg;
    });

    // repeat_back：a=[2,3] -> b=[2,1]
    compare(cpu, dev, "repeat_back", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 2, 3);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 2, 1);
        Tensor* out = repeat_back(ctx, a, b);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(6)});
        return bg;
    });

    // repeat_back 整除归约：a=[2,4] -> b=[2,2]（沿 ne1 取模分 2 组求和）
    compare(cpu, dev, "repeat_back_div", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 2, 4);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 2, 2);
        Tensor* out = repeat_back(ctx, a, b);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(8)});
        return bg;
    });

    // pad_ext 左填充补零（M2.3c 新路径）：x=[5,2] -> lp0=2, rp0=1, lp1=1
    compare(cpu, dev, "pad_zero_left", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 5, 2);
        Tensor* out = pad_ext(ctx, x, 2, 1, 1, 0, 0, 0, 0, 0, PAD_ZERO);
        return BuiltGraph{out, {{x, ramp(10)}}};
    });

    // pad reflect 1D（左右非对称）：x=[6,1] -> lp0=3, rp0=2
    compare(cpu, dev, "pad_reflect_1d", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 6, 1);
        Tensor* out = pad_ext(ctx, x, 3, 2, 0, 0, 0, 0, 0, 0, PAD_REFLECT);
        return BuiltGraph{out, {{x, ramp(6)}}};
    });

    // pad reflect 2D（四边非对称）：x=[4,3] -> lp0=2,rp0=1,lp1=1,rp1=2
    compare(cpu, dev, "pad_reflect_2d", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 4, 3);
        Tensor* out = pad_ext(ctx, x, 2, 1, 1, 2, 0, 0, 0, 0, PAD_REFLECT);
        return BuiltGraph{out, {{x, ramp(12)}}};
    });

    // acc（inplace=false：结果为 a 的完整拷贝 + 区域累加）
    compare(cpu, dev, "acc_copy", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 4, 4);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 4, 2);
        Tensor* out = acc(ctx, a, b, a->nb[1], 0, 0, a->nb[1], false);
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({a, ramp(16)});
        bg.inputs.push_back({b, ramp(8)});
        return bg;
    });

    // acc（inplace=true：结果为 a 的完整视图，区域内累加；覆盖 3D nb2/offset 布局）
    compare(cpu, dev, "acc_inplace", [](Context* ctx, Graph*) {
        Tensor* a = new_tensor_3d(ctx, TYPE_F32, 2, 3, 4);
        Tensor* b = new_tensor_3d(ctx, TYPE_F32, 2, 2, 2);
        const size_t offset = (size_t) 1 * a->nb[1] + (size_t) 1 * a->nb[2];
        Tensor* out = acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
        BuiltGraph bg;
        bg.out = out;  // a 的完整视图：读出的就是原地累加后的 a
        bg.inputs.push_back({a, ramp(24)});
        bg.inputs.push_back({b, ramp(8)});
        return bg;
    });
}

TRC_TEST(conformance_large_and_chain) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    // 大张量（跨多个 workgroup）+ 多层链
    compare(cpu, dev, "large_silu_chain", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, 100000);
        Tensor* y   = silu(ctx, x);
        Tensor* out = add(ctx, silu(ctx, add(ctx, y, x)), mul(ctx, y, y));
        return BuiltGraph{out, {{x, ramp(100000)}}};
    });
}

// M2.1a：小网格上限设备的 stride 循环回归。
// 通过环境变量 TRC_VK_MAX_GROUPS 把 x 轴网格上限强制降到 64（每批 16384 元素），
// 所有大张量 kernel 必须仍然覆盖全部元素（修复前：漏算 / 触发 dispatch 上限断言）。
// 同时覆盖 dtype/索引/卷积等 shader 家族的 stride 循环。
TRC_TEST(conformance_dispatch_grid_limit) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    TRC_EXPECT(_putenv_s("TRC_VK_MAX_GROUPS", "64") == 0);

    constexpr int64_t N = 200000;

    // 二元/一元逐元素
    compare(cpu, dev, "grid_binary_add", [N](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_1d(ctx, TYPE_F32, N);
        Tensor* b   = new_tensor_1d(ctx, TYPE_F32, N);
        Tensor* out = add(ctx, a, b);
        return BuiltGraph{out, {{a, ramp(N)}, {b, ramp(N)}}};
    });
    compare(cpu, dev, "grid_unary_silu", [N](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, N);
        Tensor* out = silu(ctx, x);
        return BuiltGraph{out, {{x, ramp(N)}}};
    });

    // 拷贝类：cont(transpose) / pad / concat（copy.comp）
    compare(cpu, dev, "grid_cont_transposed", [](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 200, 1000);
        Tensor* out = cont(ctx, transpose(ctx, x));
        return BuiltGraph{out, {{x, ramp(200000)}}};
    });
    compare(cpu, dev, "grid_pad", [N](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, N);
        Tensor* out = pad(ctx, x, 1000, 0, 0, 0);
        return BuiltGraph{out, {{x, ramp(N)}}};
    });
    compare(cpu, dev, "grid_concat", [N](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_1d(ctx, TYPE_F32, N / 2);
        Tensor* b   = new_tensor_1d(ctx, TYPE_F32, N / 2);
        Tensor* out = concat(ctx, a, b, 0);
        return BuiltGraph{out, {{a, ramp(N / 2)}, {b, ramp(N / 2)}}};
    });

    // 广播/归约类：repeat / repeat_back
    compare(cpu, dev, "grid_repeat", [](Context* ctx, Graph*) {
        Tensor* a      = new_tensor_2d(ctx, TYPE_F32, 1000, 1);
        Tensor* target = new_tensor_2d(ctx, TYPE_F32, 1000, 200);
        Tensor* out    = repeat(ctx, a, target);
        return BuiltGraph{out, {{a, ramp(1000)}}};
    });
    compare(cpu, dev, "grid_repeat_back", [N](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, N, 2);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, N, 1);
        Tensor* out = repeat_back(ctx, a, b);
        return BuiltGraph{out, {{a, ramp(N * 2)}}};
    });

    // matmul batch 平面（grid z 跨步；65 平面 > 上限 64）
    compare(cpu, dev, "grid_mul_mat_batch_z", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_4d(ctx, TYPE_F32, 8, 4, 1, 1);
        Tensor* b   = new_tensor_4d(ctx, TYPE_F32, 8, 4, 65, 1);
        Tensor* out = mul_mat(ctx, a, b);
        TRC_EXPECT_EQ(out->ne[2], 65);
        return BuiltGraph{out, {{a, ramp(32)}, {b, ramp(8 * 4 * 65)}}};
    });

    // acc（copy + acc_add）
    compare(cpu, dev, "grid_acc", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 50000, 4);
        Tensor* b   = new_tensor_2d(ctx, TYPE_F32, 50000, 2);
        Tensor* out = acc(ctx, a, b, a->nb[1], 0, 0, a->nb[1], false);
        return BuiltGraph{out, {{a, ramp(200000)}, {b, ramp(100000)}}};
    });

    // 类型转换（设备不支持 16 位存储时整例跳过；cast 两个方向）
    compare(cpu, dev, "grid_cast_roundtrip", [N](Context* ctx, Graph*) {
        Tensor* x   = new_tensor_1d(ctx, TYPE_F32, N);
        Tensor* out = cast(ctx, cast(ctx, x, TYPE_F16), TYPE_F32);
        return BuiltGraph{out, {{x, ramp(N)}}};
    });

    // 索引：get_rows / get_rows_back / set_rows
    compare(cpu, dev, "grid_get_rows", [](Context* ctx, Graph*) {
        Tensor* table = new_tensor_4d(ctx, TYPE_F32, 4, 6000, 1, 1);
        Tensor* idx   = new_tensor_4d(ctx, TYPE_I32, 6000, 1, 1, 1);
        Tensor* out   = get_rows(ctx, table, idx);
        std::vector<int32_t> id(6000);
        for (int32_t i = 0; i < 6000; ++i) {
            id[(size_t) i] = i;
        }
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({table, ramp(24000)});
        bg.int_inputs.push_back({idx, id});
        return bg;
    });
    compare(cpu, dev, "grid_get_rows_back", [](Context* ctx, Graph*) {
        Tensor* grad  = new_tensor_4d(ctx, TYPE_F32, 4, 6000, 1, 1);
        Tensor* idx   = new_tensor_4d(ctx, TYPE_I32, 6000, 1, 1, 1);
        Tensor* table = new_tensor_4d(ctx, TYPE_F32, 4, 6000, 1, 1);
        Tensor* out   = get_rows_back(ctx, grad, idx, table);
        std::vector<int32_t> id(6000);
        for (int32_t i = 0; i < 6000; ++i) {
            id[(size_t) i] = i;
        }
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({grad, ramp(24000)});
        bg.int_inputs.push_back({idx, id});
        return bg;
    });
    compare(cpu, dev, "grid_set_rows", [](Context* ctx, Graph*) {
        Tensor* dest = new_tensor_4d(ctx, TYPE_F32, 4, 8000, 1, 1);
        Tensor* vals = new_tensor_4d(ctx, TYPE_F32, 4, 6000, 1, 1);
        Tensor* idx  = new_tensor_4d(ctx, TYPE_I32, 6000, 1, 1, 1);
        Tensor* out  = set_rows(ctx, dest, vals, idx);
        std::vector<int32_t> id(6000);
        for (int32_t i = 0; i < 6000; ++i) {
            id[(size_t) i] = i;
        }
        BuiltGraph bg;
        bg.out = out;
        bg.inputs.push_back({dest, ramp(32000)});
        bg.inputs.push_back({vals, ramp(24000)});
        bg.int_inputs.push_back({idx, id});
        return bg;
    });

    // 卷积家族：im2col / im2col_back / col2im_1d / conv_transpose_1d
    compare(cpu, dev, "grid_im2col_1d", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_3d(ctx, TYPE_F32, 3, 1, 1);
        Tensor* img = new_tensor_3d(ctx, TYPE_F32, 100000, 1, 1);
        Tensor* out = im2col(ctx, k, img, 1, 0, 1, 0, 1, 0, false, TYPE_F32);
        return BuiltGraph{out, {{k, ramp(3)}, {img, ramp(100000)}}};
    });
    compare(cpu, dev, "grid_im2col_back_1d", [](Context* ctx, Graph*) {
        Tensor*       grad   = new_tensor_2d(ctx, TYPE_F32, 3, 99998);
        Tensor*       kernel = new_tensor_2d(ctx, TYPE_F32, 3, 1);
        const int64_t ne[4]  = {100000, 1, 1, 1};
        Tensor*       out    = im2col_back(ctx, grad, kernel, ne, 1, 0, 1, 0, 1, 0, false);
        return BuiltGraph{out, {{grad, ramp(299994)}, {kernel, ramp(3)}}};
    });
    compare(cpu, dev, "grid_col2im_1d", [](Context* ctx, Graph*) {
        Tensor* a   = new_tensor_2d(ctx, TYPE_F32, 3, 100000);
        Tensor* out = col2im_1d(ctx, a, 1, 1, 0);
        return BuiltGraph{out, {{a, ramp(300000)}}};
    });
    compare(cpu, dev, "grid_conv_transpose_1d", [](Context* ctx, Graph*) {
        Tensor* k   = new_tensor_3d(ctx, TYPE_F32, 3, 2, 1);
        Tensor* x   = new_tensor_2d(ctx, TYPE_F32, 100000, 1);
        Tensor* out = conv_transpose_1d(ctx, k, x, 1, 0, 1);
        return BuiltGraph{out, {{k, ramp(6)}, {x, ramp(100000)}}};
    });

    _putenv_s("TRC_VK_MAX_GROUPS", "");
}

// pad 显式写零（M2.1h / R16）：输出区域先写脏再 compute，pad 区必须为 0（CPU/Vulkan 一致）
TRC_TEST(conformance_pad_zero_fill) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    auto run = [](Device* d) {
        Context* ctx = context_new(1 << 20);
        Graph*   g   = graph_new(ctx);
        Tensor*  a   = new_tensor_2d(ctx, TYPE_F32, 3, 2);
        Tensor*  out = pad(ctx, a, 1, 2, 0, 0);  // [4, 4]：dim0 右补 1、dim1 右补 2
        graph_build_forward_expand(ctx, g, out);
        Buffer* buf = buffer_alloc_ctx_tensors(ctx, d->default_buffer_type());

        const float va[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        tensor_set(a, va, 0, sizeof(va));

        // 把输出整块写脏（999）：pad 区必须由 compute 显式清零，而非依赖"新建缓冲恰好为 0"
        std::vector<float> dirty((size_t) tensor_nelements(out), 999.0f);
        tensor_set(out, dirty.data(), 0, dirty.size() * sizeof(float));

        d->graph_compute(g);

        std::vector<float> v((size_t) tensor_nelements(out));
        tensor_get(out, v.data(), 0, v.size() * sizeof(float));

        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
        return v;
    };

    const std::vector<float> want = run(cpu);
    const std::vector<float> got  = run(dev);
    TRC_EXPECT(want.size() == 16 && got.size() == 16);  // [3+1, 2+2]

    double max_abs   = 0.0;
    int    pad_bad   = 0;
    const size_t n   = want.size() < got.size() ? want.size() : got.size();
    for (size_t i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::fabs((double) got[i] - (double) want[i]));
        if (want[i] == 0.0f && got[i] != 0.0f) {
            ++pad_bad;
        }
    }
    TRC_EXPECT_EQ(pad_bad, 0);
    TRC_EXPECT_NEAR(max_abs, 0.0, 1e-6);
    std::printf("    [ OK ] %-24s max_abs=%.3g\n", "pad_zero_fill", max_abs);
}

// 设备端优化器步（M4.1）：CPU vs Vulkan 就地更新结果对照
TRC_TEST(conformance_opt_step) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    compare(cpu, dev, "opt_step_adamw", [](Context* ctx, Graph*) {
        Tensor* p   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* g   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* m   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* v   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* out = opt_step_adamw(ctx, p, g, m, v, 0.05f, 0.0316227766f, 0.999f, 0.9f, 0.999f,
                                     1e-8f);
        return BuiltGraph{out, {{p, ramp(16)}, {g, ramp(16)}, {m, ramp(16)}, {v, ramp_pos(16)}}};
    });

    compare(cpu, dev, "opt_step_sgd", [](Context* ctx, Graph*) {
        Tensor* p   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* g   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* m   = new_tensor_1d(ctx, TYPE_F32, 16);
        Tensor* out = opt_step_sgd(ctx, p, g, m, 0.1f, 0.9f, 0.1f, 0.01f, true, false);
        return BuiltGraph{out, {{p, ramp(16)}, {g, ramp(16)}, {m, ramp(16)}}};
    });
}

// 设备端梯度裁剪原语 / WeightNorm 同步（M4.5）：CPU vs Vulkan
TRC_TEST(conformance_clip_and_weightnorm) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    （本机未检测到 Vulkan 设备，跳过）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    compare(cpu, dev, "sum_sqr_acc", [](Context* ctx, Graph*) {
        Tensor* acc = new_tensor_1d(ctx, TYPE_F32, 1);
        Tensor* a   = new_tensor_1d(ctx, TYPE_F32, 512);
        Tensor* out = sum_sqr_acc(ctx, acc, a);
        return BuiltGraph{out, {{acc, std::vector<float>{2.0f}}, {a, ramp(512)}}};
    });

    compare(cpu, dev, "clip_scale_inplace", [](Context* ctx, Graph*) {
        Tensor* a    = new_tensor_1d(ctx, TYPE_F32, 512);
        Tensor* norm = new_tensor_1d(ctx, TYPE_F32, 1);
        Tensor* out  = clip_scale_inplace(ctx, a, norm, 1.0f, 1e-6f);
        return BuiltGraph{out, {{a, ramp(512)}, {norm, std::vector<float>{1000.0f}}}};
    });

    compare(cpu, dev, "weightnorm_sync", [](Context* ctx, Graph*) {
        Tensor* v   = new_tensor_2d(ctx, TYPE_F32, 7, 4);  // [n_rest=7, oc=4]
        Tensor* g   = new_tensor_2d(ctx, TYPE_F32, 1, 4);
        Tensor* out = weightnorm_sync(ctx, v, g);
        return BuiltGraph{out, {{v, ramp(28)}}};
    });
}

// 设备端批量填充（M4.2）：CPU（主机直填）vs Vulkan（vkCmdFillBuffer）结果对照；
// 同时验证不同取值（含负值/非整数）与多张量一次提交。
TRC_TEST(conformance_fill_tensors) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }
    Device* cpu = device_cpu();
    TRC_EXPECT(cpu != nullptr);

    const float values[] = {0.0f, 1.0f, 1.5f, -2.25f};
    const int64_t sizes[] = {1, 37, 256, 4096};

    auto run = [&](Device* d) {
        std::vector<Tensor*>    ts;
        std::vector<TensorFill> fills;
        Context* ctx = context_new(1 << 24);
        for (int64_t n : sizes) {
            Tensor* t = new_tensor_1d(ctx, TYPE_F32, n);
            ts.push_back(t);
        }
        Buffer* buf = buffer_alloc_ctx_tensors(ctx, d->default_buffer_type());
        TRC_EXPECT(buf != nullptr);
        for (size_t i = 0; i < ts.size(); ++i) {
            fills.push_back({ts[i], values[i]});
        }
        d->fill_tensors(fills.data(), fills.size());
        std::vector<std::vector<float>> got(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            got[i].resize((size_t) sizes[i]);
            tensor_get(ts[i], got[i].data(), 0, got[i].size() * sizeof(float));
        }
        buffer_free(buf);
        context_free(ctx);
        return got;
    };

    const std::vector<std::vector<float>> a = run(cpu);
    const std::vector<std::vector<float>> b = run(dev);
    int64_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (a[i][j] != values[i] || b[i][j] != values[i]) {
                ++bad;
            }
        }
    }
    TRC_EXPECT_EQ(bad, 0);
    std::printf("    [ OK ] %-24s 4 张量/值全部一致\n", "fill_tensors");
}
