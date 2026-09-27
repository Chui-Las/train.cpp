// train.cpp - 逐 epoch 训练轨迹 PyTorch 对拍（M2.0d）
//
// 重放 scripts/gen_golden.py 生成的固定小 MLP（Linear+ReLU+Linear，MSE；AdamW / SGD+momentum）
// 的多 epoch 全批量训练：逐 epoch 比对全部参数与评估 loss（前向，不含更新）。
// 黄金数据缺失时自动跳过；TRC_TEST_DEVICE=vulkan 时在 Vulkan 上复跑。
#include "npy.h"
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace traincpp;
using namespace trc_test;

namespace {

std::map<std::string, std::string> load_meta(const std::string& path, bool* ok) {
    std::map<std::string, std::string> m;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        *ok = false;
        return m;
    }
    char key[128];
    char val[256];
    while (std::fscanf(f, "%127s %255s", key, val) == 2) {
        m[key] = val;
    }
    std::fclose(f);
    *ok = true;
    return m;
}

int64_t meta_i(const std::map<std::string, std::string>& m, const char* k, int64_t def) {
    const auto it = m.find(k);
    return it == m.end() ? def : (int64_t) std::atoll(it->second.c_str());
}

double meta_f(const std::map<std::string, std::string>& m, const char* k, double def) {
    const auto it = m.find(k);
    return it == m.end() ? def : std::atof(it->second.c_str());
}

std::vector<float> read_f32(const Tensor* t) {
    std::vector<float> v((size_t) tensor_nelements(t));
    tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

float read_scalar(const Tensor* t) {
    float v = 0.0f;
    tensor_get(t, &v, 0, sizeof(float));
    return v;
}

// 按 npy 形状（反转 -> ne）创建 F32 张量；数据由调用方在分配后写入
Tensor* load_tensor_f32(Context* ctx, const std::string& path, NpyArray& arr) {
    if (!npy_load(path, arr)) {
        return nullptr;
    }
    int64_t ne[MAX_DIMS] = {1, 1, 1, 1};
    const int nd = (int) arr.shape.size();
    for (int i = 0; i < nd; ++i) {
        ne[i] = arr.shape[(size_t) (nd - 1 - i)];
    }
    return new_tensor_nd(ctx, TYPE_F32, nd > 0 ? nd : 1, ne);
}

void compare_f32(const Tensor* t, const float* expect, int64_t n, const char* what) {
    const std::vector<float> got = read_f32(t);
    if ((int64_t) got.size() != n) {
        TRC_EXPECT(false && what);
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        const double tol = 1e-5 + 1e-5 * std::fabs((double) expect[(size_t) i]);
        TRC_EXPECT_NEAR(got[(size_t) i], expect[(size_t) i], tol);
    }
}

// 返回：1 = 通过，0 = 跳过
int run_case(Device* dev, const std::string& dir, const char* name) {
    bool meta_ok = false;
    const auto meta = load_meta(dir + "/trace_" + name + "_meta.txt", &meta_ok);
    if (!meta_ok) {
        return 0;
    }
    const int64_t in    = meta_i(meta, "in", 0);
    const int64_t hid   = meta_i(meta, "hid", 0);
    const int64_t out   = meta_i(meta, "out", 0);
    const int64_t epochs = meta_i(meta, "epochs", 0);
    const int64_t steps  = meta_i(meta, "steps", 1);

    Context* ctx = context_new(1 << 22);
    Graph*   g   = graph_new(ctx);

    NpyArray xa;
    NpyArray ta;
    Tensor*  x  = load_tensor_f32(ctx, dir + "/trace_" + name + "_x.npy", xa);
    Tensor*  tg = load_tensor_f32(ctx, dir + "/trace_" + name + "_target.npy", ta);
    Tensor*  w1 = new_tensor_2d(ctx, TYPE_F32, in, hid);
    Tensor*  b1 = new_tensor_1d(ctx, TYPE_F32, hid);
    Tensor*  w2 = new_tensor_2d(ctx, TYPE_F32, hid, out);
    Tensor*  b2 = new_tensor_1d(ctx, TYPE_F32, out);

    Linear l1;
    Linear l2;
    linear_init_weight(ctx, &l1, w1, b1);
    linear_init_weight(ctx, &l2, w2, b2);

    Tensor* h    = relu(ctx, linear_forward(ctx, &l1, x));
    Tensor* loss = mse_loss(ctx, linear_forward(ctx, &l2, h), tg);
    tensor_set_loss(loss);

    ParamList p1;
    ParamList p2;
    ParamList params;
    linear_params(&l1, &p1);
    linear_params(&l2, &p2);
    param_list_merge(&params, &p1);
    param_list_merge(&params, &p2);
    for (int64_t i = 0; i < param_list_count(&params); ++i) {
        tensor_set_param(param_list_get(&params, i));
    }

    graph_build_forward_expand(ctx, g, loss);
    graph_build_backward_expand(g);

    if (x == nullptr || tg == nullptr) {
        graph_free(g);
        context_free(ctx);
        return 0;
    }
    if (const Tensor* bad = first_unsupported_node(g, dev)) {
        std::printf("    [跳过] trace_%s 后端不支持 %s\n", name, op_name(bad->op));
        graph_free(g);
        context_free(ctx);
        return 0;
    }

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

    // 装载参数初值与数据
    Tensor* ps[4] = {w1, b1, w2, b2};
    bool data_ok = true;
    for (int i = 0; i < 4 && data_ok; ++i) {
        NpyArray arr;
        if (!npy_load(dir + "/trace_" + name + "_p" + std::to_string(i) + "_init.npy", arr) ||
            arr.raw.size() != (size_t) tensor_nelements(ps[i]) * sizeof(float)) {
            data_ok = false;
            break;
        }
        tensor_set(ps[i], arr.raw.data(), 0, arr.raw.size());
    }
    if (data_ok) {
        data_ok = xa.raw.size() == (size_t) tensor_nelements(x) * sizeof(float) &&
                  ta.raw.size() == (size_t) tensor_nelements(tg) * sizeof(float);
        if (data_ok) {
            tensor_set(x, xa.raw.data(), 0, xa.raw.size());
            tensor_set(tg, ta.raw.data(), 0, ta.raw.size());
        }
    }
    if (!data_ok) {
        std::printf("    [跳过] trace_%s 黄金数据缺失\n", name);
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
        return 0;
    }

    // 优化器（超参与 PyTorch 对齐）
    Optimizer* opt = nullptr;
    const std::string kind = meta.count("kind") ? meta.at("kind") : "adamw";
    if (kind == "adamw") {
        AdamwOptions o;
        o.lr           = (float) meta_f(meta, "lr", 1e-3);
        o.beta1        = (float) meta_f(meta, "beta1", 0.9);
        o.beta2        = (float) meta_f(meta, "beta2", 0.999);
        o.eps          = (float) meta_f(meta, "eps", 1e-8);
        o.weight_decay = (float) meta_f(meta, "weight_decay", 0.0);
        opt = optim_adamw_new(ctx, ps, 4, o);
    } else {
        SgdOptions o;
        o.lr           = (float) meta_f(meta, "lr", 1e-2);
        o.momentum     = (float) meta_f(meta, "momentum", 0.0);
        o.dampening    = (float) meta_f(meta, "dampening", 0.0);
        o.weight_decay = (float) meta_f(meta, "weight_decay", 0.0);
        o.nesterov     = meta_i(meta, "nesterov", 0) != 0;
        opt = optim_sgd_new(ctx, ps, 4, o);
    }

    double loss_first = 0.0;
    double loss_last  = 0.0;
    for (int64_t e = 1; e <= epochs; ++e) {
        for (int64_t s = 0; s < steps; ++s) {
            graph_reset(g);
            dev->graph_compute(g);
            optim_step(opt);
        }

        // 评估：前向 + 反向但不更新（与 Python 侧 no_grad 前向的 loss 对齐）
        graph_reset(g);
        dev->graph_compute(g);
        const float got_loss = read_scalar(loss);
        if (e == 1) {
            loss_first = got_loss;
        }
        loss_last = got_loss;

        NpyArray le;
        if (npy_load(dir + "/trace_" + name + "_loss_epoch" + std::to_string(e) + ".npy", le) &&
            le.raw.size() >= sizeof(float)) {
            const float exp_loss = ((const float*) le.raw.data())[0];
            TRC_EXPECT_NEAR(got_loss, exp_loss, 1e-5 + 1e-5 * std::fabs((double) exp_loss));
        }
        for (int i = 0; i < 4; ++i) {
            NpyArray pe;
            const std::string path = dir + "/trace_" + name + "_p" + std::to_string(i) +
                                     "_epoch" + std::to_string(e) + ".npy";
            if (!npy_load(path, pe) ||
                pe.raw.size() != (size_t) tensor_nelements(ps[i]) * sizeof(float)) {
                continue;
            }
            compare_f32(ps[i], (const float*) pe.raw.data(), tensor_nelements(ps[i]),
                        "训练轨迹参数");
        }
    }
    std::printf("    [ OK ] trace_%-12s loss %.6g -> %.6g（%lld epoch x %lld step）\n", name,
                loss_first, loss_last, (long long) epochs, (long long) steps);

    optim_free(opt);
    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return 1;
}

} // namespace

TRC_TEST(train_trace_epochs) {
    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE 指定的设备不可用\n");
        return;
    }
    const std::string dir = TRC_GOLDEN_DIR;
    const char* cases[] = {"adamw", "sgd_momentum"};
    int n_ok = 0;
    int n_skip = 0;
    for (const char* c : cases) {
        if (run_case(dev, dir, c) != 0) {
            ++n_ok;
        } else {
            ++n_skip;
        }
    }
    if (n_ok == 0 && n_skip > 0) {
        std::printf("    跳过：未找到训练轨迹黄金数据（请先运行 python scripts/gen_golden.py）\n");
    } else {
        std::printf("    训练轨迹对拍：%d 个用例通过（%d 个缺失/跳过）\n", n_ok, n_skip);
    }
}
