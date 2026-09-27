#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 PyTorch 黄金数据，供 tests/test_golden.cpp 对拍（traincpp M1.2e）。

用法（在仓库根目录执行）：
    python scripts/gen_golden.py
    python scripts/gen_golden.py --out tests/golden --seed 1234

输出：
    <out>/cases.txt               用例清单（每行一个用例）
    <out>/<name>_in<i>.npy        输入张量（第 i 个）
    <out>/<name>_out.npy          期望输出

清单格式（空格分隔）：
    <name> <op> <n_attrs> <attr...> <n_inputs> <ne0>x<ne1>x<ne2>x<ne3>:<type> ...
约定：
    - ne 采用 train.cpp 的顺序（ne0 最内层）；.npy 为 C 序，形状 = (ne3, ne2, ne1, ne0)
    - 目前所有计算均为 f32，索引输入为 i32

注意：所有算子的参考语义都与 docs/兼容性.md 中的 ggml 对齐描述保持一致，
      Python 侧实现必须逐条对应，不要"顺手"改成更符合直觉的写法。
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)


# ---------------------------------------------------------------- 基础工具
def rev(ne):
    """train.cpp 的 ne 顺序 -> .npy 形状（C 序）"""
    return tuple(reversed(ne))


def rand(ne, positive=False, dtype=torch.float32):
    x = torch.randn(*rev(ne), dtype=dtype)
    if positive:
        x = x.abs() + 0.1
    return x


class Golden:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.lines = []
        os.makedirs(out_dir, exist_ok=True)

    def add(self, name, op, attrs, inputs, output):
        """inputs: list of tensor 或 (tensor, 'i32')；output: tensor"""
        input_shapes = []
        for i, item in enumerate(inputs):
            tensor, type_name = (item if isinstance(item, tuple) else (item, "f32"))
            # 必须转成 C 连续：非连续视图会被 np.save 写成 fortran_order=True
            arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
            np.save(os.path.join(self.out_dir, f"{name}_in{i}.npy"), arr)
            input_shapes.append(f"{tensor.shape[3]}x{tensor.shape[2]}x{tensor.shape[1]}x{tensor.shape[0]}:{type_name}")
        arr = np.ascontiguousarray(output.detach().cpu().numpy())
        np.save(os.path.join(self.out_dir, f"{name}_out.npy"), arr)

        attr_str = " ".join(str(a) for a in attrs)
        line = f"{name} {op} {len(attrs)} {attr_str} {len(inputs)} {' '.join(input_shapes)}"
        self.lines.append(" ".join(line.split()))

    def flush(self):
        with open(os.path.join(self.out_dir, "cases.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"已生成 {len(self.lines)} 个用例 -> {self.out_dir}")


# ---------------------------------------------------------------- 用例
def build(g):
    two = [3, 2, 1, 1]

    # ---- 逐元素二元/广播 ----
    a = rand(two)
    b = rand(two)
    b2 = rand(two) + 2.0
    g.add("add_same", "add", [], [a, b], a + b)
    g.add("sub_same", "sub", [], [a, b], a - b)
    g.add("mul_same", "mul", [], [a, b], a * b)
    g.add("div_same", "div", [], [a, b2], a / b2)

    ab = rand([3, 1, 1, 1])
    g.add("add_bcast", "add", [], [a, ab], a + ab)

    scalar = torch.tensor([[[[1.5]]]])
    g.add("add1", "add1", [], [a, scalar], a + scalar)

    # ---- 逐元素一元 ----
    x = rand([4, 3, 1, 1])
    xp = x.abs() + 0.1
    unary_torch = {
        "neg": lambda t: -t,
        "abs": torch.abs,
        "sqr": lambda t: t * t,
        "sqrt": torch.sqrt,
        "exp": torch.exp,
        "log": torch.log,
        "sin": torch.sin,
        "cos": torch.cos,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "silu": F.silu,
    }
    for op in ["neg", "abs", "sqr", "exp", "sin", "cos", "relu", "sigmoid", "tanh", "silu"]:
        g.add(f"unary_{op}", op, [], [x], unary_torch[op](x))
    g.add("unary_sqrt", "sqrt", [], [xp], torch.sqrt(xp))
    g.add("unary_log", "log", [], [xp], torch.log(xp))

    g.add("unary_gelu", "gelu", [], [x], F.gelu(x, approximate="tanh"))
    g.add("unary_gelu_erf", "gelu_erf", [], [x], F.gelu(x, approximate="none"))
    g.add("unary_softplus", "softplus", [], [x], F.softplus(x, beta=1.0, threshold=20.0))
    g.add("unary_hardswish", "hardswish", [], [x], F.hardswish(x))

    g.add("leaky_relu", "leaky_relu", [0.1], [x], F.leaky_relu(x, 0.1))
    g.add("clamp", "clamp", [-0.5, 0.5], [x], torch.clamp(x, -0.5, 0.5))
    g.add("scale", "scale", [2.5], [x], x * 2.5)

    # ---- mul_mat：a=[k,m]，b=[k,n] -> [m,n] ----
    k, m, n = 3, 2, 4
    A = rand([k, m, 1, 1])
    B = rand([k, n, 1, 1])
    y = _mul_mat(A, B)
    g.add("mul_mat", "mul_mat", [], [A, B], y)

    # ---- mul_mat batch：a 广播到 b 的平面 / 整除广播（a2=2,b2=4）/ 跨 64 tile ----
    Ab = rand([5, 3, 1, 1])
    Bb = rand([5, 4, 2, 3])
    g.add("mul_mat_batch", "mul_mat", [], [Ab, Bb], _mul_mat(Ab, Bb))

    Ad = rand([4, 2, 2, 1])
    Bd = rand([4, 3, 4, 1])
    g.add("mul_mat_batch_bcast_div", "mul_mat", [], [Ad, Bd], _mul_mat(Ad, Bd))

    At = rand([17, 66, 1, 1])
    Bt = rand([17, 65, 1, 2])
    g.add("mul_mat_batch_tile", "mul_mat", [], [At, Bt], _mul_mat(At, Bt))

    # ---- 归约 ----
    r = rand([5, 3, 2, 1])
    g.add("sum", "sum", [], [r], r.sum().reshape(1, 1, 1, 1))
    g.add("sum_rows", "sum_rows", [], [r], r.sum(dim=-1, keepdim=True))
    g.add("mean", "mean", [], [r], r.mean(dim=-1, keepdim=True))

    # ---- 归一化 ----
    nrm = rand([4, 3, 2, 1])
    eps = 1e-5
    g.add("norm", "norm", [eps], [nrm], F.layer_norm(nrm, (nrm.shape[-1],), eps=eps))
    g.add("rms_norm", "rms_norm", [eps], [nrm],
          nrm * torch.rsqrt(nrm.pow(2).mean(dim=-1, keepdim=True) + eps))
    g.add("group_norm", "group_norm", [2, eps], [nrm], F.group_norm(nrm, 2, eps=eps))

    # ---- soft_max ----
    sm = rand([5, 3, 1, 1])
    mask = rand([5, 3, 1, 1])
    g.add("soft_max", "soft_max", [], [sm], F.softmax(sm, dim=-1))
    g.add("soft_max_ext", "soft_max_ext", [0.5, 0.0], [sm, mask], F.softmax(sm * 0.5 + mask, dim=-1))

    # ---- cross_entropy_loss（ggml 语义：target 为同形状概率分布，输出标量按行均值）----
    ce_logits = rand([5, 3, 1, 1])                       # [nc=5, nr=3]
    ce_tgt = torch.softmax(rand([5, 3, 1, 1]), dim=-1)   # 软标签（每行和为 1）
    ce_out = -(ce_tgt * F.log_softmax(ce_logits, dim=-1)).sum(dim=-1).mean().reshape(1, 1, 1, 1)
    g.add("cross_entropy_loss", "cross_entropy_loss", [], [ce_logits, ce_tgt], ce_out)

    # ---- get_rows / set_rows ----
    table = rand([4, 6, 1, 1])
    idx = torch.randint(0, 6, (1, 1, 1, 3), dtype=torch.int32)
    y = table.reshape(6, 4)[idx.reshape(-1).long()].reshape(1, 1, 3, 4)
    g.add("get_rows", "get_rows", [], [table, (idx, "i32")], y)

    dest = rand([4, 3, 1, 1])
    vals = rand([4, 2, 1, 1])
    sidx = torch.randint(0, 3, (1, 1, 1, 2), dtype=torch.int32)
    # 重复索引在并行后端下写入顺序不确定（ggml/本库 GPU 同为非原子写，未定义）；
    # 黄金数据必须使用唯一索引，否则 CPU 顺序覆盖与 GPU 竞态结果不一致。
    if int(sidx[0, 0, 0, 0]) == int(sidx[0, 0, 0, 1]):
        sidx[0, 0, 0, 1] = (sidx[0, 0, 0, 1] + 1) % 3
    d = dest.reshape(3, 4).clone()
    d.index_copy_(0, sidx.reshape(-1).long(), vals.reshape(2, 4))
    g.add("set_rows", "set_rows", [], [dest, vals, (sidx, "i32")], d.reshape(1, 1, 3, 4))

    # ---- repeat / concat / pad ----
    rep = rand([2, 1, 1, 1])
    g.add("repeat", "repeat", [2, 3, 1, 1], [rep], rep.expand(1, 1, 3, 2))

    ca = rand([2, 2, 1, 1])
    cb = rand([2, 1, 1, 1])
    g.add("concat_dim1", "concat", [1], [ca, cb], torch.cat([ca, cb], dim=2))
    ca0 = rand([2, 2, 1, 1])
    cb0 = rand([3, 2, 1, 1])
    g.add("concat_dim0", "concat", [0], [ca0, cb0], torch.cat([ca0, cb0], dim=3))

    pad_x = rand([3, 2, 1, 1])
    g.add("pad", "pad", [1, 1, 0, 0], [pad_x], F.pad(pad_x, (0, 1, 0, 1, 0, 0, 0, 0)))

    # ---- pad reflect（M2.3c；左右/上下非对称，不重复边缘）----
    pr1 = rand([5, 1, 1, 1])
    g.add("pad_reflect_1d", "pad_reflect", [3, 2, 0, 0, 0, 0, 0, 0], [pr1],
          F.pad(pr1, (3, 2, 0, 0, 0, 0), mode="reflect"))
    pr2 = rand([4, 3, 1, 1])
    g.add("pad_reflect_2d", "pad_reflect", [2, 1, 1, 2, 0, 0, 0, 0], [pr2],
          F.pad(pr2, (2, 1, 1, 2), mode="reflect"))

    # ---- im2col（1D 手写；2D 用 unfold）----
    KW1, IC1, W1, s0, p0, d0 = 3, 2, 5, 1, 1, 1
    k1 = rand([KW1, IC1, 1, 1])
    x1 = rand([W1, IC1, 1, 1])
    ow1 = (W1 + 2 * p0 - d0 * (KW1 - 1) - 1) // s0 + 1
    # x1 形状 (1,1,IC,W)：转成 (W,IC) 以便按 [w, ic] 取元素
    xt = x1.reshape(IC1, W1).t()
    cols = torch.zeros(IC1 * KW1, ow1)
    for ic in range(IC1):
        for kk in range(KW1):
            for i in range(ow1):
                src = i * s0 + kk * d0 - p0
                if 0 <= src < W1:
                    cols[ic * KW1 + kk, i] = xt[src, ic]
    g.add("im2col_1d", "im2col", [s0, 0, p0, 0, d0, 0, 0], [k1, x1],
          cols.t().reshape(1, 1, ow1, IC1 * KW1))

    KW2, KH2, IC2, W2, H2 = 3, 3, 2, 6, 5
    s0 = s1 = p0 = p1 = d0 = d1 = 1
    k2 = rand([KW2, KH2, IC2, 1])
    x2 = rand([W2, H2, IC2, 1])
    ow2 = (W2 + 2 * p0 - d0 * (KW2 - 1) - 1) // s0 + 1
    oh2 = (H2 + 2 * p1 - d1 * (KH2 - 1) - 1) // s1 + 1
    unf = F.unfold(x2, kernel_size=(KH2, KW2), dilation=(d1, d0), padding=(p1, p0), stride=(s1, s0))
    g.add("im2col_2d", "im2col", [s0, s1, p0, p1, d0, d1, 1], [k2, x2],
          unf.view(1, IC2 * KH2 * KW2, oh2, ow2).permute(0, 2, 3, 1))

    # ---- conv_1d / conv_2d（与 ggml 的组合语义一致）----
    IC, OC, KW, W = 2, 2, 3, 6
    sc, pc, dc = 2, 1, 1
    wc1 = rand([KW, IC, OC, 1])
    xc1 = rand([W, IC, 1, 1])
    y1 = F.conv1d(xc1.reshape(1, IC, W), wc1.reshape(OC, IC, KW), stride=sc, padding=pc, dilation=dc)
    # 结果 ne=[OW,OC,N,1] -> npy 形状 (1,N,OC,OW)
    g.add("conv_1d", "conv_1d", [sc, pc, dc], [wc1, xc1], y1.unsqueeze(0))

    # conv_1d 批量（N=2）：ggml v0.23.0 的 reshape 顺序在 N>1 时结果错误，本库已修正
    wc1b = rand([KW, IC, OC, 1])
    xc1b = rand([W, IC, 2, 1])
    y1b = F.conv1d(xc1b.reshape(2, IC, W), wc1b.reshape(OC, IC, KW), stride=sc, padding=pc, dilation=dc)
    g.add("conv_1d_batch", "conv_1d", [sc, pc, dc], [wc1b, xc1b], y1b.unsqueeze(0))

    IC, OC, KW, KH, W, H = 2, 2, 3, 3, 6, 5
    wc2 = rand([KW, KH, IC, OC])
    xc2 = rand([W, H, IC, 1])
    y2 = F.conv2d(xc2, wc2.reshape(OC, IC, KH, KW), stride=1, padding=1, dilation=1)
    g.add("conv_2d", "conv_2d", [1, 1, 1, 1, 1, 1], [wc2, xc2], y2)

    # ---- col2im_1d ----
    K, OC, T_in, sc, pc = 2, 2, 5, 1, 0
    ccols = rand([K * OC, T_in, 1, 1])
    T_out = (T_in - 1) * sc + K - 2 * pc
    fold_in = ccols.reshape(T_in, K * OC).t().unsqueeze(0)  # (1, K*OC, T_in)
    folded = F.fold(fold_in, output_size=(1, T_out), kernel_size=(1, K), dilation=(1, 1),
                    padding=(0, pc), stride=(1, sc))
    g.add("col2im_1d", "col2im_1d", [sc, OC, pc], [ccols], folded.permute(0, 2, 1, 3))

    # ---- conv_transpose_1d ----
    K, Cout, Cin, T_in, st = 3, 2, 2, 5, 2
    wt = rand([K, Cout, Cin, 1])
    xtt = rand([T_in, Cin, 1, 1])
    yt = F.conv_transpose1d(xtt.reshape(1, Cin, T_in), wt.reshape(Cin, Cout, K), stride=st)
    g.add("conv_transpose_1d", "conv_transpose_1d", [st, 0, 1], [wt, xtt], yt.unsqueeze(1))

    # ---- pool_2d / conv_transpose_2d / scale_bias（M2.3g）----
    # pool_2d AVG（核 3x2、步长 2、填充 1；count_include_pad=True 对齐 ggml 的 k0*k1 分母）
    ap = rand([6, 5, 2, 1])  # [W,H,C,N] -> torch (N,C,H,W)
    yp = F.avg_pool2d(ap, kernel_size=(2, 3), stride=(2, 2), padding=(1, 1),
                      count_include_pad=True)
    g.add("pool_2d_avg", "pool_2d", [1, 3, 2, 2, 2, 1, 1], [ap], yp)

    # pool_2d MAX（步长 1 重叠 + 填充；torch max_pool 的 padding 为 -inf，窗口均含实元素）
    am = rand([5, 4, 1, 1])
    ym = F.max_pool2d(am, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
    g.add("pool_2d_max", "pool_2d", [0, 2, 2, 1, 1, 1, 1], [am], ym)

    # conv_transpose_2d stride=1：w=[KW=2,KH=2,Cout=2,Cin=2]、x=[W=4,H=3,Cin=2,N=1]
    wt2 = rand([2, 2, 2, 2])
    xt2 = rand([4, 3, 2, 1])
    yt2 = F.conv_transpose2d(xt2, wt2, stride=1)
    g.add("conv_transpose_2d", "conv_transpose_2d", [1], [wt2, xt2], yt2)

    # conv_transpose_2d stride=2（覆盖 (ow-kw)%stride 过滤）
    ws2 = rand([3, 2, 2, 2])
    xs2 = rand([3, 2, 2, 1])
    ys2 = F.conv_transpose2d(xs2, ws2, stride=2)
    g.add("conv_transpose_2d_stride2", "conv_transpose_2d", [2], [ws2, xs2], ys2)

    # scale_bias：y = s*x + b（标量仿射，与 ggml_scale_bias 一致）
    sb = rand([4, 3, 1, 1])
    g.add("scale_bias", "scale_bias", [1.7, -0.3], [sb], sb * 1.7 - 0.3)

    # ---- 视图/形状 ----
    # 本库 transpose = permute(ne 的 0/1 维)；对应 torch 交换最后两维
    vt = rand([3, 4, 2, 1])
    g.add("transpose_cont", "transpose_cont", [], [vt], vt.permute(0, 1, 3, 2).contiguous())

    rs = rand([6, 1, 1, 1])
    g.add("reshape_2d", "reshape", [3, 2, 1, 1], [rs], rs.reshape(1, 1, 2, 3))

    # ---- cast（f32 -> f16 -> f32）----
    cs = rand([5, 2, 1, 1])
    g.add("cast", "cast", [], [cs], cs.half().float())


# ---------------------------------------------------------------- 反向梯度对拍（M1.3c）
# 输出：
#   grad_cases.txt                          梯度用例清单（DSL，见下）
#   <name>_in<id>.npy / <name>_grad<id>.npy 输入 / 参数梯度
#   <name>_loss.npy                         loss（1 元素）
#
# 清单格式（空白分隔，# 注释）：
#   case <name> <n_inputs>
#   in <id> <type> <ne0>x<ne1>x<ne2>x<ne3>
#   param <id>
#   node <id> <op> <n_attrs> <attr...> <n_src> <src...>
#   loss <id>
# id 为输入与节点共用的连续命名空间；node 按拓扑序记录，C++ 端顺序建图。
# 本脚本用 GradGraph 同步执行 torch 计算并记录 DSL，保证两侧图完全一致。

def _mul_mat(a, b):
    """a npy (a3,a2,m,k)、b npy (b3,b2,n,k) -> (b3,b2,n,m)

    ggml_mul_mat 语义：平面 (i2,i3) 用 a 的平面 (i2%a2, i3%a3)（要求 b2%a2==0、b3%a3==0）。
    torch 侧用 expand+reshape 显式造出取模重复的 a（torch 广播只支持 1/相等，不覆盖整除）。
    """
    a3, a2, m, k = a.shape[-4], a.shape[-3], a.shape[-2], a.shape[-1]
    b3, b2, n, bk = b.shape[-4], b.shape[-3], b.shape[-2], b.shape[-1]
    assert k == bk and b2 % a2 == 0 and b3 % a3 == 0, "mul_mat: 形状不满足 ggml_can_mul_mat"
    # i2 = q*a2 + r -> r = i2 % a2：把重复维放外层、a2 放内层再展平
    ae = a.unsqueeze(1).expand(a3, b2 // a2, a2, m, k).reshape(a3, b2, m, k)
    # i3 = p*a3 + s -> s = i3 % a3
    ae = ae.reshape(1, a3, b2, m, k).expand(b3 // a3, a3, b2, m, k).reshape(b3, b2, m, k)
    return b @ ae.transpose(-1, -2)


def _get_rows(table, idx):
    """表 npy (1,1,V,ne0)，索引 npy (1,1,1,nr) -> (1,1,nr,ne0)"""
    v, ne0 = table.shape[-2], table.shape[-1]
    nr = idx.shape[-1]
    return table.reshape(v, ne0)[idx.reshape(-1).long()].reshape(1, 1, nr, ne0)


def _conv_1d(w, x, s0, p0, d0):
    """w=[KW,IC,OC]、x=[W,IC,N] -> [OW,OC,N]；npy 形状 (1,OC,IC,KW) 与 (1,N,IC,W)"""
    oc, ic, kw = w.shape[-3], w.shape[-2], w.shape[-1]
    ww = x.shape[-1]
    y = F.conv1d(x.reshape(1, ic, ww), w.reshape(oc, ic, kw), stride=s0, padding=p0, dilation=d0)
    return y.unsqueeze(0)


def _conv_transpose_1d(w, x, attrs):
    """w=[K,Cout,Cin]、x=[T_in,Cin] -> [T_out,Cout,1,1]；npy 形状 (Cin,Cout,K) 与 (Cin,T_in)
    与核心一致：p0=0、d0=1、输出 [T_out,Cout,1,1]（T_out=(T_in-1)*s+K）"""
    cin, cout, k = w.shape[-3], w.shape[-2], w.shape[-1]
    t_in = x.shape[-1]
    y = F.conv_transpose1d(x.reshape(1, cin, t_in), w.reshape(cin, cout, k),
                           stride=int(attrs[0]), padding=int(attrs[1]), dilation=int(attrs[2]))
    return y.reshape(1, 1, cout, y.shape[-1])


def _pad_reflect(x, attrs):
    """attrs = [lp0,rp0,lp1,rp1,lp2,rp2,lp3,rp3]；torch 4D reflect 仅接受 4/6 个填充值，
    故只传前 6 个（覆盖 ne0/ne1/ne2；本库用例的 4 维输入 ne3 恒不填充）"""
    v = [int(t) for t in attrs]
    return F.pad(x, tuple(v[:6]), mode="reflect")


def _pool_2d(x, attrs):
    """attrs = [mode,k0,k1,s0,s1,p0,p1]；torch 的核/步长/填充顺序为 (H,W)。
    AVG 用 count_include_pad=True 对齐 ggml 的 k0*k1 分母；MAX 要求每窗口含实元素"""
    mode, k0, k1, s0, s1, p0, p1 = [int(v) for v in attrs]
    args = dict(kernel_size=(k1, k0), stride=(s1, s0), padding=(p1, p0))
    if mode == 0:
        return F.max_pool2d(x, **args)
    return F.avg_pool2d(x, count_include_pad=True, **args)


def _conv_2d(w, x, s0, s1, p0, p1, d0, d1):
    """w=[KW,KH,IC,OC]、x=[W,H,IC,N] -> [OW,OH,OC,N]；npy 形状 (OC,IC,KH,KW) 与 (N,IC,H,W)"""
    oc, ic, kh, kw = w.shape
    return F.conv2d(x, w.reshape(oc, ic, kh, kw), stride=(s1, s0), padding=(p1, p0),
                    dilation=(d1, d0))


GRAD_OPS = {
    # 逐元素
    "add":      lambda a, attrs: a[0] + a[1],
    "sub":      lambda a, attrs: a[0] - a[1],
    "mul":      lambda a, attrs: a[0] * a[1],
    "div":      lambda a, attrs: a[0] / a[1],
    "add1":     lambda a, attrs: a[0] + a[1],
    "neg":      lambda a, attrs: -a[0],
    "abs":      lambda a, attrs: torch.abs(a[0]),
    "sqr":      lambda a, attrs: a[0] * a[0],
    "sqrt":     lambda a, attrs: torch.sqrt(a[0]),
    "exp":      lambda a, attrs: torch.exp(a[0]),
    "log":      lambda a, attrs: torch.log(a[0]),
    "sin":      lambda a, attrs: torch.sin(a[0]),
    "cos":      lambda a, attrs: torch.cos(a[0]),
    "relu":     lambda a, attrs: torch.relu(a[0]),
    "sigmoid":  lambda a, attrs: torch.sigmoid(a[0]),
    "tanh":     lambda a, attrs: torch.tanh(a[0]),
    "silu":     lambda a, attrs: F.silu(a[0]),
    "gelu":     lambda a, attrs: F.gelu(a[0], approximate="tanh"),
    "gelu_erf": lambda a, attrs: F.gelu(a[0], approximate="none"),
    "softplus": lambda a, attrs: F.softplus(a[0], beta=1.0, threshold=20.0),
    "hardswish": lambda a, attrs: F.hardswish(a[0]),
    "leaky_relu": lambda a, attrs: F.leaky_relu(a[0], attrs[0]),
    "clamp":    lambda a, attrs: torch.clamp(a[0], attrs[0], attrs[1]),
    "scale":    lambda a, attrs: a[0] * attrs[0],
    "erf":      lambda a, attrs: torch.erf(a[0]),
    # 归约
    "sum":      lambda a, attrs: a[0].sum().reshape(1, 1, 1, 1),
    "sum_rows": lambda a, attrs: a[0].sum(dim=-1, keepdim=True),
    "mean":     lambda a, attrs: a[0].mean(dim=-1, keepdim=True),
    # 归一化 / 注意力
    "norm":       lambda a, attrs: F.layer_norm(a[0], (a[0].shape[-1],), eps=attrs[0]),
    "rms_norm":   lambda a, attrs: a[0] * torch.rsqrt(a[0].pow(2).mean(dim=-1, keepdim=True) + attrs[0]),
    "group_norm": lambda a, attrs: F.group_norm(a[0], int(attrs[0]), eps=attrs[1]),
    "soft_max":     lambda a, attrs: F.softmax(a[0], dim=-1),
    "soft_max_ext": lambda a, attrs: F.softmax(a[0] * attrs[0] + a[1], dim=-1),
    # 线代 / 索引 / 形状
    "mul_mat": lambda a, attrs: _mul_mat(a[0], a[1]),
    "get_rows": lambda a, attrs: _get_rows(a[0], a[1]),
    "concat": lambda a, attrs: torch.cat([a[0], a[1]], dim=3 - int(attrs[0])),
    "pad":    lambda a, attrs: F.pad(a[0], (0, int(attrs[0]), 0, int(attrs[1]),
                                             0, int(attrs[2]), 0, int(attrs[3]))),
    "pad_reflect": lambda a, attrs: _pad_reflect(a[0], attrs),
    "repeat": lambda a, attrs: a[0].expand(*rev([int(v) for v in attrs])),
    "reshape": lambda a, attrs: a[0].reshape(*rev([int(v) for v in attrs])),
    "transpose_cont": lambda a, attrs: a[0].permute(0, 1, 3, 2).contiguous(),
    "cont": lambda a, attrs: a[0].contiguous(),
    # 卷积
    "conv_1d": lambda a, attrs: _conv_1d(a[0], a[1], int(attrs[0]), int(attrs[1]), int(attrs[2])),
    "conv_2d": lambda a, attrs: _conv_2d(a[0], a[1], *[int(v) for v in attrs]),
    "conv_transpose_1d": lambda a, attrs: _conv_transpose_1d(a[0], a[1], attrs),
    "pool_2d": lambda a, attrs: _pool_2d(a[0], attrs),
    "conv_transpose_2d": lambda a, attrs: F.conv_transpose2d(a[1], a[0], stride=int(attrs[0])),
    "scale_bias": lambda a, attrs: a[0] * attrs[0] + attrs[1],
    # 损失（与 ggml cross_entropy_loss 语义一致：同形状概率目标，按行均值）
    "cross_entropy_loss": lambda a, attrs: (-(a[1] * F.log_softmax(a[0], dim=-1)).sum(dim=-1)
                                            .mean().reshape(1, 1, 1, 1)),
}


class GradGraph:
    """梯度用例构建器：执行 torch 前向并同步记录 DSL，flush 时求参数梯度并存盘"""

    def __init__(self, name, out_dir):
        self.name = name
        self.out_dir = out_dir
        self.next_id = 0
        self.tensors = {}
        self.decl = []      # (id, tensor, type_name, is_param)
        self.params = []
        self.nodes = []     # (id, op, attrs, srcs)
        self.loss_id = -1
        self.loss = None

    def inp(self, tensor, param=False, type_name="f32"):
        tid = self.next_id
        self.next_id += 1
        if param:
            tensor.requires_grad_(True)
            self.params.append(tid)
        self.tensors[tid] = tensor
        self.decl.append((tid, tensor, type_name, param))
        return tid

    def op(self, name, *srcs, attrs=()):
        args = [self.tensors[s] for s in srcs]
        out = GRAD_OPS[name](args, list(attrs))
        tid = self.next_id
        self.next_id += 1
        self.tensors[tid] = out
        self.nodes.append((tid, name, list(attrs), list(srcs)))
        return tid

    def set_loss(self, tid):
        self.loss_id = tid
        self.loss = self.tensors[tid]

    def flush(self):
        assert self.loss is not None and self.params, f"梯度用例 {self.name} 不完整"
        with torch.enable_grad():
            grads = torch.autograd.grad(self.loss.sum(), [self.tensors[p] for p in self.params])

        lines = [f"case {self.name} {len(self.decl)}"]
        for tid, tensor, type_name, _ in self.decl:
            ne = tuple(reversed(tensor.shape))
            assert len(ne) == 4, f"梯度用例 {self.name} 的输入必须为 4D npy 形状"
            lines.append(f"in {tid} {type_name} {ne[0]}x{ne[1]}x{ne[2]}x{ne[3]}")
        for tid in self.params:
            lines.append(f"param {tid}")
        for tid, op, attrs, srcs in self.nodes:
            attr_str = " ".join(str(v) for v in attrs)
            lines.append(" ".join(
                f"node {tid} {op} {len(attrs)} {attr_str} {len(srcs)} "
                f"{' '.join(str(s) for s in srcs)}".split()))
        lines.append(f"loss {self.loss_id}")

        np.save(os.path.join(self.out_dir, f"{self.name}_loss.npy"),
                np.ascontiguousarray(self.loss.detach().cpu().numpy()))
        for tid, tensor, _, _ in self.decl:
            np.save(os.path.join(self.out_dir, f"{self.name}_in{tid}.npy"),
                    np.ascontiguousarray(tensor.detach().cpu().numpy()))
        for tid, gr in zip(self.params, grads):
            np.save(os.path.join(self.out_dir, f"{self.name}_grad{tid}.npy"),
                    np.ascontiguousarray(gr.detach().cpu().numpy()))

        with open(os.path.join(self.out_dir, "grad_cases.txt"), "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  梯度用例 {self.name}: 输入 {len(self.decl)} 个（参数 {len(self.params)} 个）")


def grad_mlp_mse(out_dir):
    """两层 MLP（relu）+ MSE：覆盖 mul_mat/add 广播/relu/sub/sqr/sum/scale 的反向"""
    G = GradGraph("grad_mlp_mse", out_dir)
    x   = G.inp(rand([3, 4, 1, 1]))               # [in=3, batch=4]
    W1  = G.inp(rand([3, 5, 1, 1]), param=True)   # [in, hidden]
    b1  = G.inp(rand([5, 1, 1, 1]), param=True)   # [hidden, 1]
    W2  = G.inp(rand([5, 2, 1, 1]), param=True)   # [hidden, out]
    b2  = G.inp(rand([2, 1, 1, 1]), param=True)
    tgt = G.inp(rand([2, 4, 1, 1]))               # [out, batch]
    h = G.op("relu", G.op("add", G.op("mul_mat", W1, x), b1))
    y = G.op("add", G.op("mul_mat", W2, h), b2)
    G.set_loss(G.op("scale", G.op("sum", G.op("sqr", G.op("sub", y, tgt))), attrs=[1.0 / 8.0]))
    G.flush()


def grad_mul_mat_batch(out_dir):
    """mul_mat batch：a 广播到 b 的平面（dA 需沿 batch 归约）+ 整除广播（a2=2,b2=4）"""
    G = GradGraph("grad_mul_mat_batch", out_dir)
    a = G.inp(rand([3, 2, 1, 1]), param=True)   # [k=3, m=2, 1, 1]
    b = G.inp(rand([3, 4, 2, 2]), param=True)   # [k=3, n=4, b2=2, b3=2]
    G.set_loss(G.op("sum", G.op("sqr", G.op("mul_mat", a, b))))
    G.flush()

    Gd = GradGraph("grad_mul_mat_batch_div", out_dir)
    ad = Gd.inp(rand([2, 2, 2, 1]), param=True)  # [k=2, m=2, a2=2, 1]
    bd = Gd.inp(rand([2, 3, 4, 1]), param=True)  # [k=2, n=3, b2=4, 1]
    Gd.set_loss(Gd.op("sum", Gd.op("sqr", Gd.op("mul_mat", ad, bd))))
    Gd.flush()


def grad_conv_transpose_1d(out_dir):
    """conv_transpose_1d 反向（M2.3b）：dW/dX，stride=2 覆盖 im2col+mul_mat 组合"""
    G = GradGraph("grad_conv_transpose_1d", out_dir)
    w = G.inp(rand([3, 2, 2, 1]), param=True)  # [K=3, Cout=2, Cin=2]
    x = G.inp(rand([5, 2, 1, 1]), param=True)  # [T_in=5, Cin=2]
    y = G.op("conv_transpose_1d", w, x, attrs=[2, 0, 1])
    G.set_loss(G.op("sum", G.op("sqr", y)))
    G.flush()


def grad_pad_reflect(out_dir):
    """pad reflect 反向（M2.3c）：左右/上下非对称，覆盖折叠区叠加"""
    G = GradGraph("grad_pad_reflect", out_dir)
    x = G.inp(rand([4, 3, 1, 1]), param=True)  # [ne0=4, ne1=3]
    y = G.op("pad_reflect", x, attrs=[2, 1, 1, 1, 0, 0, 0, 0])
    G.set_loss(G.op("sum", G.op("sqr", y)))
    G.flush()


def grad_pool_2d(out_dir):
    """pool_2d 反向（M2.3g）：AVG（重叠窗口+填充）与 MAX（值间隔明确，避免 argmax 并列）"""
    for mode, name in ((1, "grad_pool_2d_avg"), (0, "grad_pool_2d_max")):
        G = GradGraph(name, out_dir)
        base = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) * 0.25
        x = G.inp(base, param=True)
        y = G.op("pool_2d", x, attrs=[mode, 2, 2, 1, 1, 1, 1])
        G.set_loss(G.op("sum", G.op("sqr", y)))
        G.flush()


def grad_conv_transpose_2d(out_dir):
    """conv_transpose_2d 反向（M2.3g）：stride=1 与 stride=2（含非整除过滤）"""
    G = GradGraph("grad_conv_transpose_2d", out_dir)
    w = G.inp(rand([2, 2, 2, 2]), param=True)  # [KW,KH,Cout,Cin]
    x = G.inp(rand([3, 2, 2, 1]), param=True)  # [W,H,Cin,N]
    y = G.op("conv_transpose_2d", w, x, attrs=[1])
    G.set_loss(G.op("sum", G.op("sqr", y)))
    G.flush()

    G2 = GradGraph("grad_conv_transpose_2d_s2", out_dir)
    w2 = G2.inp(rand([3, 2, 2, 2]), param=True)
    x2 = G2.inp(rand([3, 2, 2, 1]), param=True)
    y2 = G2.op("conv_transpose_2d", w2, x2, attrs=[2])
    G2.set_loss(G2.op("sum", G2.op("sqr", y2)))
    G2.flush()


def grad_norm_gelu(out_dir):
    """norm + gelu_erf + 逐元素乘：覆盖 norm_back 与 gelu_erf 反向组合"""
    G = GradGraph("grad_norm_gelu", out_dir)
    x = G.inp(rand([4, 3, 1, 1]), param=True)
    w = G.inp(rand([4, 3, 1, 1]), param=True)
    n = G.op("norm", x, attrs=[1e-5])
    G.set_loss(G.op("sum", G.op("mul", G.op("gelu_erf", n), w)))
    G.flush()


def grad_conv2d(out_dir):
    """conv_2d + relu + sqr：覆盖 im2col_back（输入梯度）与卷积核梯度"""
    G = GradGraph("grad_conv2d", out_dir)
    k   = G.inp(rand([3, 2, 2, 1]), param=True)    # [KW,KH,IC,OC]
    img = G.inp(rand([4, 4, 2, 1]), param=True)    # [W,H,IC,N]
    c = G.op("conv_2d", k, img, attrs=[1, 1, 1, 1, 1, 1])
    G.set_loss(G.op("sum", G.op("sqr", G.op("relu", c))))
    G.flush()


def grad_softmax(out_dir):
    """soft_max_ext(scale,mask) + 逐元素乘：覆盖 softmax 反向公式"""
    G = GradGraph("grad_softmax", out_dir)
    x = G.inp(rand([5, 3, 1, 1]), param=True)
    m = G.inp(rand([5, 3, 1, 1]))
    sm = G.op("soft_max_ext", x, m, attrs=[0.7, 0.0])
    G.set_loss(G.op("sum", G.op("mul", sm, m)))
    G.flush()


def grad_embedding(out_dir):
    """get_rows（embedding）+ sqr：覆盖 scatter-add 表梯度"""
    G = GradGraph("grad_embedding", out_dir)
    table = G.inp(rand([4, 5, 1, 1]), param=True)  # [ne0=4, n_vocab=5]
    idx = G.inp(torch.randint(0, 5, (1, 1, 1, 4), dtype=torch.int32), type_name="i32")
    G.set_loss(G.op("sum", G.op("sqr", G.op("get_rows", table, idx))))
    G.flush()


def grad_broadcast(out_dir):
    """concat + 广播 add + mean：覆盖 repeat_back / concat / mean 反向"""
    G = GradGraph("grad_broadcast", out_dir)
    x = G.inp(rand([2, 3, 1, 1]), param=True)      # ne=[2,3]
    y = G.inp(rand([1, 3, 1, 1]), param=True)      # ne=[1,3]
    b = G.inp(rand([3, 1, 1, 1]), param=True)      # ne=[3,1] 沿 ne1 广播
    c = G.op("concat", x, y, attrs=[0])            # [3,3]
    z = G.op("add", c, b)
    G.set_loss(G.op("add", G.op("sum", G.op("sqr", z)), G.op("sum", G.op("mean", x))))
    G.flush()


def grad_cross_entropy(out_dir):
    """cross_entropy_loss（logits 为参数、软标签目标）：覆盖 CE 反向 kernel"""
    G = GradGraph("grad_cross_entropy", out_dir)
    logits = G.inp(rand([5, 3, 1, 1]), param=True)          # [nc=5, nr=3]
    tgt = G.inp(torch.softmax(rand([5, 3, 1, 1]), dim=-1))  # 软标签（非参数）
    G.set_loss(G.op("cross_entropy_loss", logits, tgt))
    G.flush()


def grad_cross_entropy_mlp(out_dir):
    """两层 MLP + cross_entropy_loss：覆盖 CE 反向经 mul_mat/relu 的链式传导"""
    G = GradGraph("grad_cross_entropy_mlp", out_dir)
    x  = G.inp(rand([3, 2, 1, 1]))                          # [in=3, batch=2]
    w1 = G.inp(rand([3, 4, 1, 1]), param=True)              # [in, hidden]
    b1 = G.inp(rand([4, 1, 1, 1]), param=True)              # [hidden, 1]
    w2 = G.inp(rand([4, 3, 1, 1]), param=True)              # [hidden, classes]
    tgt = G.inp(torch.softmax(rand([3, 2, 1, 1]), dim=-1))  # [classes, batch]
    h = G.op("relu", G.op("add", G.op("mul_mat", w1, x), b1))
    logits = G.op("mul_mat", w2, h)
    G.set_loss(G.op("cross_entropy_loss", logits, tgt))
    G.flush()


def build_grad(out_dir):
    manifest = os.path.join(out_dir, "grad_cases.txt")
    if os.path.exists(manifest):
        os.remove(manifest)
    with torch.enable_grad():
        grad_mlp_mse(out_dir)
        grad_mul_mat_batch(out_dir)
        grad_conv_transpose_1d(out_dir)
        grad_pad_reflect(out_dir)
        grad_pool_2d(out_dir)
        grad_conv_transpose_2d(out_dir)
        grad_norm_gelu(out_dir)
        grad_conv2d(out_dir)
        grad_softmax(out_dir)
        grad_embedding(out_dir)
        grad_broadcast(out_dir)
        grad_cross_entropy(out_dir)
        grad_cross_entropy_mlp(out_dir)
    with open(manifest, "r", encoding="utf-8") as f:
        n = sum(1 for line in f if line.startswith("case "))
    print(f"已生成 {n} 个梯度用例 -> {manifest}")


# ---------------------------------------------------------------- 优化器/调度器对拍（M1.5a）
# 输出：
#   optim_<case>_meta.txt                  超参与步数（key value 行）
#   optim_<case>_p<i>_init.npy             参数初值（形状 = ne 反转）
#   optim_<case>_p<i>_grads.npy            每步梯度（形状 (steps,)+torch 形状）
#   optim_<case>_p<i>_traj.npy             每步参数（形状 (steps+1,)+torch 形状，首项为初值）
#   optim_sched_<case>_meta.txt/_lr.npy    调度器学习率序列（长度 steps+1，首项为初始 lr）
#
# 数值语义与 torch.optim.SGD / torch.optim.AdamW、torch.optim.lr_scheduler.* 对齐；
# C++ 侧由 tests/test_optim.cpp 逐步对拍。warmup_cosine 为本库扩展，两侧独立实现公式。

def _write_meta(out_dir, name, kv):
    with open(os.path.join(out_dir, f"optim_{name}_meta.txt"), "w", encoding="utf-8") as f:
        for k, v in kv.items():
            f.write(f"{k} {v}\n")


def build_optim(out_dir):
    steps = 6
    cases = [
        ("sgd_plain", "sgd", dict(lr=0.1), [[3, 2, 1, 1], [4, 1, 1, 1]]),
        ("sgd_momentum", "sgd", dict(lr=0.1, momentum=0.9), [[3, 2, 1, 1], [4, 1, 1, 1]]),
        ("sgd_nesterov", "sgd", dict(lr=0.05, momentum=0.9, nesterov=True), [[3, 2, 1, 1], [4, 1, 1, 1]]),
        ("sgd_wd", "sgd", dict(lr=0.02, momentum=0.8, dampening=0.2, weight_decay=0.01),
         [[3, 2, 1, 1], [4, 1, 1, 1]]),
        ("adamw_basic", "adamw", dict(lr=1e-2, weight_decay=1e-2), [[3, 2, 1, 1], [4, 1, 1, 1]]),
        ("adamw_nowd", "adamw", dict(lr=5e-3, beta1=0.8, beta2=0.99, weight_decay=0.0),
         [[3, 2, 1, 1], [4, 1, 1, 1]]),
    ]

    for name, kind, hp, shapes in cases:
        torch.manual_seed(0x5A17 + len(name))
        init = [torch.randn(*rev(sh), dtype=torch.float32) * 0.5 for sh in shapes]
        grads = [[torch.randn_like(p) for p in init] for _ in range(steps)]

        ps = [p.clone().requires_grad_(True) for p in init]
        if kind == "sgd":
            opt = torch.optim.SGD(ps, lr=hp["lr"], momentum=hp.get("momentum", 0.0),
                                  dampening=hp.get("dampening", 0.0),
                                  weight_decay=hp.get("weight_decay", 0.0),
                                  nesterov=hp.get("nesterov", False))
        else:
            opt = torch.optim.AdamW(ps, lr=hp["lr"],
                                    betas=(hp.get("beta1", 0.9), hp.get("beta2", 0.999)),
                                    eps=hp.get("eps", 1e-8),
                                    weight_decay=hp.get("weight_decay", 0.0))
        traj = [[p.detach().clone() for p in ps]]
        for t in range(steps):
            for p, g in zip(ps, grads[t]):
                p.grad = g.clone()
            opt.step()
            opt.zero_grad(set_to_none=True)
            traj.append([p.detach().clone() for p in ps])

        meta = {
            "kind": kind,
            "lr": hp["lr"],
            "momentum": hp.get("momentum", 0.0),
            "dampening": hp.get("dampening", 0.0),
            "weight_decay": hp.get("weight_decay", 0.0),
            "nesterov": 1 if hp.get("nesterov", False) else 0,
            "beta1": hp.get("beta1", 0.9),
            "beta2": hp.get("beta2", 0.999),
            "eps": hp.get("eps", 1e-8),
            "steps": steps,
            "nparams": len(init),
        }
        for i, sh in enumerate(shapes):
            meta[f"shape{i}"] = " ".join(str(v) for v in sh)
            np.save(os.path.join(out_dir, f"optim_{name}_p{i}_init.npy"),
                    np.ascontiguousarray(init[i].numpy()))
            np.save(os.path.join(out_dir, f"optim_{name}_p{i}_grads.npy"),
                    np.ascontiguousarray(torch.stack([grads[t][i] for t in range(steps)]).numpy()))
            np.save(os.path.join(out_dir, f"optim_{name}_p{i}_traj.npy"),
                    np.ascontiguousarray(torch.stack([traj[t][i] for t in range(steps + 1)]).numpy()))
        _write_meta(out_dir, name, meta)
        print(f"  优化器用例 {name}: {kind}, {steps} 步, {len(init)} 个参数")


def build_optim_sched(out_dir):
    steps = 6
    sched_cases = [
        ("step", dict(lr=0.1, step_size=2, gamma=0.5)),
        ("exponential", dict(lr=0.1, gamma=0.9)),
        ("cosine", dict(lr=0.1, t_max=5, eta_min=0.01)),
    ]
    for name, hp in sched_cases:
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=hp["lr"])
        opt.step()  # 让调度器认为优化器已在工作（避免 PyTorch 顺序告警）
        if name == "step":
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=hp["step_size"], gamma=hp["gamma"])
        elif name == "exponential":
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=hp["gamma"])
        else:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["t_max"], eta_min=hp["eta_min"])
        lrs = [float(opt.param_groups[0]["lr"])]
        for _ in range(steps):
            sch.step()
            lrs.append(float(opt.param_groups[0]["lr"]))
        np.save(os.path.join(out_dir, f"optim_sched_{name}_lr.npy"),
                np.asarray(lrs, dtype=np.float32))
        _write_meta(out_dir, f"sched_{name}", {
            "kind": name, "lr": hp["lr"], "steps": steps,
            "step_size": hp.get("step_size", 1), "gamma": hp.get("gamma", 1.0),
            "t_max": hp.get("t_max", 1), "eta_min": hp.get("eta_min", 0.0),
            "warmup": 0, "total": 0,
        })
        print(f"  调度器用例 {name}: lr {lrs[0]:.6g} -> {lrs[-1]:.6g}")

    # warmup_cosine（本库扩展）：t 从 1 开始（t=0 为初始 base lr）
    base, warmup, total, eta_min = 0.1, 2, 6, 0.01

    def warmup_cosine(t):
        if t <= warmup:
            return base * t / warmup if warmup > 0 else base
        remain = total - warmup
        if remain <= 0:
            return base
        x = min(1.0, (t - warmup) / remain)
        return eta_min + (base - eta_min) * (1 + math.cos(math.pi * x)) / 2

    lrs = [base] + [warmup_cosine(t) for t in range(1, steps + 1)]
    np.save(os.path.join(out_dir, "optim_sched_warmup_cosine_lr.npy"),
            np.asarray(lrs, dtype=np.float32))
    _write_meta(out_dir, "sched_warmup_cosine", {
        "kind": "warmup_cosine", "lr": base, "steps": steps,
        "step_size": 1, "gamma": 1.0, "t_max": total, "eta_min": eta_min,
        "warmup": warmup, "total": total,
    })
    print(f"  调度器用例 warmup_cosine: lr {lrs[0]:.6g} -> {lrs[-1]:.6g}")


# ---------------------------------------------------------------- nn 层对拍（M1.5b）
# 输出（每个用例）：
#   nn_<name>_meta.txt / _w.npy / [_b.npy] / _x.npy / _out.npy
# 权重布局与 PyTorch state_dict 一致（npy 形状 = torch 形状；本库 ne = 反转）
# C++ 侧由 tests/test_nn.cpp 用对应层的前向对拍。

def _save(out_dir, name, suffix, t):
    np.save(os.path.join(out_dir, f"nn_{name}_{suffix}.npy"),
            np.ascontiguousarray(t.detach().cpu().numpy()))


def build_nn(out_dir):
    def meta(name, **kv):
        with open(os.path.join(out_dir, f"nn_{name}_meta.txt"), "w", encoding="utf-8") as f:
            for k, v in kv.items():
                f.write(f"{k} {v}\n")

    # ---- Linear（带/不带 bias）----
    in_f, out_f, batch = 3, 4, 2
    w = torch.randn(out_f, in_f) * 0.5
    b = torch.randn(out_f) * 0.5
    x = torch.randn(batch, in_f)
    _save(out_dir, "linear_bias", "w", w)
    _save(out_dir, "linear_bias", "b", b)
    _save(out_dir, "linear_bias", "x", x)
    _save(out_dir, "linear_bias", "out", x @ w.t() + b)
    meta("linear_bias", kind="linear", has_bias=1, in_features=in_f, out_features=out_f)

    _save(out_dir, "linear_nobias", "w", w)
    _save(out_dir, "linear_nobias", "x", x)
    _save(out_dir, "linear_nobias", "out", x @ w.t())
    meta("linear_nobias", kind="linear", has_bias=0, in_features=in_f, out_features=out_f)

    # ---- Embedding ----
    n_vocab, dim, nr = 6, 4, 5
    table = torch.randn(n_vocab, dim) * 0.5
    idx = torch.randint(0, n_vocab, (nr,), dtype=torch.int32)
    _save(out_dir, "embedding", "w", table)
    _save(out_dir, "embedding", "x", idx)
    _save(out_dir, "embedding", "out", table[idx.long()])
    meta("embedding", kind="embedding", has_bias=0, num_embeddings=n_vocab, embedding_dim=dim)

    # ---- Conv1d ----
    ic, oc, k, n, ww, si, pa, di = 2, 2, 3, 2, 6, 1, 1, 1
    wc = torch.randn(oc, ic, k) * 0.5
    bc = torch.randn(oc) * 0.5
    xc = torch.randn(n, ic, ww)
    yc = F.conv1d(xc, wc, bc, stride=si, padding=pa, dilation=di)
    _save(out_dir, "conv1d_bias", "w", wc)
    _save(out_dir, "conv1d_bias", "b", bc)
    _save(out_dir, "conv1d_bias", "x", xc)
    _save(out_dir, "conv1d_bias", "out", yc)
    meta("conv1d_bias", kind="conv1d", has_bias=1, in_channels=ic, out_channels=oc,
         kernel=k, stride0=si, padding0=pa, dilation0=di)

    # ---- grouped Conv1d（groups=2/3，N=2 覆盖 batch 路径）----
    ic, oc, k, n, ww, groups = 4, 4, 3, 2, 5, 2
    wg = torch.randn(oc, ic // groups, k) * 0.5   # torch (OC, IC/g, K)
    bg = torch.randn(oc) * 0.5
    xg = torch.randn(n, ic, ww)
    yg = F.conv1d(xg, wg, bg, groups=groups)
    _save(out_dir, "conv1d_grouped2", "w", wg)
    _save(out_dir, "conv1d_grouped2", "b", bg)
    _save(out_dir, "conv1d_grouped2", "x", xg)
    _save(out_dir, "conv1d_grouped2", "out", yg)
    meta("conv1d_grouped2", kind="conv1d_grouped", has_bias=1, in_channels=ic, out_channels=oc,
         kernel=k, stride0=1, padding0=0, dilation0=1, num_groups=groups)

    ic, oc, k, n, ww, groups = 6, 3, 2, 2, 6, 3
    wg3 = torch.randn(oc, ic // groups, k) * 0.5
    bg3 = torch.randn(oc) * 0.5
    xg3 = torch.randn(n, ic, ww)
    yg3 = F.conv1d(xg3, wg3, bg3, groups=groups)
    _save(out_dir, "conv1d_grouped3", "w", wg3)
    _save(out_dir, "conv1d_grouped3", "b", bg3)
    _save(out_dir, "conv1d_grouped3", "x", xg3)
    _save(out_dir, "conv1d_grouped3", "out", yg3)
    meta("conv1d_grouped3", kind="conv1d_grouped", has_bias=1, in_channels=ic, out_channels=oc,
         kernel=k, stride0=1, padding0=0, dilation0=1, num_groups=groups)

    # ---- Conv2d ----
    ic, oc, kh, kw, n, hh, ww = 2, 2, 2, 3, 1, 4, 5
    si, pa = 1, 1
    wc2 = torch.randn(oc, ic, kh, kw) * 0.5
    bc2 = torch.randn(oc) * 0.5
    xc2 = torch.randn(n, ic, hh, ww)
    yc2 = F.conv2d(xc2, wc2, bc2, stride=si, padding=pa, dilation=1)
    _save(out_dir, "conv2d_bias", "w", wc2)
    _save(out_dir, "conv2d_bias", "b", bc2)
    _save(out_dir, "conv2d_bias", "x", xc2)
    _save(out_dir, "conv2d_bias", "out", yc2)
    meta("conv2d_bias", kind="conv2d", has_bias=1, in_channels=ic, out_channels=oc,
         kernel_w=kw, kernel_h=kh, stride0=si, stride1=si, padding0=pa, padding1=pa,
         dilation0=1, dilation1=1)

    # ---- LayerNorm / RMSNorm ----
    eps = 1e-5
    rows, nrm = 3, 4
    xs = torch.randn(rows, nrm)
    gw = torch.randn(nrm)
    gb = torch.randn(nrm)
    _save(out_dir, "layernorm", "w", gw)
    _save(out_dir, "layernorm", "b", gb)
    _save(out_dir, "layernorm", "x", xs)
    _save(out_dir, "layernorm", "out", F.layer_norm(xs, (nrm,), gw, gb, eps))
    meta("layernorm", kind="layernorm", has_bias=1, normalized_size=nrm, eps=eps)

    _save(out_dir, "rmsnorm", "w", gw)
    _save(out_dir, "rmsnorm", "x", xs)
    _save(out_dir, "rmsnorm", "out",
          xs * torch.rsqrt(xs.pow(2).mean(dim=-1, keepdim=True) + eps) * gw)
    meta("rmsnorm", kind="rmsnorm", has_bias=0, normalized_size=nrm, eps=eps)

    # ---- GroupNorm（npy 形状 (N,C,H,W)，本库 ne=[W,H,C,N]）----
    n, c, hh2, ww2, groups = 2, 4, 3, 2, 2
    xg = torch.randn(n, c, hh2, ww2)
    ggw = torch.randn(c)
    ggb = torch.randn(c)
    _save(out_dir, "groupnorm", "w", ggw)
    _save(out_dir, "groupnorm", "b", ggb)
    _save(out_dir, "groupnorm", "x", xg)
    _save(out_dir, "groupnorm", "out", F.group_norm(xg, groups, ggw, ggb, eps))
    meta("groupnorm", kind="groupnorm", has_bias=1, num_groups=groups, num_channels=c, eps=eps)

    # ---- ConvTranspose1d（stride=1 与 stride=2+padding 裁剪；N=2 覆盖层内批次组合）----
    ic, oc, k, t_in, n = 2, 2, 3, 6, 2
    wt = torch.randn(ic, oc, k) * 0.5   # torch (Cin, Cout, K) → 本库 ne=[K,Cout,Cin]
    bt = torch.randn(oc) * 0.5
    xt = torch.randn(n, ic, t_in)
    yt = F.conv_transpose1d(xt, wt, bt, stride=1, padding=0)
    _save(out_dir, "convtranspose1d_bias", "w", wt)
    _save(out_dir, "convtranspose1d_bias", "b", bt)
    _save(out_dir, "convtranspose1d_bias", "x", xt)
    _save(out_dir, "convtranspose1d_bias", "out", yt)
    meta("convtranspose1d_bias", kind="convtranspose1d", has_bias=1, in_channels=ic,
         out_channels=oc, kernel=k, stride0=1, padding0=0)

    ic, oc, k, t_in, n = 2, 3, 4, 5, 2
    st, pa = 2, 1
    wt2 = torch.randn(ic, oc, k) * 0.5
    bt2 = torch.randn(oc) * 0.5
    xt2 = torch.randn(n, ic, t_in)
    yt2 = F.conv_transpose1d(xt2, wt2, bt2, stride=st, padding=pa)
    _save(out_dir, "convtranspose1d_stride2_pad", "w", wt2)
    _save(out_dir, "convtranspose1d_stride2_pad", "b", bt2)
    _save(out_dir, "convtranspose1d_stride2_pad", "x", xt2)
    _save(out_dir, "convtranspose1d_stride2_pad", "out", yt2)
    meta("convtranspose1d_stride2_pad", kind="convtranspose1d", has_bias=1, in_channels=ic,
         out_channels=oc, kernel=k, stride0=st, padding0=pa)

    # ---- WeightNorm（w = g·v/||v||；范数按输出通道，即 torch weight_norm(dim=0) 语义）----
    # 约定：npy 的 w 槽位存 v、b 槽位存 g（C++ 端 kind 前缀为 weightnorm_ 时按此读取）
    out_f, in_f, batch = 3, 4, 2
    vv = torch.randn(out_f, in_f) * 0.5
    nv = torch.norm(vv, dim=1, keepdim=True)   # (out,1)：对除 dim0（输出通道）外的维求范数
    gg = nv + 0.1
    xx = torch.randn(batch, in_f)
    ww_ = gg * vv / nv
    _save(out_dir, "weightnorm_linear", "w", vv)
    _save(out_dir, "weightnorm_linear", "b", gg)
    _save(out_dir, "weightnorm_linear", "x", xx)
    _save(out_dir, "weightnorm_linear", "out", xx @ ww_.t())
    meta("weightnorm_linear", kind="weightnorm_linear", has_bias=0, in_features=in_f,
         out_features=out_f)

    oc, ic, k, n, ww = 3, 2, 3, 2, 6
    vc = torch.randn(oc, ic, k) * 0.5
    nvc = torch.norm(vc, dim=(1, 2), keepdim=True)   # (OC,1,1)
    gc = nvc + 0.1
    xc = torch.randn(n, ic, ww)
    wc_ = gc * vc / nvc
    _save(out_dir, "weightnorm_conv1d", "w", vc)
    _save(out_dir, "weightnorm_conv1d", "b", gc)
    _save(out_dir, "weightnorm_conv1d", "x", xc)
    _save(out_dir, "weightnorm_conv1d", "out", F.conv1d(xc, wc_))
    meta("weightnorm_conv1d", kind="weightnorm_conv1d", has_bias=0, in_channels=ic,
         out_channels=oc, kernel=k, stride0=1, padding0=0)

    # ---- WeightNorm × 分组卷积（M3.1）：先按输出通道重参数化，再做 grouped conv1d ----
    # 注意：局部重设种子以隔离随机流，保证既有用例（含其后 build_loss 等）逐位不变
    torch.manual_seed(20260923)
    ic, oc, k, n, ww, groups = 4, 4, 3, 2, 6, 2
    vg = torch.randn(oc, ic // groups, k) * 0.5
    nvg = torch.norm(vg, dim=(1, 2), keepdim=True)   # (OC,1,1)：对 IC/g 与 K 求范数
    gg = nvg + 0.1
    xg = torch.randn(n, ic, ww)
    wg_ = gg * vg / nvg
    _save(out_dir, "weightnorm_conv1d_grouped2", "w", vg)
    _save(out_dir, "weightnorm_conv1d_grouped2", "b", gg)
    _save(out_dir, "weightnorm_conv1d_grouped2", "x", xg)
    _save(out_dir, "weightnorm_conv1d_grouped2", "out",
          F.conv1d(xg, wg_, stride=1, padding=1, groups=groups))
    meta("weightnorm_conv1d_grouped2", kind="weightnorm_conv1d", has_bias=0, in_channels=ic,
         out_channels=oc, kernel=k, stride0=1, padding0=1, num_groups=groups)

    print("  已生成 nn 层对拍用例：linear/embedding/conv1d/conv2d/layernorm/rmsnorm/groupnorm"
          "/convtranspose1d/grouped Conv1d/weightnorm（含分组 weightnorm）")


# ---------------------------------------------------------------- 损失对拍（M1.5c）
# 输出（每个用例）：loss_<kind>_pred.npy / loss_<kind>_target.npy / loss_<kind>_out.npy
# reduction 与 PyTorch 默认一致（mean：全元素均值）；C++ 侧由 tests/test_loss.cpp 对拍。

def build_loss(out_dir):
    pred = rand([4, 3, 2, 1])
    target = rand([4, 3, 2, 1])
    loss_mse = ((pred - target) ** 2).mean().reshape(1, 1, 1, 1)
    loss_l1 = (pred - target).abs().mean().reshape(1, 1, 1, 1)
    for kind, out in (("mse", loss_mse), ("l1", loss_l1)):
        np.save(os.path.join(out_dir, f"loss_{kind}_pred.npy"), np.ascontiguousarray(pred.numpy()))
        np.save(os.path.join(out_dir, f"loss_{kind}_target.npy"), np.ascontiguousarray(target.numpy()))
        np.save(os.path.join(out_dir, f"loss_{kind}_out.npy"), np.ascontiguousarray(out.numpy()))
    print("  已生成损失对拍用例：mse/l1")


# ---------------------------------------------------------------- 逐 epoch 训练轨迹对拍（M2.0d）
# 固定小 MLP（Linear(HID,IN)+ReLU+Linear(OUT,HID)，MSE）在全批量上训练多个 epoch；
# 每个 epoch 结束保存参数与评估 loss（前向，不含更新）。C++ 侧重放同一训练循环并逐 epoch 对拍。
# 输出（每个用例）：
#   trace_<case>_meta.txt            超参与形状（key value 行）
#   trace_<case>_x.npy               [batch, in]
#   trace_<case>_target.npy          [batch, out]
#   trace_<case>_p<i>_init.npy       参数初值（torch 形状；i: 0=w1 1=b1 2=w2 3=b2）
#   trace_<case>_p<i>_epoch<e>.npy   epoch e（1 起）结束后的参数
#   trace_<case>_loss_epoch<e>.npy   评估 loss（标量，形状 (1,)）

def build_train_trace(out_dir):
    IN, HID, OUT, BATCH = 4, 8, 3, 8
    EPOCHS, STEPS = 3, 4

    torch.manual_seed(20260922)
    x = torch.randn(BATCH, IN)
    target = torch.randn(BATCH, OUT)

    cases = [
        ("adamw", "adamw",
         dict(lr=0.02, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8), 0x7EACE),
        ("sgd_momentum", "sgd",
         dict(lr=0.05, momentum=0.9, dampening=0.0, weight_decay=0.0), 0x7EACF),
    ]
    for name, kind, hp, seed in cases:
        torch.manual_seed(seed)
        init = [torch.randn(HID, IN) * 0.5, torch.randn(HID) * 0.5,
                torch.randn(OUT, HID) * 0.5, torch.randn(OUT) * 0.5]
        ps = [p.clone().requires_grad_(True) for p in init]
        if kind == "adamw":
            opt = torch.optim.AdamW(ps, lr=hp["lr"], betas=(hp["beta1"], hp["beta2"]),
                                    eps=hp["eps"], weight_decay=hp["weight_decay"])
        else:
            opt = torch.optim.SGD(ps, lr=hp["lr"], momentum=hp["momentum"],
                                  dampening=hp["dampening"], weight_decay=hp["weight_decay"])

        def forward():
            h = torch.relu(x @ ps[0].t() + ps[1])
            return h @ ps[2].t() + ps[3]

        np.save(os.path.join(out_dir, f"trace_{name}_x.npy"), np.ascontiguousarray(x.numpy()))
        np.save(os.path.join(out_dir, f"trace_{name}_target.npy"), np.ascontiguousarray(target.numpy()))

        # 本脚本全局 set_grad_enabled(False)（前向用例用），训练段需显式开启梯度
        with torch.no_grad():
            ev = float(((forward() - target) ** 2).mean())
        print(f"  训练轨迹用例 {name}: {kind}, {EPOCHS} epoch x {STEPS} step, loss {ev:.6g}", end="")
        for e in range(1, EPOCHS + 1):
            for _ in range(STEPS):
                with torch.enable_grad():
                    loss = ((forward() - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                ev = float(((forward() - target) ** 2).mean())
            for i, p in enumerate(ps):
                np.save(os.path.join(out_dir, f"trace_{name}_p{i}_epoch{e}.npy"),
                        np.ascontiguousarray(p.detach().numpy()))
            np.save(os.path.join(out_dir, f"trace_{name}_loss_epoch{e}.npy"),
                    np.asarray([ev], dtype=np.float32))
        for i, p in enumerate(init):
            np.save(os.path.join(out_dir, f"trace_{name}_p{i}_init.npy"),
                    np.ascontiguousarray(p.numpy()))
        print(f" -> {ev:.6g}")

        meta = {
            "kind": kind, "lr": hp["lr"], "weight_decay": hp.get("weight_decay", 0.0),
            "momentum": hp.get("momentum", 0.0), "dampening": hp.get("dampening", 0.0),
            "beta1": hp.get("beta1", 0.9), "beta2": hp.get("beta2", 0.999),
            "eps": hp.get("eps", 1e-8), "nesterov": 0,
            "epochs": EPOCHS, "steps": STEPS,
            "in": IN, "hid": HID, "out": OUT, "batch": BATCH, "nparams": len(init),
        }
        with open(os.path.join(out_dir, f"trace_{name}_meta.txt"), "w", encoding="utf-8") as f:
            for k, v in meta.items():
                f.write(f"{k} {v}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.abspath(args.out)
    g = Golden(out_dir)
    build(g)
    g.flush()

    build_grad(out_dir)
    build_optim(out_dir)
    build_optim_sched(out_dir)
    build_nn(out_dir)
    build_loss(out_dir)
    build_train_trace(out_dir)


if __name__ == "__main__":
    sys.exit(main())
