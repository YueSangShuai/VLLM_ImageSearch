# -*- coding: utf-8 -*-
"""导出 ProjMLP 为板端可加载的 vproj_head.bin（沿用 attr_head.bin 的显式小端惯例）。

格式 v1：
    6s   magic  b"VPROJ1"
    u32  version = 1
    i32  in_dim (2048), i32 hidden (1024), i32 out_dim (512)
    f32  ln_eps
    f32  W1[hidden x in_dim] (row-major), b1[hidden]
    f32  ln_gamma[hidden], ln_beta[hidden]
    f32  W2[out_dim x hidden], b2[out_dim]
    u32  has_text_proj; 若 1: f32 Wt[512x512], bt[512]

板端推理（与 PyTorch 对齐）:
    x -> x@W1.T+b1 -> LayerNorm(gamma,beta,eps) -> GELU -> h@W2.T+b2 -> L2 归一化
GELU 用 tanh 近似或与 PyTorch 精确式对齐均可，误差量级 <1e-3（导出校验会给出实测差）。

例：
    python export_head.py --ckpt runs/full/best.pt --out models/vproj_head.bin
    python export_head.py --verify --bin models/vproj_head.bin --cache "cache/test_i_*"

注意：--verify 的 ref 来自 --ckpt，必须与当初导出用的 ckpt/config 一致，
否则比的是两套权重（报 FAIL 属误用）。
"""
import argparse
import glob
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

from common import abspath, load_config
from model import load_model

MAGIC = b"VPROJ1"
VERSION = 1


def get_params(model):
    net = model.net
    l1, ln, _, l2 = net[0], net[1], net[2], net[3]
    parts = {
        "W1": l1.weight.detach().numpy().astype("<f4"),
        "b1": l1.bias.detach().numpy().astype("<f4"),
        "gamma": ln.weight.detach().numpy().astype("<f4"),
        "beta": ln.bias.detach().numpy().astype("<f4"),
        "eps": float(ln.eps),
        "W2": l2.weight.detach().numpy().astype("<f4"),
        "b2": l2.bias.detach().numpy().astype("<f4"),
    }
    if model.text_net is not None:
        parts["Wt"] = model.text_net.weight.detach().numpy().astype("<f4")
        parts["bt"] = model.text_net.bias.detach().numpy().astype("<f4")
    return parts


def write_bin(path, model, in_dim, hidden, out_dim):
    p = get_params(model)
    with open(path, "wb") as f:
        f.write(struct.pack("<6sIIII", MAGIC, VERSION, in_dim, hidden, out_dim))
        f.write(struct.pack("<f", p["eps"]))
        f.write(p["W1"].tobytes()); f.write(p["b1"].tobytes())
        f.write(p["gamma"].tobytes()); f.write(p["beta"].tobytes())
        f.write(p["W2"].tobytes()); f.write(p["b2"].tobytes())
        has_tp = "Wt" in p
        f.write(struct.pack("<I", 1 if has_tp else 0))
        if has_tp:
            f.write(p["Wt"].tobytes()); f.write(p["bt"].tobytes())
    print(f"wrote {path}  ({in_dim}->{hidden}->{out_dim}, text_proj={has_tp})")


def read_bin(path):
    with open(path, "rb") as f:
        magic, ver, in_dim, hidden, out_dim = struct.unpack("<6sIIII", f.read(22))
        assert magic == MAGIC and ver == VERSION, f"magic/version 不符: {magic} v{ver}"
        (eps,) = struct.unpack("<f", f.read(4))

        def rd(rows, cols):
            return np.frombuffer(f.read(4 * rows * cols), dtype="<f4").reshape(rows, cols).copy()

        def rd1(n):
            return np.frombuffer(f.read(4 * n), dtype="<f4").copy()

        p = {"W1": rd(hidden, in_dim), "b1": rd1(hidden),
             "gamma": rd1(hidden), "beta": rd1(hidden),
             "W2": rd(out_dim, hidden), "b2": rd1(out_dim), "eps": eps}
        (has_tp,) = struct.unpack("<I", f.read(4))
        if has_tp:
            p["Wt"] = rd(out_dim, out_dim); p["bt"] = rd1(out_dim)
        assert f.read(1) == b"", "bin 尾部有多余字节"
    return p


def np_forward(p, x):
    """numpy 复刻图像路径（GELU 精确 erf 式），x [N,in_dim] -> 归一化 [N,out_dim]"""
    from math import erf

    h = x @ p["W1"].T + p["b1"]
    mu = h.mean(-1, keepdims=True)
    var = h.var(-1, keepdims=True)          # torch LayerNorm 用有偏方差，numpy 默认一致
    h = (h - mu) / np.sqrt(var + p["eps"]) * p["gamma"] + p["beta"]
    gelu = np.vectorize(erf, otypes=[np.float64])
    h = 0.5 * h * (1 + gelu(h / np.sqrt(2.0)))
    o = h @ p["W2"].T + p["b2"]             # text_proj 只作用于文本侧，图像路径不含
    return o / np.maximum(np.linalg.norm(o, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default="runs/full/best.pt")
    ap.add_argument("--out", default="models/vproj_head.bin")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bin", dest="bin_path", default=None)
    ap.add_argument("--cache", nargs="+", default=["cache/test_i_*"])
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.verify:
        p = read_bin(abspath(args.bin_path or args.out))
        x = torch.cat([torch.load(gg, map_location="cpu")["feats"].float()
                       for pat in args.cache for gg in sorted(glob.glob(abspath(pat)))])
        x = x[:args.limit].numpy()
        ref_model, _ = load_model(abspath(args.ckpt), cfg, "cpu")
        with torch.no_grad():
            ref = F.normalize(ref_model.net(torch.from_numpy(x)), dim=-1).numpy()
        got = np_forward(p, x)
        diff = np.abs(ref - got).max()
        ok = diff < 5e-4
        print(f"VERIFY bin={args.bin_path or args.out}  max|Δ|={diff:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    model, _ = load_model(abspath(args.ckpt), cfg, "cpu")
    out = abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_bin(out, model, model.in_dim, model.hidden, model.out_dim)
    print("校验：python export_head.py --verify --bin", out)


if __name__ == "__main__":
    main()
