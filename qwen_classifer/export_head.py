# -*- coding: utf-8 -*-
"""导出线性属性头为板端 C++ 可加载的 attr_head.bin。

格式（显式小端）：
    6s  magic   b"AHEAD1"
    u32 version = 1
    i32 in_dim  (2048)
    i32 num_attrs (54)
    f32 thr[num_attrs]            # 逐属性最优 F1 阈值（prob >= thr 判 1，与 eval_head 语义一致）
    names: num_attrs × (u32 len + utf8 bytes)
    f32 W[num_attrs × in_dim]     # row-major，行 j = 属性 j
    f32 b[num_attrs]

例：
    python export_head.py --ckpt runs/full_448px/best.pt \
        --cache "cache/val_s*_c*.pt" --out ../web_vllm_rknn/output_cpp/models/attr_head.bin
    python export_head.py --verify --bin <上式产物> --cache "cache/val_s*_c*.pt"
"""
import argparse
import glob
import json
import struct

import numpy as np
import torch

from common import ATTRS, NUM_ATTRS, compute_map, per_attribute_accuracy
from train_head import Head, expand_cache_paths

MAGIC = b"AHEAD1"
VERSION = 1


def load_head(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    assert ck.get("hidden", 0) == 0, f"仅支持线性头，实际 hidden={ck.get('hidden')}"
    assert not ck.get("use_deepstack", False), "训练用了 deepstack 特征，需另行处理"
    in_dim = int(ck["in_dim"])
    model = Head(in_dim, hidden=0)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    W = model.net.weight.detach().numpy().astype(np.float32)   # [54, in_dim]
    b = model.net.bias.detach().numpy().astype(np.float32)
    assert list(ck.get("attrs", ATTRS)) == ATTRS, "checkpoint 属性表与 common.ATTRS 不一致"
    return W, b, in_dim, ck


def iter_caches(patterns):
    for p in expand_cache_paths(list(patterns)):
        d = torch.load(p, map_location="cpu")
        yield p, d["feats"].float(), d["labels"].long()
        del d


def score_all(W, b, cache_patterns):
    """流式打分：-> scores [N,54] float32, labels [N,54] int8"""
    scores, labels = [], []
    for p, feats, lab in iter_caches(cache_patterns):
        x = feats.numpy().astype(np.float32)
        del feats
        logits = x @ W.T + b
        del x
        scores.append(1.0 / (1.0 + np.exp(-logits)).astype(np.float32))
        del logits
        labels.append(lab.numpy().astype(np.int8))
        del lab
        print(f"scored {p}", flush=True)
    return np.concatenate(scores), np.concatenate(labels)


def best_f1_threshold(scores, y, mask):
    """与 eval_head.best_f1_threshold 完全一致：网格 [0.05,0.95]，判定 prob >= thr。"""
    ths = np.arange(0.05, 0.96, 0.05)
    s, t = scores[mask], y[mask]
    best = (0.5, 0.0)
    for th in ths:
        pred = s >= th
        tp = (pred & (t == 1)).sum()
        fp = (pred & (t == 0)).sum()
        fn = ((~pred) & (t == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best[1]:
            best = (float(th), float(f1))
    return best


def write_bin(path, W, b, thr):
    n, d = W.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<6sIII", MAGIC, VERSION, d, n))
        f.write(struct.pack(f"<{n}f", *thr))
        for a in ATTRS:
            raw = a.encode("utf-8")
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
        f.write(W.astype("<f4").tobytes())
        f.write(b.astype("<f4").tobytes())
    print(f"wrote {path} ({n}×{d}, {n} thresholds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/full_448px/best.pt")
    ap.add_argument("--cache", nargs="+", default=["cache/val_s*_c*.pt"])
    ap.add_argument("--out", default="../web_vllm_rknn/output_cpp/models/attr_head.bin")
    ap.add_argument("--verify", action="store_true", help="重载 bin 全 val 重打分复现指标")
    ap.add_argument("--bin", dest="bin_path", default=None)
    args = ap.parse_args()

    if args.verify:
        bin_path = args.bin_path or args.out
        W, b, thr, names = read_bin(bin_path)
        assert names == ATTRS, "bin 属性表与 common.ATTRS 不一致"
        scores, labels = score_all(W, b, args.cache)
        mask = labels >= 0
        y = (labels == 1).astype(np.int32)
        map_, _ = compute_map(scores, labels)
        mean_acc, _ = per_attribute_accuracy(scores, labels)
        # 用 bin 内阈值再给一遍 macro 判 1 的准确率，贴近板端行为
        pred = scores >= np.asarray(thr, dtype=np.float32)[None, :]
        acc_thr = float(((pred == (y == 1)) & mask).sum() / max(mask.sum(), 1))
        print(f"VERIFY {bin_path}\n  mAP={map_:.4f}  Acc@0.5={mean_acc:.4f}  Acc@bin_thr={acc_thr:.4f}")
        return

    W, b, in_dim, ck = load_head(args.ckpt)
    print(f"head {in_dim}->{NUM_ATTRS}, train best mAP={ck.get('val_map')}")

    scores, labels = score_all(W, b, args.cache)
    assert abs(scores.shape[1] - NUM_ATTRS) == 0
    mask = labels >= 0
    y = (labels == 1).astype(np.int32)
    map_, _ = compute_map(scores, labels)
    mean_acc, _ = per_attribute_accuracy(scores, labels)
    print(f"val mAP={map_:.4f}  Acc@0.5={mean_acc:.4f}  (ckpt 记录 mAP={ck.get('val_map')})")

    thr, report = [], []
    for j, a in enumerate(ATTRS):
        mj = mask[:, j]
        if mj.sum() == 0:
            t, f1 = 0.5, 0.0
        else:
            t, f1 = best_f1_threshold(scores[:, j], y[:, j], mj)
        thr.append(t)
        report.append({"attr": a, "thr": t, "f1": round(f1, 4),
                       "pos": int((y[:, j] == 1)[mj].sum()), "valid": int(mj.sum())})
    with open("attr_head_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("阈值表: attr_head_report.json")

    write_bin(args.out, W, b, np.asarray(thr, dtype=np.float32))


def read_bin(path):
    with open(path, "rb") as f:
        magic, version, in_dim, n = struct.unpack("<6sIII", f.read(18))
        assert magic == MAGIC and version == VERSION, f"magic/version 不符: {magic} v{version}"
        thr = list(struct.unpack(f"<{n}f", f.read(4 * n)))
        names = []
        for _ in range(n):
            (ln,) = struct.unpack("<I", f.read(4))
            names.append(f.read(ln).decode("utf-8"))
        W = np.frombuffer(f.read(4 * n * in_dim), dtype="<f4").reshape(n, in_dim).copy()
        b = np.frombuffer(f.read(4 * n), dtype="<f4").copy()
        assert f.read(1) == b"", "bin 尾部有多余字节"
    return W, b, thr, names


if __name__ == "__main__":
    main()
