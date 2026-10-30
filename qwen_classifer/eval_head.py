# -*- coding: utf-8 -*-
"""最终评测：加载 best.pt + 特征缓存，输出总体 mAP 与逐属性 AP/最优阈值表。

例：
  python eval_head.py --ckpt runs/exp1/best.pt --cache cache/val.pt
"""
import argparse

import numpy as np
import torch

from common import ATTRS, average_precision, per_attribute_accuracy
from train_head import Head, load_caches


@torch.no_grad()
def predict(model, feats, device, bs=8192):
    outs = []
    for i in range(0, len(feats), bs):
        x = feats[i:i + bs].to(device).float()
        outs.append(torch.sigmoid(model(x)).float().cpu())
    return torch.cat(outs).numpy()


def best_f1_threshold(scores, y, mask):
    """在 [0.05, 0.95] 网格上找该属性最优 F1 阈值与对应 F1。"""
    ths = np.arange(0.05, 0.96, 0.05)
    s, t = scores[mask], y[mask]
    best = (0.5, 0.0)
    for th in ths:
        pred = s >= th
        tp = (pred & (t == 1)).sum()
        fp = (pred & (t == 0)).sum()
        fn = (~pred) & (t == 1)
        fn = fn.sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best[1]:
            best = (float(th), float(f1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", nargs="+", required=True, help="可给多个分片缓存")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    model = Head(ck["in_dim"], ck.get("hidden", 0)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    feats, labels_t, meta = load_caches(args.cache)
    assert meta["use_deepstack"] == ck["use_deepstack"], "deepstack 配置不一致"
    scores = predict(model, feats, device)
    labels = labels_t.numpy()
    mask = labels >= 0
    y = (labels == 1).astype(np.int32)

    mean_acc, accs = per_attribute_accuracy(scores, labels)  # 阈值 0.5
    aps, f1s, ths = {}, {}, {}
    for j, a in enumerate(ATTRS):
        apv = average_precision(scores[:, j].astype(np.float64), y[:, j], mask[:, j])
        th, f1 = best_f1_threshold(scores[:, j], y[:, j], mask[:, j])
        if apv is not None:
            aps[a] = apv
        f1s[a], ths[a] = f1, th

    rows = sorted(aps.items(), key=lambda kv: kv[1])
    print(f"{'attr':<20} {'AP':>8} {'Acc@.5':>8} {'F1@best':>9} {'thr':>5}  pos(val)")
    for a, apv in rows:
        npos = int((y[:, ATTRS.index(a)] == 1)[mask[:, ATTRS.index(a)]].sum())
        acc = accs.get(a, float('nan'))
        print(f"{a:<20} {apv:8.4f} {acc:8.4f} {f1s[a]:9.4f} {ths[a]:5.2f}  {npos}")

    macro_f1 = float(np.mean(list(f1s.values())))
    print(f"\n总样本 {len(labels)}  属性数 {len(aps)}")
    print(f"mAP      = {np.mean(list(aps.values())):.4f}")
    print(f"平均准确率(各类 Acc@0.5 求平均) = {mean_acc:.4f}")
    print(f"macro F1 = {macro_f1:.4f}  (best.pt 训练期 mAP: {ck.get('val_map')})")


if __name__ == "__main__":
    main()
