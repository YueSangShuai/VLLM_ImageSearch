# -*- coding: utf-8 -*-
"""阶段二：在缓存特征上训练分类头（方案 A 的核心训练循环）。

特征冻结缓存后训练一个线性/MLP 头，mask 掉 -1 未知标签，
加权 BCE 损失，按 PA-100K mAP 选最优 checkpoint。

例：
  python train_head.py --train-cache cache/train.pt --val-cache cache/val.pt
  python train_head.py --hidden 2048 --epochs 80   # 两层 MLP
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import ATTRS, NUM_ATTRS, compute_map, per_attribute_accuracy, valid_pos_weight


def expand_cache_paths(paths):
    """展开目录/通配符为具体 .pt 列表（支持 cache/train_s0_c*.pt 这类分块产物）。"""
    import glob as _glob
    import os
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(_glob.glob(os.path.join(p, "*.pt"))))
        elif any(c in p for c in "*?["):
            out.extend(sorted(_glob.glob(p)))
        else:
            out.append(p)
    out = [p for p in out if p.endswith(".pt")]
    if not out:
        raise FileNotFoundError(f"未匹配到任何 .pt 缓存：{paths}")
    return out


def load_caches(paths):
    """加载一个/多个（分块）特征缓存并拼接进内存。"""
    paths = expand_cache_paths(paths)
    feats, labels = [], []
    for p in paths:
        d = torch.load(p)
        feats.append(d["feats"])
        labels.append(d["labels"].long())
        del d
    meta = torch.load(paths[0])
    print(f"loaded {len(paths)} cache files")
    return torch.cat(feats), torch.cat(labels), meta


class Head(nn.Module):
    def __init__(self, in_dim, hidden=0, num_attrs=NUM_ATTRS, dropout=0.1):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, num_attrs))
        else:
            self.net = nn.Linear(in_dim, num_attrs)

    def forward(self, x):
        return self.net(x)


def masked_bce(logits, labels, pos_weight, valid):
    """labels: -1/0/1；valid=(labels!=-1)，无效位置不计损失。"""
    pw = torch.as_tensor(pos_weight, device=logits.device).unsqueeze(0)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float(), pos_weight=pw, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, feats, labels, device, bs=8192):
    """-> (mAP, per-attr AP dict, 平均准确率@0.5, per-attr acc dict)"""
    model.eval()
    outs = []
    for i in range(0, len(feats), bs):
        x = feats[i:i + bs].to(device, non_blocking=True).float()
        outs.append(model(x).float().cpu())
    scores = torch.cat(outs).numpy()
    lab = labels.numpy()
    map_, aps = compute_map(scores, lab)  # AP 只看排序，logit 空间等价
    # 准确率必须在 sigmoid 概率空间取 0.5 阈值（logit 空间的 0.5 是错的）
    mean_acc, accs = per_attribute_accuracy(1 / (1 + np.exp(-scores.astype(np.float64))), lab)
    return map_, aps, mean_acc, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", nargs="+", default=["cache/train.pt"],
                    help="可给多个分片缓存（如多卡并行提取的 train_s*.pt）")
    ap.add_argument("--val-cache", nargs="+", default=["cache/val.pt"])
    ap.add_argument("--hidden", type=int, default=0, help="0=纯线性头；>0 加一层 MLP")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--run-name", default="exp1")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr, tr_meta = load_caches(args.train_cache)
    Xva, Yva, va_meta = load_caches(args.val_cache)
    assert tr_meta["use_deepstack"] == va_meta["use_deepstack"], \
        "train/val 缓存的 deepstack 配置不一致"
    assert Ytr.shape[1] == NUM_ATTRS, f"标签维度 {Ytr.shape[1]} != {NUM_ATTRS}"

    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  "
          f"deepstack={tr_meta['use_deepstack']}")

    pos_weight = valid_pos_weight(Ytr.numpy())
    model = Head(Xtr.shape[1], args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"head params: {n_params:,}")

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.1)

    out_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_map, best_epoch, best_acc, best_accs = 0.0, -1, 0.0, {}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            valid = (y != -1)
            loss = masked_bce(model(x), y, pos_weight, valid.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        map_, _, mean_acc, accs = evaluate(model, Xva, Yva, device)
        tag = "*" if map_ > best_map else " "
        if map_ > best_map:
            best_map, best_epoch = map_, epoch
            best_acc, best_accs = mean_acc, accs
            torch.save({
                "state_dict": model.state_dict(),
                "in_dim": Xtr.shape[1],
                "hidden": args.hidden,
                "use_deepstack": tr_meta["use_deepstack"],
                "attrs": ATTRS,
                "size": tr_meta.get("size"),
                "val_map": map_,
                "val_mean_acc": mean_acc,
                "epoch": epoch,
            }, os.path.join(out_dir, "best.pt"))
        with open(os.path.join(out_dir, "history.csv"), "a") as f:
            f.write(f"{epoch},{tot/len(Xtr):.6f},{map_:.6f},{mean_acc:.6f}\n")
        print(f"{tag} ep{epoch:3d}  loss {tot/len(Xtr):.4f}  val mAP {map_:.4f}  "
              f"val Acc {mean_acc:.4f}  best mAP {best_map:.4f}@{best_epoch}  "
              f"{(time.time()-t0)/60:.1f}min", flush=True)

    # best epoch 的每类准确率单独存档，方便直接查看
    with open(os.path.join(out_dir, "per_class_acc.csv"), "w") as f:
        f.write("attr,acc@0.5\n")
        for a in ATTRS:
            if a in best_accs:
                f.write(f"{a},{best_accs[a]:.6f}\n")
        f.write(f"__mean__,{best_acc:.6f}\n")

    print(f"\nDONE  best mAP {best_map:.4f} @ep{best_epoch}  "
          f"mean Acc {best_acc:.4f}  -> {out_dir}/best.pt")
    print(f"每类准确率: {out_dir}/per_class_acc.csv")


if __name__ == "__main__":
    main()
