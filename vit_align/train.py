# -*- coding: utf-8 -*-
"""训练 ProjMLP（唯一可训练件）：直接吃两路离线缓存矩阵，主干零参与。

例：
  python train.py                                  # 用 configs/default.yaml
  python train.py --name smoke --epochs 3          # 冒烟
产物：runs/<name>/best.pt（含 config 快照 + 指标），log.jsonl。
"""
import argparse
import json
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from common import abspath, expand_cache_paths, load_config, read_chunk, set_seed
from data.dataset import PairDataset, build_index
from loss import info_nce
from metrics import recall_ap_chunked
from model import ProjMLP


def stack_cache(patterns, want_dim=None):
    """按文件名序拼接分块 -> feats [N,D] float32, ids [N]。"""
    feats, ids = [], []
    for p in expand_cache_paths([patterns] if isinstance(patterns, str) else patterns):
        d = read_chunk(p)
        feats.append(d["feats"])
        ids.append(d["ids"])
    x = torch.cat(feats)
    if want_dim and x.shape[1] != want_dim:
        raise SystemExit(f"缓存维度 {x.shape[1]} != 期望 {want_dim}（{p}）")
    return x, torch.cat(ids)


def evaluate(model, ev, device, ks=(1, 5, 10)):
    """test 集 i2t/t2i 指标（无前缀 caption 空间）。"""
    with torch.no_grad():
        img = model.encode_image(ev["img"].to(device))
        txt = model.encode_text(ev["txt"].to(device))
        img_ids = ev["img_ids"].to(device)
        cap_ids = ev["cap_ids"].to(device)
        i2t = recall_ap_chunked(img @ txt.t(), img_ids, cap_ids, ks)
        t2i = recall_ap_chunked(txt @ img.t(), cap_ids, img_ids, ks)
    return i2t, t2i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--multi-positive", dest="multi_positive", default=None,
                    choices=["true", "false"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    epochs = args.epochs if args.epochs is not None else int(tcfg["epochs"])
    bs = args.batch_size or int(tcfg["batch_size"])
    multi_pos = (str(tcfg.get("multi_positive", True)).lower() == "true")
    if args.multi_positive is not None:
        multi_pos = args.multi_positive == "true"
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_dir = abspath(cfg["data"]["json_dir"])
    train_json = [os.path.join(json_dir, "train.json")]
    test_json = os.path.join(json_dir, "test.json")

    # ---- 训练缓存：图像 [Ni,2048]，caption [Nc,512] ----
    entries, captions, cap_ids, pairs = build_index(train_json)
    img_x, img_ids = stack_cache(tcfg["img_cache"], cfg["vision"]["feat_dim"])
    txt_x, _ = stack_cache(tcfg["txt_cache"], cfg["text"]["feat_dim"])
    if img_x.shape[0] != len(entries):
        raise SystemExit(f"图像缓存 {img_x.shape[0]} != json 条目 {len(entries)}，请先重新提取")
    if txt_x.shape[0] != len(captions):
        raise SystemExit(f"文本缓存 {txt_x.shape[0]} != caption 数 {len(captions)}")
    ds = PairDataset(img_x.to(device), txt_x.to(device), pairs)
    print(f"train pairs: {len(ds)}  imgs {img_x.shape}  caps {txt_x.shape}")

    # ---- 评估缓存（test，doc 无前缀空间） ----
    t_entries, t_caps, t_cap_ids, _ = build_index([test_json])
    t_img, t_img_ids = stack_cache(cfg["eval"]["img_cache"], cfg["vision"]["feat_dim"])
    t_txt, _ = stack_cache(cfg["eval"]["txt_cache"], cfg["text"]["feat_dim"])
    if t_img.shape[0] != len(t_entries):
        raise SystemExit(f"test 图像缓存 {t_img.shape[0]} != test.json 条目 {len(t_entries)}")
    if t_txt.shape[0] != len(t_caps):
        raise SystemExit(f"test 文本缓存 {t_txt.shape[0]} != test caption 数 {len(t_caps)}")
    ev = {"img": t_img, "txt": t_txt, "img_ids": t_img_ids, "cap_ids": t_cap_ids}
    print(f"test imgs {tuple(t_img.shape)}  caps {tuple(t_txt.shape)}")

    h = cfg["head"]
    model = ProjMLP(int(cfg["vision"]["feat_dim"]), int(h["hidden"]), int(h["out_dim"]),
                    float(h["init_logit_scale"]), bool(h.get("text_proj", False)),
                    int(cfg["text"]["feat_dim"])).to(device)
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (nodecay if p.ndim <= 1 or "logit_scale" in n else decay).append(p)
    opt = AdamW([{"params": decay, "weight_decay": float(tcfg["weight_decay"])},
                 {"params": nodecay, "weight_decay": 0.0}], lr=float(tcfg["lr"]))
    n_iter = max(1, len(ds) // bs)
    total_steps = epochs * n_iter
    warmup = int(tcfg["warmup_steps"])

    def lr_at(s):
        if s < warmup:
            return (s + 1) / max(1, warmup)
        import math
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))

    sched = LambdaLR(opt, lr_at)

    runs_dir = abspath(os.path.join(tcfg["runs_dir"], args.name))
    os.makedirs(runs_dir, exist_ok=True)
    logf = open(os.path.join(runs_dir, "log.jsonl"), "a", encoding="utf-8")
    best_r1 = 0.0
    eval_every = int(tcfg["eval_every"])

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ds), device=device)
        tot = acc_sum = 0.0
        seen = 0
        t0 = time.time()
        for i in range(0, len(ds) - bs + 1, bs):
            idx = perm[i:i + bs]
            x, t, ids = ds.batch(idx)
            img_e = model.encode_image(x)
            txt_e = model.encode_text(t)
            loss, acc = info_nce(img_e, txt_e, model.logit_scale, ids, multi_pos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            acc_sum += acc.item() * len(idx)
            seen += len(idx)
        n = max(seen, 1)
        rec = {"epoch": epoch, "loss": tot / max(n, 1), "i2t_acc1": acc_sum / max(n, 1),
               "lr": opt.param_groups[0]["lr"], "logit_scale":
                   round(float(model.logit_scale.exp().clamp(max=100.0)), 2),
               "sec": round(time.time() - t0, 1)}

        if epoch % eval_every == 0 or epoch == epochs:
            i2t, t2i = evaluate(model, ev, device)
            rec.update(i2t_i2t=i2t, t2i=t2i)
            print(f"epoch {epoch}: loss {rec['loss']:.4f} acc1 {rec['i2t_acc1']:.3f} "
                  f"| i2t R@1 {i2t['R@1']:.4f} R@5 {i2t['R@5']:.4f} mAP {i2t['mAP']:.4f} "
                  f"| t2i R@1 {t2i['R@1']:.4f} mAP {t2i['mAP']:.4f}")
            if i2t["R@1"] > best_r1:
                best_r1 = i2t["R@1"]
                torch.save({"trainable": model.trainable_state(), "config": cfg,
                            "epoch": epoch, "i2t": i2t, "t2i": t2i,
                            "best_r1": best_r1},
                           os.path.join(runs_dir, "best.pt"))
        else:
            print(f"epoch {epoch}: loss {rec['loss']:.4f} acc1 {rec['i2t_acc1']:.3f} "
                  f"lr {rec['lr']:.2e} ({rec['sec']}s)")
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()

    torch.save({"trainable": model.trainable_state(), "config": cfg,
                "epoch": epochs, "best_r1": best_r1},
               os.path.join(runs_dir, "last.pt"))
    print(f"best i2t R@1 = {best_r1:.4f} -> {runs_dir}/best.pt")


if __name__ == "__main__":
    main()
