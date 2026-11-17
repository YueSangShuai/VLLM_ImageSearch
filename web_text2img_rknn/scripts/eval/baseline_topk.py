#!/usr/bin/env python3
"""x86 torch 基线：对 gallery 目录 + queries.json 算 topk，输出 baseline.json。
板端服务返回结果与之对比（排序一致 + 分数误差）。
用法: /home/yuesang/miniconda3/envs/MARS/bin/python scripts/eval/baseline_topk.py \
        [--gallery DIR] [--queries golden/queries.json] [--out /tmp/baseline.json] [--topk 20]
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "export"))
import numpy as np
import torch
from onnx_export import build_model, load_u8, u8_to_nchw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=os.path.join(ROOT, "output", "gallery"))
    ap.add_argument("--queries", default=os.path.join(ROOT, "golden", "queries.json"))
    ap.add_argument("--out", default="/tmp/baseline.json")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    model, tok, cfg = build_model()
    from onnx_export import ImageTower, TextTower
    img_tower = ImageTower(model.visual_encoder, model.vision_proj)
    txt_tower = TextTower(model.text_encoder.bert, model.text_proj,
                          int(getattr(model.text_encoder.config, "fusion_layer", 6)))

    files = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(args.gallery, "**", "*.jpg"), recursive=True) +
                   glob.glob(os.path.join(args.gallery, "**", "*.png"), recursive=True))
    g_emb = []
    with torch.no_grad():
        for f in files:
            x = torch.from_numpy(u8_to_nchw(load_u8(os.path.join(args.gallery, f)))).float()
            g_emb.append(img_tower(x).numpy()[0])
    g_emb = np.stack(g_emb)

    queries = json.load(open(args.queries))
    out = {}
    with torch.no_grad():
        for q in queries:
            ti = tok(q, padding="max_length", truncation=True, max_length=128,
                     return_tensors="pt")
            ids = ti.input_ids[:, :128]
            attn = (ids != 0).long()
            e = txt_tower(ids.int(), attn.float()).numpy()[0]
            sims = g_emb @ e
            idx = np.argsort(-sims)[:args.topk]
            out[q] = [[files[i], round(float(sims[i]), 4)] for i in idx]
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"baseline: {len(files)} imgs x {len(queries)} queries -> {args.out}")


if __name__ == "__main__":
    main()
