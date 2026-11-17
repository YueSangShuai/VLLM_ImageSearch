#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructBERT Siamese 专测：红色上衣+黑背包 描述 vs 「穿黑色上衣的男人」规则。

模型：damo/nlp_structbert_sentence-similarity_chinese-base（ModelScope）
两种打分口径：
  1) 孪生双塔 (siamese)：A、B 各自单独编码，CLS pooling + L2 归一，cos 相似度；
  2) pair 分类 (checkpoint 原口径)：[CLS]A[SEP]B[SEP] 过分类头，P(相似)。
另按线上口径：描述按「，」分句（剔噪声），规则 query 对分句取 max。
参考对照：A×A、B×B（正样本对上界）。
运行：/home/yuesang/miniconda3/envs/nomic/bin/python test_structbert_red_vs_black.py
"""
import os

import numpy as np
import torch

MODEL_DIR = ("/data/yuesang/LLM/FG-CLIP/models/.ms_cache/damo/"
             "nlp_structbert_sentence-similarity_chinese-base")

A = "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包"   # 描述（doc 侧）
B = "一名穿黑色上衣的男人"                                  # 规则（query 侧）

NOISE = ["无法", "难以", "看不清", "看不出", "看不清楚", "视角",
         "模糊", "氛围", "感觉", "图像显示", "画面显示", "画质", "分辨率"]
CLAUSES = [c.strip() for c in A.split("，")
           if c.strip() and not any(b in c for b in NOISE)]


def main():
    from modelscope.models import Model
    from transformers import AutoTokenizer

    model = Model.from_pretrained(MODEL_DIR)
    model.eval()
    print("模型类:", type(model).__name__)
    print("子模块:", [(n, type(c).__name__) for n, c in model.named_children()])
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"\n描述 A : {A}\n  分句: {CLAUSES}\n规则 B : {B}\n")

    # ---------- 1) 孪生双塔：单独编码 → CLS cos ----------
    @torch.no_grad()
    def embed(texts):
        enc = tok(list(texts), return_tensors="pt", padding=True,
                  truncation=True, max_length=128)
        out = model.encoder(**enc)              # 孪生单塔
        vec = getattr(out, "pooled_output", None)
        if vec is None:
            hs = out[1] if isinstance(out, tuple) and len(out) > 1 else out[0]
            vec = hs[:, 0]                                  # CLS pooling
        return torch.nn.functional.normalize(vec, dim=-1).numpy()

    eA, eB = embed([A])[0], embed([B])[0]
    eC = embed(CLAUSES)
    print("[孪生 cos]  B×A (整句)  = %.4f" % float(eB @ eA))
    sims = [float(eB @ c) for c in eC]
    print("[孪生 cos]  B×分句max   = %.4f  (线上口径)" % max(sims))
    for c, s in zip(CLAUSES, sims):
        print(f"    {s:.4f}  {c}")
    print("[孪生 cos]  对照 A×A    = %.4f   B×B = %.4f" %
          (float(eA @ embed([A])[0]), float(eB @ embed([B])[0])))

    # ---------- 2) pair 分类：[CLS]A[SEP]B[SEP] → P(相似) ----------
    @torch.no_grad()
    def pair_p_sim(text1, text2):
        enc = tok(list(text1), list(text2), return_tensors="pt",
                  padding=True, truncation=True, max_length=128)
        out = model.forward(**enc, return_dict=True)
        logits = out["logits"] if isinstance(out, dict) \
            else getattr(out, "logits", None)
        p = torch.softmax(logits, dim=-1)
        return p[:, 1].numpy()                  # label 1 = 相似

    pairs = [(B, A), (B, CLAUSES[0]), (B, CLAUSES[1]), (B, CLAUSES[2]),
             (A, A), (B, B), (B, "一名男性，穿着黑色上衣"),
             (B, "一名男性，背着黑色背包")]
    src = [p[0] for p in pairs]
    tgt = [p[1] for p in pairs]
    ps = pair_p_sim(src, tgt)
    print("\n[pair P(相似)]")
    for (s_, t_), v in zip(pairs, ps):
        print(f"    {v:.4f}  {s_[:12]}…  vs  {t_}")


if __name__ == "__main__":
    main()
