#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BGE(bge-small-zh-v1.5) 作为级联初筛的可行性：同 6 案，看漏检与候选通过率。
口径：非对称检索——规则侧加官方 query 前缀，描述侧裸文本；last-token + L2。
运行：/home/yuesang/miniconda3/envs/llama/bin/python bge_prefilter_check.py
"""
import numpy as np

MODEL = "/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5"
QPRE = "为这个句子生成表示以用于检索相关文章："
R = "一名穿黑色上衣的男人"
PAIRS = [
    (R, "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包", "低"),
    (R, "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包", "高"),
    (R, "一名男性，穿着黑色上衣", "高"),
    (R, "该人员穿黑色短袖", "高"),
    (R, "一名男性，穿着蓝色上衣", "低"),
    (R, "一名穿白色连衣裙的女性", "低"),
]


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()

    def enc(texts):
        e = tok(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
        with torch.no_grad():
            h = m(**e).last_hidden_state
        idx = e["attention_mask"].sum(1) - 1
        return F.normalize(h[torch.arange(len(idx)), idx].float(), dim=-1).numpy()

    Q = enc([QPRE + q for q, _, _ in PAIRS])
    D = enc([d for _, d, _ in PAIRS])
    s = (Q * D).sum(1)
    for (q, d, exp), v in zip(PAIRS, s):
        print(f"  cos={v:.4f}  期望{exp}  {d}")
    pos = [v for v, (_, _, e) in zip(s, PAIRS) if e == "高"]
    neg = [v for v, (_, _, e) in zip(s, PAIRS) if e == "低"]
    lo = min(pos)                       # 不漏正样本的最高阈值
    print(f"\n正样本 min={lo:.4f}，负样本 max={max(neg):.4f}")
    print(f"初筛阈值若取 {lo - 0.005:.3f}（保证 6/6 不漏）→ 负样本放行 "
          f"{sum(v >= lo - 0.005 for v in neg)}/{len(neg)}")


if __name__ == "__main__":
    main()
