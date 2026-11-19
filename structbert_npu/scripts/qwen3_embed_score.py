#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-Embedding-0.6B：规则(query, 带指令) × 描述(document) → cosine 判定。

口径（官方用法）：last-token pooling + L2 归一化；query 侧带指令
「Instruct: {task}\\nQuery: {query}」，document 侧裸文本。
对照组：qwen3_reranker_score.py（6/6）、pair_score.py（bge，4/6）、
siglip_text_score.py（无判别力）。
运行（llama 环境，CPU bf16，6 条各编码一次约几秒）：
  /home/yuesang/miniconda3/envs/llama/bin/python qwen3_embed_score.py
"""
import numpy as np

MODEL = "/data/yuesang/LLM/FG-CLIP/models/.ms_cache/Qwen/Qwen3-Embedding-0.6B"

TASK = ("Given a pedestrian rule, retrieve person descriptions that satisfy "
        "the rule. A description satisfies the rule if it contains all "
        "attributes required by the rule (extra attributes are allowed).")

# ==================== 在这里改输入 ====================
R = "一名穿黑色上衣的男人"
PAIRS = [
    (R, "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包", "低"),   # 红≠黑
    (R, "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包", "高"),   # 关键稀释案例
    (R, "一名男性，穿着黑色上衣", "高"),
    (R, "该人员穿黑色短袖", "高"),                                     # 词汇缺口
    (R, "一名男性，穿着蓝色上衣", "低"),
    (R, "一名穿白色连衣裙的女性", "低"),
]
# ======================================================


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).eval()

    def embed(texts):
        e = tok(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
        with torch.no_grad():
            h = model(**e).last_hidden_state
        # last-token pooling（与 1_Pooling/lasttoken 配置一致）
        idx = e["attention_mask"].sum(1) - 1
        v = h[torch.arange(len(idx)), idx]
        return F.normalize(v.float(), dim=-1).numpy()

    qs = [f"Instruct: {TASK}\nQuery: {q}" for q, _, _ in PAIRS]
    ds = [d for _, d, _ in PAIRS]
    Q = embed(qs)
    D = embed(ds)
    sims = (Q * D).sum(1)
    # 参照系：规则自我相似度 & 描述间相似度，看锥体问题比 SigLIP 好多少
    RR = embed([R])
    allD = embed([d for _, d, _ in PAIRS])
    print(f"cos(query,doc):")
    for (q, d, exp), s in zip(PAIRS, sims):
        print(f"  {s:+.4f}  期望{exp}  {q}\n            ×  {d}")
    print(f"\n自检: cos(规则,规则)={float((RR*RR).sum(1)[0]):.4f}  "
          f"desc 两两 cos 范围=[{float((allD@allD.T)[np.triu_indices(6,1)].min()):.3f},"
          f"{float((allD@allD.T)[np.triu_indices(6,1)].max()):.3f}]")
    pos = [s for s, (_, _, e) in zip(sims, PAIRS) if e == "高"]
    neg = [s for s, (_, _, e) in zip(sims, PAIRS) if e == "低"]
    print(f"正样本范围 [{min(pos):.4f}, {max(pos):.4f}]  "
          f"负样本范围 [{min(neg):.4f}, {max(neg):.4f}]  "
          f"{'可分，阈值≈' + format((min(pos)+max(neg))/2, '.3f') if min(pos) > max(neg) else '区间重叠，不可分'}")


if __name__ == "__main__":
    main()
