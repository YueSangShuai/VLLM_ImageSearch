#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-Reranker-0.6B 推理：一个 query × 一批文档 → P(yes)，改顶部输入直接跑。

口径：模型自带 reranker chat template 按 role=query/document 取值（传 user/system
会静默丢内容），末位 yes/no 两 token logit softmax = P(yes) ∈ [0,1]，阈值 0.5。
输出按分数降序：分数<TAB>文档。
运行（llama 环境，CPU bf16 约 2s/对；有 GPU 自动用）：
  /home/yuesang/miniconda3/envs/llama/bin/python qwen3_rerank.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data/yuesang/LLM/FG-CLIP/models/.ms_cache/Qwen/Qwen3-Reranker-0.6B"

INSTRUCTION = (
    "Given a pedestrian rule, retrieve person descriptions that satisfy "
    "the rule. A description satisfies the rule if it contains all "
    "attributes required by the rule (extra attributes are allowed)."
)

# ==================== 在这里改输入 ====================
QUERY = "一名穿红色上衣的长头发女性"
DOCS = [
    "一名穿黑色上衣的长头发女性",
    "一名穿红色上衣的短头发女性",
    "一名女性有着长头发，穿着黑色短袖"
]
# ======================================================


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"                     # 末位对齐，批量取 [:, -1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()

    yes_id = tok("yes", add_special_tokens=False).input_ids[0]
    no_id = tok("no", add_special_tokens=False).input_ids[0]

    texts = []
    for doc in DOCS:
        # 注意：role 必须是 query/document，模板按名取值；system=任务指令
        msgs = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "query", "content": QUERY},
            {"role": "document", "content": doc},
        ]
        t = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
        assert QUERY in t and doc in t, "模板未注入内容（模板被替换？）"
        texts.append(t)

    with torch.no_grad():
        e = tok(texts, padding=True, truncation=True, max_length=1024,
                return_tensors="pt").to(device)
        logits = model(**e).logits[:, -1, :].float()
        p = torch.softmax(logits[:, [no_id, yes_id]], -1)[:, 1].cpu()

    for s, doc in sorted(zip(p.tolist(), DOCS), key=lambda x: -x[0]):
        print(f"{s:.4f}\t{doc}")


if __name__ == "__main__":
    main()
