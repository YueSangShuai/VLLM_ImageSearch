#!/usr/bin/env python3
"""主机侧预渲染 6 案（官方 query/document 模板）+ 分词 → 板端 rerank_bench 输入。"""
from transformers import AutoTokenizer

MODEL = "/data/yuesang/LLM/FG-CLIP/models/.ms_cache/Qwen/Qwen3-Reranker-0.6B"
OUT = "/data/yuesang/LLM/FG-CLIP/structbert_npu/rkllm/rerank_tokens.txt"
INSTR = ("Given a pedestrian rule, retrieve person descriptions that satisfy "
         "the rule. A description satisfies the rule if it contains all "
         "attributes required by the rule (extra attributes are allowed).")
R = "一名穿黑色上衣的男人"
PAIRS = [
    (R, "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包", "低"),
    (R, "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包", "高"),
    (R, "一名男性，穿着黑色上衣", "高"),
    (R, "该人员穿黑色短袖", "高"),
    (R, "一名男性，穿着蓝色上衣", "低"),
    (R, "一名穿白色连衣裙的女性", "低"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
lines = []
for i, (q, d, e) in enumerate(PAIRS):
    msgs = [{"role": "system", "content": INSTR},
            {"role": "query", "content": q},
            {"role": "document", "content": d}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert q in t and d in t
    ids = tok(t, add_special_tokens=False)["input_ids"]
    lines.append(f"{i}{e}\t" + " ".join(map(str, ids)))
open(OUT, "w").write("\n".join(lines) + "\n")
print("lines:", len(lines), "ntok:", [len(l.split()[1:]) for l in lines])
