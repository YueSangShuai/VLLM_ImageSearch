#!/usr/bin/env python3
"""生成 rerank_bench2 --prompts 模式输入：官方模板渲染的完整 ChatML 文本，
每行一条，换行符转义为字面 \\n（与 C++ unescape 对应）。"""
from transformers import AutoTokenizer

MODEL = "/data/yuesang/LLM/FG-CLIP/models/.ms_cache/Qwen/Qwen3-Reranker-0.6B"
INSTR = ("Given a pedestrian rule, retrieve person descriptions that satisfy "
         "the rule. A description satisfies the rule if it contains all "
         "attributes required by the rule (extra attributes are allowed).")
R = "一名穿黑色上衣的男人"
DOCS = [
    "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包",
    "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包",
    "一名男性，穿着黑色上衣",
    "该人员穿黑色短袖",
    "一名男性，穿着蓝色上衣",
    "一名穿白色连衣裙的女性",
]

tok = AutoTokenizer.from_pretrained(MODEL)
lines = []
for d in DOCS:
    msgs = [{"role": "system", "content": INSTR},
            {"role": "query", "content": R},
            {"role": "document", "content": d}]
    t = tok.apply_chat_template(msgs, tokenize=False,
                                add_generation_prompt=True)
    lines.append(t.replace("\n", "\\n"))
open("/data/yuesang/LLM/FG-CLIP/structbert_npu/rkllm/rerank_prompts.txt", "w")\
    .write("\n".join(lines) + "\n")
print("lines:", len(lines))
