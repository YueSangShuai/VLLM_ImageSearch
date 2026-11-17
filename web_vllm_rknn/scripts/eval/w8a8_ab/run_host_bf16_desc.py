# -*- coding: utf-8 -*-
"""W8A8 A/B 之 host bf16 基线（线上描述路径）：与 run_web_rk.sh 生产参数一致。

生产参数：描述 prompt（VLM_DESCRIPTION_PROMPT 同款）、repetition_penalty=1.2、
max_new_tokens=120（VLM_MAX_NEW_TOKENS 默认）、greedy。预处理与 run_host_bf16.py 同
（expand2square(127)+resize448，对齐板端 attr_head 启用时的 VlmDescriber 行为）。
输出 desc_bf16.json（items: file/raw/sec）。

用法：CUDA_VISIBLE_DEVICES=0 conda run -n llama python run_host_bf16_desc.py
"""
import argparse, json, os, sys, time

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/yuesang/LLM/FG-CLIP/qwen_classifer")
from data import expand2square

# run_web_rk.sh VLM_DESCRIPTION_PROMPT 默认值，逐字一致
DESC_PROMPT = ("一句中文详细描述行人外观：性别、衣着颜色款式、携带物，"
               "只写看得见的，不需要描述看不见的，不用解释。")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/yuesang/LLM/Qwen3-VL-2B-Instruct")
ap.add_argument("--set", default=HERE)
ap.add_argument("--size", type=int, default=448)
ap.add_argument("--max-new-tokens", type=int, default=120)   # = VLM_MAX_NEW_TOKENS 默认
ap.add_argument("--penalty", type=float, default=1.2)        # = VLM_REPETITION_PENALTY 默认
args = ap.parse_args()

gt = json.load(open(os.path.join(args.set, "gt.json")))
items = gt["items"]

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
processor = AutoProcessor.from_pretrained(args.model)
print("model loaded", flush=True)

results = []
for k, it in enumerate(items):
    path = os.path.join(args.set, "images", it["file"])
    img = Image.open(path).convert("RGB")
    img = expand2square(img)
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BILINEAR)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": DESC_PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=False).to(model.device, torch.bfloat16)

    t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False, repetition_penalty=args.penalty)
    raw = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    results.append({"file": it["file"], "raw": raw, "sec": round(time.time() - t1, 2)})
    print(f"[{k+1}/{len(items)}] {it['file']} len={len(raw)} {time.time()-t1:.1f}s", flush=True)

with open(os.path.join(args.set, "desc_bf16.json"), "w") as f:
    json.dump({"model": args.model, "penalty": args.penalty,
               "max_new_tokens": args.max_new_tokens, "prompt": DESC_PROMPT,
               "items": results}, f, ensure_ascii=False, indent=1)
print("DONE -> desc_bf16.json")
