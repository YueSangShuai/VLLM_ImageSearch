# -*- coding: utf-8 -*-
"""W8A8 A/B 之 host bf16 基线：原始 Qwen3-VL-2B-Instruct 跑服务同款结构化路径。

对齐线上参数：greedy(do_sample=False)、repetition_penalty=1.4、max_new_tokens=128、
预处理 expand2square(127.5,floor)+resize448（与板端 imgenc/训练端一致）。
顺带 dump 视觉塔 bf16 mean-pool 特征（供与板端 VLM_ATTR_HEAD_DUMP 的 fp16 pool 对齐），
并用 attr_head.bin 权重+阈值算头属性（bf16 特征版）。

用法：CUDA_VISIBLE_DEVICES=6 conda run -n llama python run_host_bf16.py
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/yuesang/LLM/FG-CLIP/qwen_classifer")
from w8a8_common import parse_attr_text, structured_prompt
from data import expand2square
from export_head import read_bin

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/yuesang/LLM/Qwen3-VL-2B-Instruct")
ap.add_argument("--set", default=HERE)
ap.add_argument("--head-bin",
                default="/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/output_cpp/models/attr_head.bin")
ap.add_argument("--size", type=int, default=448)
ap.add_argument("--max-new-tokens", type=int, default=128)   # = 线上 structured_max_new_tokens
ap.add_argument("--penalty", type=float, default=1.4)        # = 线上 structured_repetition_penalty
args = ap.parse_args()

gt = json.load(open(os.path.join(args.set, "gt.json")))
items = gt["items"]
prompt = structured_prompt()

t0 = time.time()
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
processor = AutoProcessor.from_pretrained(args.model)
print(f"model loaded {time.time()-t0:.1f}s", flush=True)

W, b, thr, names = read_bin(args.head_bin)   # W [54,2048] f4, thr list, names list（54 属性序）
W = torch.tensor(W)
b = torch.tensor(b)
thr = np.asarray(thr)


def head_attrs(pooled_f32):
    probs = torch.sigmoid(W @ torch.tensor(pooled_f32) + b).numpy()
    return [names[j] for j in range(len(names)) if probs[j] >= thr[j]]


results, pools = [], []
MERGE = 2
for k, it in enumerate(items):
    path = os.path.join(args.set, "images", it["file"])
    img = Image.open(path).convert("RGB")
    img = expand2square(img)
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BILINEAR)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=False).to(model.device, torch.bfloat16)

    t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        vis = model.model.visual(hidden_states=inputs["pixel_values"].to(model.model.visual.dtype),
                                 grid_thw=inputs["image_grid_thw"].to(torch.int64))
        counts = (inputs["image_grid_thw"].prod(-1) // MERGE ** 2).tolist()
        pooled = torch.stack([p.mean(0) for p in vis.pooler_output.split(counts)])[0]
        pooled = pooled.float().cpu().numpy()
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False, repetition_penalty=args.penalty)
    raw = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    valid, invalid = parse_attr_text(raw)
    pools.append(pooled)
    results.append({"file": it["file"], "raw": raw, "valid": valid, "invalid": invalid,
                    "head_attrs": head_attrs(pooled), "sec": round(time.time() - t1, 2)})
    print(f"[{k+1}/{len(items)}] {it['file']}  valid={len(valid)} invalid={len(invalid)} "
          f"{time.time()-t1:.1f}s", flush=True)

np.save(os.path.join(args.set, "pool_bf16.npy"), np.stack(pools))
with open(os.path.join(args.set, "host_bf16.json"), "w") as f:
    json.dump({"model": args.model, "penalty": args.penalty,
               "max_new_tokens": args.max_new_tokens, "items": results},
              f, ensure_ascii=False, indent=1)
print("DONE -> host_bf16.json + pool_bf16.npy")
