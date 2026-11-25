#!/usr/bin/env python3
# 宿主 GPU bf16 基线：LiquidAI/LFM2.5-VL-1.6B（SigLIP2 NaFlex + LFM2.5-1.2B）
# 与 gecko/host_baseline.py 完全同源口径：同 5 张 A/B 行人图、desc/guard 双 prompt、
# greedy、max_new_tokens=128，产物 jsonl 供两模型并排对比。
# 用法（llama env，transformers 5.8 原生支持 lfm2_vl）：
#   CUDA_VISIBLE_DEVICES=0 python host_baseline_lfm.py
import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT_DESC = ("一句中文详细描述行人外观：性别、衣着颜色款式、携带物，"
               "只写看得见的，不需要描述看不见的，不用解释。")
PROMPT_GUARD = ("你现在是一个优秀的智能安防助手,现在开启了哨兵模式,"
                "请分析图片中人员的异常行为与穿着特点,并给出风险等级。"
                "请严格输出“行为描述。穿着描述。<|TAG|>风险等级”。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/LFM2.5-VL-1.6B")
    ap.add_argument("--images",
                    default="../../web_vllm_rknn/scripts/eval/w8a8_ab/images")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="baseline_lfm_bf16.jsonl")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).cuda().eval()

    imgs = sorted(os.listdir(args.images))[: args.n]
    t0 = time.time()
    n_tok = 0
    with open(args.out, "w", encoding="utf-8") as fw:
        for name in imgs:
            img = Image.open(os.path.join(args.images, name)).convert("RGB")
            for tag, prompt in (("desc", PROMPT_DESC), ("guard", PROMPT_GUARD)):
                conv = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}]
                inputs = proc.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=True,
                    return_tensors="pt", return_dict=True).to(model.device)
                n_in = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=128,
                                         do_sample=False)
                gen = out[0][n_in:]
                n_tok += int(gen.shape[0])
                text = proc.decode(gen, skip_special_tokens=True).strip()
                rec = {"image": name, "tag": tag, "text": text}
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{name}][{tag}] {text[:110]}")
    dt = time.time() - t0
    print(f"total {dt:.1f}s for {n_tok} tokens -> {dt / max(n_tok,1) * 1000:.1f} ms/tok (含加载外全部)")


if __name__ == "__main__":
    main()
