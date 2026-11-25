#!/usr/bin/env python3
# 宿主 GPU bf16 基线：OpenSparX-gecko-guard-1B-v1（InternVL3-1B 微调）
# 官方 model.chat() 接口，对 A/B 管线同源行人图跑两种 prompt
# （desc=线上描述口径 / guard=README 强约束格式），产物 jsonl 作板端 w8a8 质量对照基线。
import argparse
import json
import os

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

PROMPT_DESC = ("一句中文详细描述行人外观：性别、衣着颜色款式、携带物，"
               "只写看得见的，不需要描述看不见的，不用解释。")
PROMPT_GUARD = ("你现在是一个优秀的智能安防助手,现在开启了哨兵模式,"
                "请分析图片中人员的异常行为与穿着特点,并给出风险等级。"
                "请严格输出“行为描述。穿着描述。<|TAG|>风险等级”。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/gecko-guard-1b")
    ap.add_argument("--images",
                    default="../../web_vllm_rknn/scripts/eval/w8a8_ab/images")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="baseline_gecko_bf16.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        use_fast=False)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True,
                                      low_cpu_mem_usage=False).cuda().eval()

    tfm = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    imgs = sorted(os.listdir(args.images))[: args.n]
    with open(args.out, "w", encoding="utf-8") as fw:
        for name in imgs:
            img = Image.open(os.path.join(args.images, name))
            pv = tfm(img).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            for tag, prompt in (("desc", PROMPT_DESC), ("guard", PROMPT_GUARD)):
                out = model.chat(tokenizer=tok, pixel_values=pv, question=prompt,
                                 generation_config={"max_new_tokens": 128,
                                                    "do_sample": False},
                                 verbose=False)
                rec = {"image": name, "tag": tag, "text": str(out).strip()}
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{name}][{tag}] {rec['text'][:110]}")


if __name__ == "__main__":
    main()
