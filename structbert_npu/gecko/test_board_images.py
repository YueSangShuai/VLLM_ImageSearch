#!/usr/bin/env python3
# gecko bf16 本地测试：对板上拉回的真实告警 ROI 图跑 desc/guard 双 prompt，
# 并排打印线上模型（alerts_history 里的 description）供肉眼对照。
# 用法（vllm env GPU）：CUDA_VISIBLE_DEVICES=1 python test_board_images.py
import glob
import json
import os

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/gecko-guard-1b"
IMAGES = "/data/yuesang/LLM/FG-CLIP/structbert_npu/gecko/board_images"
HIST = os.path.join(IMAGES, "alerts_history")

PROMPT_DESC = ("一句中文详细描述行人外观：性别、衣着颜色款式、携带物，"
               "只写看得见的，不需要描述看不见的，不用解释。")
PROMPT_GUARD = ("你现在是一个优秀的智能安防助手,现在开启了哨兵模式,"
                "请分析图片中人员的异常行为与穿着特点,并给出风险等级。"
                "请严格输出“行为描述。穿着描述。<|TAG|>风险等级”。")


def load_live_descriptions():
    """alerts_history 目录/文件 → {图片时间戳数字: 线上描述}（文件名 alert_<ts>_*.jpg 对齐）。"""
    out = {}
    files = []
    if os.path.isdir(HIST):
        files = [os.path.join(HIST, f) for f in sorted(os.listdir(HIST))]
    elif os.path.exists(HIST):
        files = [HIST]
    for fp in files:
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else data.get("alerts", [])
        for it in items:
            desc = it.get("description") or it.get("desc") or ""
            for key in ("image", "image_path", "original", "roi", "image_roi"):
                v = it.get(key)
                if isinstance(v, str):
                    base = os.path.basename(v)
                    num = base.split("_")[1] if base.startswith("alert_") and "_" in base[7:] else ""
                    if num:
                        out[num] = desc
                        out[base] = desc
    return out


def main():
    live = load_live_descriptions()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True,
                                      low_cpu_mem_usage=False).cuda().eval()
    tfm = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    imgs = sorted(set(glob.glob(os.path.join(IMAGES, "*.jpg"))) |
                  set(glob.glob(os.path.join(IMAGES, "*__*"))))
    results = []
    for path in imgs:
        name = os.path.basename(path)
        pv = tfm(Image.open(path)).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        rec = {"image": name}
        for tag, prompt in (("desc", PROMPT_DESC), ("guard", PROMPT_GUARD)):
            out = model.chat(tokenizer=tok, pixel_values=pv, question=prompt,
                             generation_config={"max_new_tokens": 128, "do_sample": False},
                             verbose=False)
            rec[tag] = str(out).strip()
        num = name.split("_")[1] if name.startswith("alert_") else name
        rec["live"] = live.get(num) or live.get(name) or ""
        results.append(rec)
        print(f"== {name}")
        print(f"  [线上] {rec['live'][:100]}")
        print(f"  [desc ] {rec['desc'][:120]}")
        print(f"  [guard] {rec['guard'][:120]}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "baseline_gecko_board.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved:", out)


if __name__ == "__main__":
    main()
