#!/usr/bin/env python3
# LFM2.5-VL-1.6B bf16 本地测试：与 gecko/test_board_images.py 同口径，
# 对板上拉回的真实告警 ROI 图跑 desc/guard 双 prompt，并排打印线上描述对照。
# 用法（llama env）：CUDA_VISIBLE_DEVICES=0 python test_board_images_lfm.py
import glob
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/LFM2.5-VL-1.6B"
IMAGES = "/data/yuesang/LLM/FG-CLIP/structbert_npu/gecko/board_images"
HIST = os.path.join(IMAGES, "alerts_history")

PROMPT_DESC = ("一句中文详细描述行人外观：性别、衣着颜色款式、携带物，"
               "只写看得见的，不需要描述看不见的，不用解释。")
PROMPT_GUARD = ("你现在是一个优秀的智能安防助手,现在开启了哨兵模式,"
                "请分析图片中人员的异常行为与穿着特点,并给出风险等级。"
                "请严格输出“行为描述。穿着描述。<|TAG|>风险等级”。")


def load_live_descriptions():
    """alerts_history → {文件名/时间戳: 线上描述}，同 gecko 版。"""
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
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda().eval()

    imgs = sorted(set(glob.glob(os.path.join(IMAGES, "*.jpg"))) |
                  set(glob.glob(os.path.join(IMAGES, "*__*"))))
    results = []
    for path in imgs:
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        rec = {"image": name}
        for tag, prompt in (("desc", PROMPT_DESC), ("guard", PROMPT_GUARD)):
            conv = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            inputs = proc.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            text = proc.decode(out[0][inputs["input_ids"].shape[-1]:],
                               skip_special_tokens=True).strip()
            rec[tag] = text
        # 文件名 tag: 日期__规则__alert_<ts>_xxx.jpg 或 alert_<ts>_xxx.jpg
        base = name.split("__")[-1]
        num = base.split("_")[1] if base.startswith("alert_") else name
        rec["live"] = live.get(num) or live.get(name) or ""
        results.append(rec)
        print(f"== {name}")
        print(f"  [线上] {rec['live'][:100]}")
        print(f"  [desc ] {rec['desc'][:120]}")
        print(f"  [guard] {rec['guard'][:120]}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "baseline_lfm_board.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved:", out)


if __name__ == "__main__":
    main()
