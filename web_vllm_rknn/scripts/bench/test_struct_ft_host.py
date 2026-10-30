#!/usr/bin/env python
# 宿主端直测 qwen3vl-ft-merged (bf16) 结构化精度：
# 与 scripts/bench/bench_struct_ab_board.sh 同一批告警 ROI 图、同一结构化属性表，
# greedy 解码。用于对照板上 W8A8 输出，定位「结构化突然不准」是模型还是量化/平台问题。
# 用法: CUDA_VISIBLE_DEVICES=0 python test_struct_ft_host.py [model_dir]
import sys, time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/models/qwen3vl-ft-merged"
D = "/data/yuesang/LLM/FG-CLIP/web_vllm_cpp/data/alerts_images/2026-10-19/一楼电梯厅"
IMGS = {
    "戴眼镜的男生": f"{D}/戴眼镜的男生/roi/alert_1792370895049_895048.jpg",
    "带帽子的人":   f"{D}/带帽子的人/roi/alert_1792370129563_129548.jpg",
    "穿裙子的女生": f"{D}/穿裙子的女生/roi/alert_1792370697027_697026.jpg",
    "黑衣服男生":   f"{D}/黑衣服男生/roi/alert_1792368184578_184578.jpg",
}
SP = ("请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），"
      "只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。\n"
      "预定义属性列表：Female,AgeOver60,Age18-60,AgeLess18,Front,Side,Back,Hat,Glasses,"
      "HandBag,ShoulderBag,Backpack,HoldObjectsInFront,ShortSleeve,LongSleeve,UpperStride,"
      "UpperLogo,UpperPlaid,UpperSplice,LowerStripe,LowerPattern,LongCoat,Trousers,Shorts,"
      "Skirt&Dress,boots,ublack,ugray,ublue,ugreen,uwhite,upurple,ured,ubrown,uyellow,upink,"
      "uorange,ubeige,ustriped_color,umulticolor,lwhite,lpink,lred,lgreen,lyellow,lpurple,"
      "lbrown,lblack,lorange,lblue,lgray,lbeige,lstriped_color,lmulticolor")

ATTRS = set(SP.split("预定义属性列表：")[1].split(","))
print(f"model: {MODEL}")
t0 = time.time()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0")
processor = AutoProcessor.from_pretrained(MODEL)
print(f"loaded in {time.time()-t0:.1f}s")

for rule, path in IMGS.items():
    messages = [{"role": "user", "content": [
        {"type": "image", "image": path},
        {"type": "text", "text": SP},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=False).to(model.device, torch.bfloat16)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    txt = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    valid = [a for a in txt.split(",") if a.strip() in ATTRS]
    invalid = [a for a in txt.split(",") if a.strip() and a.strip() not in ATTRS]
    print(f"\n[{rule}] {time.time()-t1:.1f}s")
    print(f"  raw    : {txt}")
    print(f"  合法 {len(valid)} 项: {','.join(a.strip() for a in valid)}")
    if invalid:
        print(f"  非法token {len(invalid)} 项: {invalid}")
print("\nDONE")
