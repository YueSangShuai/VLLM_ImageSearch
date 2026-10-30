#!/usr/bin/env python
# 为 reranker 的 .rkllm 量化生成校准数据：真实行人 ROI(缩到448,与板端一致) + reranker 格式 prompt，
# hook language_model 前向拿 inputs_embeds，产出 llm_inputs/*.pkl + llm_inputs.json。
# 用法：<Qwen-env-python> scripts/make_reranker_calib.py [--n 24] [--gpu 6]
import argparse, os, sys, glob, json, pickle, collections

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/yuesang/LLM/FG-CLIP/models/Qwen3-VL-Reranker-2B")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "export_reranker", "data"))
ap.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "..", "calib"))
ap.add_argument("--instruction", default="Given a pedestrian description, retrieve surveillance images of pedestrians matching it.")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--gpu", default="6")
args = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

sys.path.insert(0, os.path.join(args.model, "scripts"))
import torch
from PIL import Image
from qwen3_vl_reranker import Qwen3VLReranker

# 收集 (ROI图, 规则文本) 样本，正负都有
rules = collections.defaultdict(lambda: {"pos": set(), "imgs": []})
for f in sorted(glob.glob(os.path.join(args.calib, "alerts_history", "*.json"))):
    for a in json.load(open(f)):
        p = a.get("positive", "")
        if p: rules[a.get("rule_name", "?")]["pos"].add(p)
for f in sorted(glob.glob(os.path.join(args.calib, "alerts_images", "*", "*", "*", "roi", "*.jpg"))):
    rules[f.split(os.sep)[-3]]["imgs"].append(f)

samples = []
names = sorted(k for k, v in rules.items() if v["pos"] and v["imgs"])
per = max(1, args.n // max(1, len(names)))
for i, name in enumerate(names):
    texts = sorted(rules[name]["pos"])
    for img in rules[name]["imgs"][:per]:
        t = texts[i % len(texts)]
        samples.append((img, t))
samples = samples[: args.n]
print(f"calib samples: {len(samples)}")

wrapper = Qwen3VLReranker(model_name_or_path=args.model, torch_dtype=torch.bfloat16)
model = wrapper.model  # lm.model（含 visual + language_model）
processor = wrapper.processor
# 校准图统一 448x448：与板端视觉塔固定分辨率、C++ 送 embedding 的 token 数一致
out_dir = os.path.abspath(args.out)
os.makedirs(os.path.join(out_dir, "llm_inputs"), exist_ok=True)

def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(o) for o in obj)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj

class StopForward(Exception): pass
captured = []
def pre_hook(module, a, kwargs):
    captured.append(kwargs)
    raise StopForward
handle = model.language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)

info = []
for idx, (imgp, text) in enumerate(samples):
    img = Image.open(imgp).convert("RGB").resize((448, 448), Image.BILINEAR)
    pairs = wrapper.format_mm_instruction(text, None, None, None, img, None,
                                          instruction=args.instruction)
    # 图片对象已在 pairs 里；processor 直接吃 PIL
    rendered = processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[img], padding=True, return_tensors="pt",
                       do_resize=False)
    inputs = inputs.to(model.device)
    try:
        model(**inputs)
    except StopForward:
        pass
    temp = captured[-1]
    embeds = to_cpu(temp)
    with open(os.path.join(out_dir, "llm_inputs", f"sample_{idx}"), "wb") as f:
        pickle.dump(embeds, f)
    info.append({"sample": f"llm_inputs/sample_{idx}",
                 "token_nums": int(embeds["inputs_embeds"].shape[1])})
    print(info[-1], flush=True)
handle.remove()
json.dump(info, open(os.path.join(out_dir, "llm_inputs.json"), "w"), indent=2, ensure_ascii=False)
print("done ->", out_dir)
