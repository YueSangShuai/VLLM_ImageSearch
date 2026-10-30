#!/usr/bin/env python
# 主机参考分数：用官方 Qwen3VLReranker 对 calib/alerts_images 里的 ROI 图 × 规则文本打分。
# 产出：calib/host_scores.json（板端一致性对照）、calib/prompt_sample.txt（C++ prompt 对齐）、
#       calib/yes_no_ids.txt、calib/rerank_yes_no_diff.bin（W_yes-W_no, float32，hidden 兜底用）。
# 用法：CUDA_VISIBLE_DEVICES=6 <Qwen-env-python> scripts/host_ref_rerank.py
import argparse, json, os, sys, glob, collections

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/yuesang/LLM/FG-CLIP/models/Qwen3-VL-Reranker-2B")
ap.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "..", "calib"))
ap.add_argument("--max-per-rule", type=int, default=25)
ap.add_argument("--instruction", default="Given a pedestrian description, retrieve surveillance images of pedestrians matching it.")
ap.add_argument("--gpu", default="6")
args = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

sys.path.insert(0, os.path.join(args.model, "scripts"))
import torch
from qwen3_vl_reranker import Qwen3VLReranker

calib = os.path.abspath(args.calib)

# ---- 规则文本：从告警历史提取 ----
rules = collections.defaultdict(lambda: {"pos": set(), "imgs": []})
for f in sorted(glob.glob(os.path.join(calib, "alerts_history", "*.json"))):
    for a in json.load(open(f)):
        name = a.get("rule_name", "?")
        pos = a.get("positive", "") or ""
        if pos:
            rules[name]["pos"].add(pos)
for f in sorted(glob.glob(os.path.join(calib, "alerts_images", "*", "*", "*", "roi", "*.jpg"))):
    rule = f.split(os.sep)[-3]
    rules[rule]["imgs"].append(f)
rules = {k: v for k, v in rules.items() if v["pos"] and v["imgs"]}
print(f"rules: {[(k, len(v['pos']), len(v['imgs'])) for k, v in rules.items()]}")

model = Qwen3VLReranker(model_name_or_path=args.model, torch_dtype=torch.bfloat16)
# yes/no id + W_yes-W_no 向量（score_linear.weight 即差向量）
vocab = model.processor.tokenizer.get_vocab()
yes_id, no_id = vocab["yes"], vocab["no"]
diff = model.score_linear.weight[0].detach().float().cpu().numpy()
open(os.path.join(calib, "yes_no_ids.txt"), "w").write(f"{yes_id} {no_id}\n")
diff.astype("float32").tofile(os.path.join(calib, "rerank_yes_no_diff.bin"))
print(f"yes={yes_id} no={no_id} diff.shape={diff.shape}")

# ---- prompt 样例（C++ 渲染对齐用）----
sample = model.format_mm_instruction("一名穿黑色上衣的男人", None, None,
                                     None, rules["黑衣服男生"]["imgs"][0], None,
                                     instruction=args.instruction)
rendered = model.processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
open(os.path.join(calib, "prompt_sample.txt"), "w").write(rendered)
print("---- prompt sample ----\n" + rendered + "----")

# ---- 打分：每条规则自己的图（正例）+ 其它规则抽样图（跨规则负例），按 query 批量 ----
results = []
for name, v in sorted(rules.items()):
    texts = sorted(v["pos"])
    own = v["imgs"][: args.max_per_rule]
    cross = []
    for other, ov in sorted(rules.items()):
        if other != name:
            cross.extend(ov["imgs"][:3])
    for t in texts:
        for kind, imgs in (("pos", own), ("cross", cross)):
            if not imgs:
                continue
            inputs = {"instruction": args.instruction,
                      "query": {"text": t},
                      "documents": [{"image": p} for p in imgs]}
            scores = model.process(inputs)
            for p, s in zip(imgs, scores):
                results.append({"image": os.path.relpath(p, calib), "rule_name": name,
                                "query_text": t, "kind": kind, "score": round(float(s), 5)})
            pos = [s for s in scores]
            print(f"{name} | {t} | {kind} n={len(imgs)} mean={sum(pos)/len(pos):.3f} "
                  f"min={min(pos):.3f} max={max(pos):.3f}", flush=True)

out = os.path.join(calib, "host_scores.json")
json.dump({"instruction": args.instruction, "yes_id": yes_id, "no_id": no_id,
           "entries": results}, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out, len(results), "pairs")
