# -*- coding: utf-8 -*-
"""W8A8 vs bf16 A/B 指标汇总。

输入（同 --set 目录）：
  gt.json          54 属性 GT（make_set.py 产出）
  host_bf16.json   host 结构化基线（run_host_bf16.py）
  desc_bf16.json   host 描述基线（run_host_bf16_desc.py）
  w8a8.jsonl       板端批量输出（ab_dump，tag=structured/description）
  board_rules/rules.json(+tasks.json 阈值)  线上规则（BGE 描述打分用）

指标：
  structured: vs PA-100K GT micro-P/R/F1（-1 未标注位剔除）、W8A8 对 bf16 一致率(F1)、
              非法 token 数、枚举复读率、头属性对比
  description: BGE(线上规则口径) 每图最优规则分 bf16 vs W8A8、超阈告警率、长度/hedge 计数
输出 metrics.json + stdout json。
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/yuesang/LLM/FG-CLIP/qwen_classifer")
from w8a8_common import ATTRS, parse_attr_text, is_enumeration

ap = argparse.ArgumentParser()
ap.add_argument("--set", default=HERE)
ap.add_argument("--no-bge", action="store_true", help="跳过描述 BGE 打分")
args = ap.parse_args()
S = args.set

gt = json.load(open(os.path.join(S, "gt.json")))
host = json.load(open(os.path.join(S, "host_bf16.json")))
desc = json.load(open(os.path.join(S, "desc_bf16.json")))
board = {}
with open(os.path.join(S, "w8a8.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        board.setdefault(r["tag"], {})[r["file"]] = r

gt_by_file = {it["file"]: it["labels"] for it in gt["items"]}
files = [it["file"] for it in gt["items"]]


def micro_f1(pred_sets):
    """pred_sets: {file: [attr,...]}；只统计 GT 已标注位(0/1)，-1 未标注剔除。"""
    tp = fp = fn = 0
    for f, preds in pred_sets.items():
        labels = gt_by_file[f]
        P = set(preds)
        for a in ATTRS:
            lab = labels.get(a, -1)
            if lab not in (0, 1):
                continue
            p = a in P
            tp += p and lab == 1
            fp += p and lab == 0
            fn += (not p) and lab == 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, P=round(prec, 4), R=round(rec, 4), F1=round(f1, 4))


def agreement_f1(ref_sets, test_sets):
    """test 对 ref 的集合级 micro-F1（ref 作伪 GT）。"""
    tp = fp = fn = 0
    for f in files:
        R, P = set(ref_sets.get(f, [])), set(test_sets.get(f, []))
        tp += len(R & P); fp += len(P - R); fn += len(R - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(P=round(prec, 4), R=round(rec, 4), F1=round(f1, 4))


out = {}

# ==================== structured ====================
if "structured" in board:
    h_str = {it["file"]: it["valid"] for it in host["items"]}
    h_head = {it["file"]: it.get("head_attrs", []) for it in host["items"]}
    b_valid = {f: parse_attr_text(board["structured"][f]["raw"])[0] for f in files}
    b_head = {f: board["structured"][f].get("head_attrs", []) for f in files}

    st = out["structured"] = {}
    st["gt_micro_f1"] = {"host_bf16_valid": micro_f1(h_str),
                         "w8a8_valid": micro_f1(b_valid),
                         "host_bf16_head": micro_f1(h_head),
                         "w8a8_head": micro_f1(b_head)}
    st["w8a8_vs_bf16_agreement"] = {"valid": agreement_f1(h_str, b_valid),
                                    "head": agreement_f1(h_head, b_head)}
    inv = Counter()
    enum = 0
    for f in files:
        valid, invalid = parse_attr_text(board["structured"][f]["raw"])
        inv.update(invalid)
        enum += is_enumeration(valid)
    st["w8a8_invalid_tokens"] = {"total": sum(inv.values()),
                                 "per_img": round(sum(inv.values()) / len(files), 2),
                                 "top10": inv.most_common(10)}
    st["w8a8_enumeration_imgs"] = enum
    st["bf16_invalid_tokens_total"] = sum(len(it["invalid"]) for it in host["items"])
    st["bf16_enumeration_imgs"] = sum(is_enumeration(it["valid"]) for it in host["items"])
    diffs = sorted(files, key=lambda f: -len(set(h_str[f]) ^ set(b_valid[f])))
    st["top_divergent_examples"] = [
        {"file": f, "bf16_valid": h_str[f],
         "w8a8_valid": b_valid[f], "w8a8_raw": board["structured"][f]["raw"][:200]}
        for f in diffs[:3]]

# ==================== description ====================
if "description" in board and not args.no_bge:
    import re
    import torch
    from sentence_transformers import SentenceTransformer

    BGE = "/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5"
    PREFIX = "为这个句子生成表示以用于检索相关文章："
    # 规则文本字段候选。注意线上 BGE query 具体用哪个字段与 matcher 实现相关，
    # 这里用「正向描述」优先（positive/feature/description），退化到 name。
    RULE_KEYS = ("positive", "feature", "features", "description", "rule", "query",
                 "text", "content", "name")

    def rule_text(r):
        if isinstance(r, str):
            return r
        for k in RULE_KEYS:
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    # rules.json 现网为空 []；规则在 tasks.json 的 tasks[].rules[] 里。
    # 线上口径：每条规则有 positive query 列表（BGE query 侧加前缀）+ threshold。
    rules, thr_default = [], 0.45
    tasks = json.load(open(os.path.join(S, "board_rules", "tasks.json")))
    for t in tasks if isinstance(tasks, list) else []:
        for r in t.get("rules", []) if isinstance(t, dict) else []:
            qs = [q for q in (r.get("positive") or []) if isinstance(q, str) and q.strip()]
            if not qs:
                rt = rule_text(r)
                qs = [rt] if rt else []
            if not qs:
                continue
            th = r.get("threshold")
            rules.append({"name": r.get("name", qs[0]), "queries": qs,
                          "thr": th if isinstance(th, (int, float)) else thr_default,
                          "enabled": r.get("enabled", True)})

    m = SentenceTransformer(BGE, device="cuda:0" if torch.cuda.is_available() else "cpu")

    def clauses(text):
        parts = [p.strip() for p in re.split(r"[。！？；，,]", text) if len(p.strip()) >= 4]
        return parts or [text]

    h_desc = {it["file"]: it["raw"] for it in desc["items"]}
    b_desc = {f: board["description"][f]["raw"] for f in files}
    # 全部 positive query 展平编码；按 rule 归组取 max
    q_texts, q_owner = [], []
    for i, r in enumerate(rules):
        for q in r["queries"]:
            q_texts.append(q)
            q_owner.append(i)
    Q = m.encode([PREFIX + q for q in q_texts], normalize_embeddings=True, convert_to_numpy=True)

    def per_image(dmap):
        """-> (每图最优规则分 list, 每规则逐图分 dict, 每图是否过阈告警)"""
        best, alert = [], []
        per_rule = {f"{i}:{r['name']}": [] for i, r in enumerate(rules)}
        for f in files:
            C = m.encode(clauses(dmap[f]), normalize_embeddings=True, convert_to_numpy=True)
            S_ = Q @ C.T                                  # [queries, clauses]
            q_best = S_.max(axis=1)                       # 每 query 取本图最佳分句
            rule_scores = []
            for i, r in enumerate(rules):
                s = max(float(q_best[j]) for j in range(len(q_texts)) if q_owner[j] == i)
                rule_scores.append(s)
                if r["enabled"]:
                    per_rule[f"{i}:{r['name']}"].append(s)
            best.append(max(rule_scores))
            alert.append(any(s >= r["thr"] for s, r in zip(rule_scores, rules) if r["enabled"]))
        return best, per_rule, alert

    hb, h_pr, h_al = per_image(h_desc)
    bb, b_pr, b_al = per_image(b_desc)
    deltas = [x - y for x, y in zip(bb, hb)]
    rule_delta = sorted(({"rule": key,
                          "bf16": round(float(np.mean(vals)), 4),
                          "w8a8": round(float(np.mean(b_pr[key])), 4),
                          "delta": round(float(np.mean(b_pr[key]) - np.mean(vals)), 4)}
                         for key, vals in h_pr.items()), key=lambda x: x["delta"])

    out["description"] = {
        "n_rules": len(rules),
        "best_rule_score": {"mean_bf16": round(float(np.mean(hb)), 4),
                            "mean_w8a8": round(float(np.mean(bb)), 4),
                            "delta_mean": round(float(np.mean(deltas)), 4),
                            "delta_min": round(float(np.min(deltas)), 4),
                            "delta_max": round(float(np.max(deltas)), 4)},
        "alert_rate": {"bf16": round(float(np.mean(h_al)), 3),
                       "w8a8": round(float(np.mean(b_al)), 3),
                       "note": "任意图上任规则分≥板端阈值(缺省0.45)记告警"},
        "len_mean_chars": {"bf16": round(float(np.mean([len(h_desc[f]) for f in files])), 1),
                           "w8a8": round(float(np.mean([len(b_desc[f]) for f in files])), 1)},
        "hedge_hits_per_img": {
            "bf16": round(float(np.mean([sum(h_desc[f].count(k) for k in ("由于", "无法", "看不清", "模糊"))
                                         for f in files])), 2),
            "w8a8": round(float(np.mean([sum(b_desc[f].count(k) for k in ("由于", "无法", "看不清", "模糊"))
                                         for f in files])), 2)},
        "per_rule_delta_worst5": rule_delta[:5],
    }

with open(os.path.join(S, "metrics.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
