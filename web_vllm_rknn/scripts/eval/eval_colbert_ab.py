#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BGE(线上现状) vs GTE-ModernColBERT(ColBERT late-interaction) 离线 A/B。

目的：验证「黑色裤子+白色上衣」这类属性错绑定的描述，在 MaxSim 打分下
能否与规则「穿黑色上衣的人」拉开分差，同时真告警描述不掉分。

打分口径与 C++ 线上一致：
  - 规则文本 = query（BGE 加中文指令前缀 / ColBERT 加 "[Q] "），
    描述按「。」→「，」分句（剔除噪声分句）= doc（BGE 无前缀 / ColBERT "[D] "），
    规则对分句取 max。
  - 统一用 maxsim 实现：单向量模型退化为 cos（1×1 情形）。

数据：output_cpp/data/tasks.json 的规则 + alerts_history 真实告警描述。
运行：python3 scripts/eval/eval_colbert_ab.py
     GTE_ONNX=... 可切换 fp32/int8；默认 models/GTE-ModernColBERT-v1/model.onnx
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS = "/data/yuesang/LLM/FG-CLIP/models"
GTE_DIR = os.path.join(MODELS, "GTE-ModernColBERT-v1")
BGE_NAME = os.path.join(MODELS, "bge-small-zh-v1.5")
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TASKS = os.path.join(REPO, "output_cpp", "data", "tasks.json")
ALERT_GLOB = os.path.join(REPO, "output_cpp", "data", "alerts_history", "*.json")

# 与 matcher.cpp isNoiseClause / buildEmbeddingClauses 对齐
NOISE = ["无法", "难以", "看不清", "看不出", "看不清楚", "视角",
         "模糊", "氛围", "感觉", "图像显示", "画面显示", "画质", "分辨率"]


def build_clauses(desc: str):
    out = []
    for seg in desc.split("。"):
        for clause in seg.split("，"):
            c = clause.strip(" ，。")
            if not c or any(b in c for b in NOISE):
                continue
            out.append(c)
    return out or [desc]


def load_rules():
    rules = []
    for task in json.load(open(TASKS)):
        for r in task.get("rules") or []:
            pos = [p for p in (r.get("positive") or [r.get("text", "")]) if p]
            neg = [n for n in (r.get("negative") or []) if n]
            if pos:
                rules.append({"id": str(r.get("id", "")), "name": r.get("name", ""),
                              "pos": pos, "neg": neg})
    return rules


def load_alerts():
    alerts = []
    for f in sorted(glob.glob(ALERT_GLOB)):
        for it in json.load(open(f)):
            desc = (it.get("description") or "").strip()
            if not desc:
                continue
            alerts.append({
                "id": it.get("id", ""),
                "rule": str(it.get("rule_id", it.get("matched_rule_id", ""))),
                "rule_name": it.get("rule_name", ""),
                "description": desc,
            })
    return alerts


def maxsim(q, d):
    """q:[m,dim] d:[n,dim]，均已逐行 L2 归一 → 每 query token 最大 cos 的平均。"""
    return float(np.max(d @ q.T, axis=0).mean())


# ==================== 打分器 ====================

class BgeScorer:
    name = "bge-small-zh cos"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(BGE_NAME, device="cpu")

    def embed_queries(self, texts):
        v = np.asarray(self.m.encode([BGE_QUERY_PREFIX + t for t in texts],
                                     normalize_embeddings=True,
                                     convert_to_numpy=True), dtype=np.float32)
        return [x[None, :] for x in v]

    def embed_docs(self, clauses):
        v = np.asarray(self.m.encode(clauses, normalize_embeddings=True,
                                     convert_to_numpy=True), dtype=np.float32)
        return [x[None, :] for x in v]


class GteMcScorer:
    name = "gte-moderncolbert maxsim"

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(GTE_DIR, "tokenizer.json"))
        onnx = os.environ.get("GTE_ONNX", os.path.join(GTE_DIR, "model.onnx"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()
                         if i.name in ("input_ids", "attention_mask")]
        self.out_name = self.sess.get_outputs()[0].name
        cfg = json.load(open(os.path.join(GTE_DIR, "onnx_config.json")))
        self.qid, self.did, self.pad = cfg["query_prefix_id"], cfg["document_prefix_id"], cfg["pad_token_id"]
        self.cls, self.sep = 50281, 50282
        self.projection = self._probe_projection()

    def _probe_projection(self):
        outs = self._run([self.tok.encode("[Q] 测试test").ids])
        dim = outs.shape[-1]
        print(f"[gte] onnx 输出 {self.out_name} shape={outs.shape}")
        if dim != 768:
            return None  # ONNX 已含投影头
        from safetensors import safe_open
        st = os.path.join(GTE_DIR, "model.safetensors")
        with safe_open(st, framework="numpy") as f:
            keys = list(f.keys())
            print("[gte] safetensors 未匹配投影头，keys 示例:", [k for k in keys if "embed" in k or "proj" in k or "linear" in k][:10])
            for k in ("embedding.weight", "projection.weight", "colbert_linear.out_proj.weight"):
                if k in f.keys():
                    w = f.get_tensor(k).astype(np.float32)
                    print(f"[gte] 外挂投影头 {k} shape={w.shape}")
                    return w
        raise RuntimeError("ONNX 输出 768 维 hidden 且未找到投影头")

    def _run(self, id_lists):
        n = len(id_lists)
        max_len = max(len(x) for x in id_lists)
        ids_b = np.full((n, max_len), self.pad, dtype=np.int64)
        for i, x in enumerate(id_lists):
            ids_b[i, :len(x)] = x
        feeds = {}
        if "input_ids" in self.in_names:
            feeds["input_ids"] = ids_b
        if "attention_mask" in self.in_names:
            feeds["attention_mask"] = (ids_b != self.pad).astype(np.int64)
        return self.sess.run([self.out_name], feeds)[0].astype(np.float32)

    def _tokenize(self, text, prefix):
        ids = self.tok.encode(prefix + text).ids
        keep = np.array([i for i, t in enumerate(ids)
                         if t not in (self.cls, self.sep, self.pad, self.qid, self.did)],
                        dtype=np.int64)
        return ids, keep

    def _encode(self, texts, prefix):
        tok_out = [self._tokenize(t, prefix) for t in texts]
        hs = self._run([ids for ids, _ in tok_out])
        if self.projection is not None:
            hs = hs @ self.projection.T
        hs = hs / np.maximum(np.linalg.norm(hs, axis=-1, keepdims=True), 1e-9)
        return [hs[i][keep] if len(keep) else hs[i, :1]
                for i, (_, keep) in enumerate(tok_out)]

    def embed_queries(self, texts):
        return self._encode(texts, "[Q] ")

    def embed_docs(self, clauses):
        return self._encode(clauses, "[D] ")


def rule_score(sc, q_mats, doc_mats):
    """规则(query 多条) × 单条告警的分句集：全部取 max，与 C++ 双层 max 一致。"""
    return max(maxsim(q, d) for q in q_mats for d in doc_mats)


# ==================== 评估 ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "scripts", "eval", "eval_colbert_ab.json"))
    args = ap.parse_args()

    rules = load_rules()
    alerts = load_alerts()
    clauses_per_alert = [build_clauses(a["description"]) for a in alerts]
    offsets = np.cumsum([0] + [len(c) for c in clauses_per_alert])
    all_clauses = [c for cs in clauses_per_alert for c in cs]
    print(f"规则 {len(rules)} 条，告警 {len(alerts)} 条，分句 {len(all_clauses)} 个")

    scorers = [BgeScorer(), GteMcScorer()]

    report = {"latency_ms_per_clause": {}, "top1_hit_rate": {}, "per_rule": {}, "fp_cases": []}
    score_mats = {}
    for sc in scorers:
        t0 = time.time()
        docs = sc.embed_docs(all_clauses)
        report["latency_ms_per_clause"][sc.name] = round((time.time() - t0) * 1000 / len(all_clauses), 1)

        S = np.zeros((len(alerts), len(rules)), dtype=np.float32)
        per_rule = []
        for ri, r in enumerate(rules):
            q = sc.embed_queries(r["pos"])
            for ai in range(len(alerts)):
                S[ai, ri] = rule_score(sc, q, docs[offsets[ai]:offsets[ai + 1]])
            tp = [round(float(S[ai, ri]), 3)
                  for ai in range(len(alerts)) if alerts[ai]["rule"] == r["id"]]
            row = {"id": r["id"], "name": r["name"],
                   "tp_n": len(tp),
                   "tp_min": min(tp) if tp else None,
                   "tp_med": round(float(np.median(tp)), 3) if tp else None}
            # 负样本分离：neg 文本作为 doc，与规则 query 的最高分（越高说明 neg 与 pos 难分）
            if r["neg"]:
                qn = sc.embed_queries(r["neg"])
                row["pos_vs_neg_max"] = round(max(
                    maxsim(qa, qb) for qa in q for qb in qn), 3)
            per_rule.append(row)
        score_mats[sc.name] = S

        top1 = 0
        for ai, a in enumerate(alerts):
            if a["rule"] and int(np.argmax(S[ai])) == next(
                    (i for i, r in enumerate(rules) if r["id"] == a["rule"]), -1):
                top1 += 1
        hit = sum(1 for a in alerts if a["rule"])
        report["top1_hit_rate"][sc.name] = f"{top1}/{hit}"
        report["per_rule"][sc.name] = per_rule
        print(f"[{sc.name}] {report['latency_ms_per_clause'][sc.name]}ms/分句  "
              f"top1 命中: {top1}/{hit}")

    # 错绑定 FP 专项：黑色上衣类规则 × 提及白色/深色上衣的告警
    names = [r["name"] or r["pos"][0] for r in rules]
    fp_rows = []
    for ai, a in enumerate(alerts):
        d = a["description"]
        if ("白色上衣" in d or "白色短袖" in d or "白上衣" in d) and ("黑色" in d or "黑裤" in d):
            for ri, r in enumerate(rules):
                ptxt = " ".join(r["pos"])
                if "黑色上衣" in ptxt or "黑色短袖" in ptxt:
                    fp_rows.append({"desc": d[:60], "rule": names[ri],
                                    **{sc: round(float(S[ai, ri]), 3)
                                       for sc, S in score_mats.items()}})
    bge_name, gte_name = [sc.name for sc in scorers]
    # 同一告警在生产规则上的 TP 分对照
    tp_rows = []
    for ai, a in enumerate(alerts):
        for ri, r in enumerate(rules):
            if r["id"] == a["rule"]:
                tp_rows.append({"desc": a["description"][:60], "rule": names[ri],
                                **{bge_name: round(float(score_mats[bge_name][ai, ri]), 3),
                                   gte_name: round(float(score_mats[gte_name][ai, ri]), 3)}})
    report["fp_mismatch_rows"] = fp_rows[:40]
    report["tp_rows"] = sorted(tp_rows, key=lambda x: x[gte_name])[:40]

    report["summary"] = {
        bge_name: {"tp_min_all": min([r["tp_min"] for r in report["per_rule"][bge_name] if r["tp_min"]]),
                   "fp_max_all": (max((row[bge_name] for row in fp_rows), default=None))},
        gte_name: {"tp_min_all": min([r["tp_min"] for r in report["per_rule"][gte_name] if r["tp_min"]]),
                   "fp_max_all": (max((row[gte_name] for row in fp_rows), default=None))},
    }

    json.dump({"alerts": [{k: a[k] for k in ("id", "rule", "rule_name")} for a in alerts],
               "rules": [{"id": r["id"], "name": r["name"], "pos": r["pos"], "neg": r["neg"]}
                         for r in rules],
               "scores": {k: v.tolist() for k, v in score_mats.items()},
               "report": report},
              open(args.out, "w"), ensure_ascii=False, indent=1)

    print("\n===== 汇总 =====")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    print(f"\n错绑定 FP 案例 {len(fp_rows)} 组（前 15）:")
    for row in fp_rows[:15]:
        print(" ", row)
    print(f"\n报告写入 {args.out}")


if __name__ == "__main__":
    main()
