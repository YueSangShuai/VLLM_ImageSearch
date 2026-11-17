#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 RKNN 转换与板端精度对比准备数据（宿主运行）。

产出（均在 models/rk3588/ 下）：
  sbert_calib/    int8 校准 npy 三元组（真实告警分句 + 规则文本）
  sbert_calib/dataset_tower.txt / dataset_pair.txt   rknn.build 数据集列表
  sbert_test/     判别用测试集（单句三元组 + pair 拼接三元组 + cases.json）
  sbert_golden_tower.npz / sbert_golden_pair.npz     onnxruntime fp32 金标准
运行：/home/yuesang/miniconda3/envs/nomic/bin/python prepare_structbert_rknn.py
"""
import glob
import json
import os

import numpy as np

BASE = "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn"
MODEL_DIR = ("/data/yuesang/LLM/FG-CLIP/models/.ms_cache/damo/"
             "nlp_structbert_sentence-similarity_chinese-base")
OUT = os.path.join(BASE, "models", "rk3588")
CAL_DIR = os.path.join(OUT, "sbert_calib")
TEST_DIR = os.path.join(OUT, "sbert_test")
SEQ = 128
ALERT_GLOB = os.path.join(BASE, "output_cpp", "data", "alerts_history", "*.json")
TASKS = os.path.join(BASE, "output_cpp", "data", "tasks.json")
NOISE = ["无法", "难以", "看不清", "看不出", "看不清楚", "视角",
         "模糊", "氛围", "感觉", "图像显示", "画面显示", "画质", "分辨率"]


def clauses(desc):
    out = []
    for seg in desc.split("。"):
        for c in seg.split("，"):
            c = c.strip(" ，。")
            if c and not any(b in c for b in NOISE):
                out.append(c)
    return out or [desc]


def main():
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    os.makedirs(CAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # ---------- 语料 ----------
    calib_sents, rule_texts = [], []
    for f in sorted(glob.glob(ALERT_GLOB)):
        for it in json.load(open(f)):
            d = (it.get("description") or "").strip()
            if d:
                calib_sents.extend(clauses(d))
    for t in json.load(open(TASKS)):
        for r in t.get("rules") or []:
            for p in (r.get("positive") or []):
                rule_texts.append(p)
    calib_sents = list(dict.fromkeys(calib_sents))
    print(f"校准语料：告警分句 {len(calib_sents)} + 规则 {len(rule_texts)}")

    # 判别测试集：单句（塔）与 pair（交叉编码）
    A = "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包"
    B = "一名穿黑色上衣的男人"
    texts = [A, B] + clauses(A) + [
        "一名男性，穿着黑色上衣", "该人员穿黑色短袖",
        "一名黑衣男子，背着黑色背包", "画面显示一名男性，上身为黑色衣服",
        "一名男性，穿黑色上衣和深色裤子，背着黑色背包",
        "一名男性，穿着白色上衣，旁边放着黑色背包", "一名男性，穿着蓝色上衣",
        "一名穿白色连衣裙的女性"]
    pairs = [(B, A), (B, "一名男性，穿着黑色上衣"), (B, "背着一个黑色背包"),
             (B, "穿着红色上衣和深色裤子"), (B, "一名男性"),
             (A, A), (B, B), (B, "一名男性，穿着白色上衣，旁边放着黑色背包")]

    # ---------- token 化（int32，定长 128） ----------
    def enc(text_or_pair, tag, d):
        if isinstance(text_or_pair, tuple):
            e = tok(text_or_pair[0], text_or_pair[1], max_length=SEQ,
                    truncation=True, padding="max_length")
        else:
            e = tok(text_or_pair, max_length=SEQ, truncation=True,
                    padding="max_length")
        return (np.array(e["input_ids"], dtype=np.int32)[None],
                np.array(e["attention_mask"], dtype=np.int32)[None],
                np.array(e["token_type_ids"], dtype=np.int32)[None])

    def dump(items, prefix, d, listfile, abs_paths=False):
        names = []
        for k, item in enumerate(items):
            ids, am, tt = enc(item, prefix, d)
            f = os.path.join(d, f"{prefix}_{k:03d}")
            np.save(f + "_input_ids.npy", ids)
            np.save(f + "_attention_mask.npy", am)
            np.save(f + "_token_type_ids.npy", tt)
            names.append([f"{prefix}_{k:03d}_{s}.npy" for s in
                          ("input_ids", "attention_mask", "token_type_ids")])
        # 宿主 rknn.build 按 CWD 解析相对路径 → 校准列表必须绝对路径；板上用裸文件名
        if abs_paths:
            names = [[os.path.join(d, p) for p in row] for row in names]
        open(os.path.join(d, listfile), "w").write(
            "\n".join(" ".join(row) for row in names) + "\n")
        return len(items)

    # 校准集：塔用分句/规则；pair 模型用「规则 × 分句」拼接（与线上调用分布一致）
    cal_pairs = [(r, c) for r in rule_texts[:18] for c in calib_sents[:4]][:72]
    n1 = dump((calib_sents + rule_texts)[:300], "cal", CAL_DIR,
              "dataset_tower.txt", abs_paths=True)
    n2 = dump(cal_pairs, "pair", CAL_DIR, "dataset_pair.txt", abs_paths=True)
    print(f"校准 npy：塔 {n1} 条，pair {n2} 条")

    # 测试集
    dump(texts, "t", TEST_DIR, "list_tower.txt")
    dump(pairs, "p", TEST_DIR, "list_pair.txt")
    json.dump({"texts": texts, "pairs": [list(p) for p in pairs]},
              open(os.path.join(TEST_DIR, "cases.json"), "w"),
              ensure_ascii=False, indent=1)

    # ---------- fp32 金标准（onnxruntime CPU） ----------
    so = ort.SessionOptions()
    feeds_for = lambda d, prefix, k: {
        "input_ids": np.load(os.path.join(d, f"{prefix}_{k:03d}_input_ids.npy")),
        "attention_mask": np.load(os.path.join(d, f"{prefix}_{k:03d}_attention_mask.npy")),
        "token_type_ids": np.load(os.path.join(d, f"{prefix}_{k:03d}_token_type_ids.npy"))}

    s_t = ort.InferenceSession(os.path.join(OUT, "sbert_tower.onnx"), so,
                               providers=["CPUExecutionProvider"])
    emb = np.concatenate([s_t.run(None, feeds_for(TEST_DIR, "t", k))[0]
                          for k in range(len(texts))], axis=0)
    np.savez(os.path.join(OUT, "sbert_golden_tower.npz"), pooled=emb)

    s_p = ort.InferenceSession(os.path.join(OUT, "sbert_pair.onnx"), so,
                               providers=["CPUExecutionProvider"])
    logits = np.concatenate([s_p.run(None, feeds_for(TEST_DIR, "p", k))[0]
                             for k in range(len(pairs))], axis=0)
    np.savez(os.path.join(OUT, "sbert_golden_pair.npz"), logits=logits)
    p_sim = np.exp(logits[:, 1]) / (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
    print("\nfp32 金标准 P(相似)：")
    for (s_, t_), v in zip(pairs, p_sim):
        print(f"    {v:.4f}  {s_[:10]}… × {t_}")


if __name__ == "__main__":
    main()
