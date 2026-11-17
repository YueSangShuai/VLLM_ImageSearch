#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 StructBERT RKNN（板端输出）与宿主 fp32 onnxruntime 金标准。

输入：板上拉回的 tower_fp16.npy / tower_int8.npy [13,768]、
      pair_fp16.npy / pair_int8.npy [8,2]（与 cases.json 行序一致）
输出：塔嵌入逐用例 cos、平均误差；pair P(相似) 误差与关键判别用例保持情况。
"""
import json
import os

import numpy as np

OUT = "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/models/rk3588"
BOARD = "/home/web_vllm_rknn/sbert_bench"   # 拉回目录改这里
LOCAL_PULL = os.path.join(OUT, "board_out")


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def main():
    cases = json.load(open(os.path.join(OUT, "sbert_test", "cases.json")))
    gt = np.load(os.path.join(OUT, "sbert_golden_tower.npz"))["pooled"]
    gp = np.load(os.path.join(OUT, "sbert_golden_pair.npz"))["logits"]
    gpt = np.exp(gp) / np.exp(gp).sum(axis=-1, keepdims=True)

    def softmax(x):
        return np.exp(x - x.max(-1, keepdims=True)) / \
            np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)

    print("== 孪生塔 pooled [13,768] ==")
    for tag in ("tower_fp16", "tower_int8"):
        p = os.path.join(LOCAL_PULL, tag + ".npy")
        if not os.path.exists(p):
            print(f"  缺 {p}"); continue
        y = np.load(p)
        cos = (l2(gt) * l2(y)).sum(-1)
        print(f"[{tag}] cos(golden, rknn): mean={cos.mean():.5f} min={cos.min():.5f}")

    print("\n== pair P(相似) [8,2] ==")
    for tag in ("pair_fp16", "pair_int8"):
        p = os.path.join(LOCAL_PULL, tag + ".npy")
        if not os.path.exists(p):
            print(f"  缺 {p}"); continue
        y = np.load(p).reshape(len(cases["pairs"]), 2)
        pyt = softmax(y)
        d = np.abs(pyt[:, 1] - gpt[:, 1])
        print(f"[{tag}] |ΔP(相似)| max={d.max():.4f} mean={d.mean():.4f}")
        for (s_, t_), a, b in zip(cases["pairs"], gpt[:, 1], pyt[:, 1]):
            print(f"    {a:.4f} → {b:.4f}  {s_[:10]}… × {t_}")


if __name__ == "__main__":
    main()
