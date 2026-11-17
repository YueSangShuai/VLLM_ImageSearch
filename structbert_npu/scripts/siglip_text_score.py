#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SigLIP2-reid 文本塔（web_siglip_reid_rknn）：规则 × 描述 → 文本向量 cosine。

口径（与线上 C++/导出脚本一致）：Gemma 分词 padding="max_length" 到 64（pad_id=0），
mask=ids≠0 右填充屏蔽，文本塔输出 pooler L2 归一化 768-d，两向量直接点积。
对照组：pair_score.py（bge）、qwen3_reranker_score.py（Qwen3-R-0.6B，6/6）、
NOTES.md 里的 StructBERT 表。中文+英文（线上规则是英文属性短句）双口径。
运行（rknn 环境）：
  /home/yuesang/miniconda3/envs/rknn/bin/python siglip_text_score.py
"""
import os

ROOT = "/data/yuesang/LLM/FG-CLIP/web_siglip_reid_rknn"
ONNX_FP32 = os.path.join(ROOT, "onnx", "siglip_text.onnx")
ONNX_INT8 = os.path.join(ROOT, "onnx", "siglip_text_int8.onnx")
BASE = os.path.join(ROOT, "models", "siglip2-base-patch16-224")

# ==================== 在这里改输入 ====================
R_ZH = "一名穿黑色上衣的男人"
R_EN = "man, black upper clothing"
PAIRS = [
    (R_ZH, R_EN, "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包",
     "man, red upper clothing, dark pants, black backpack", "低"),   # 红≠黑
    (R_ZH, R_EN, "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包",
     "man, black upper clothing, dark pants, black backpack", "高"),  # 关键稀释案例
    (R_ZH, R_EN, "一名男性，穿着黑色上衣",
     "man, black upper clothing", "高"),
    (R_ZH, R_EN, "该人员穿黑色短袖",
     "person, black short-sleeve shirt", "高"),                       # 词汇缺口
    (R_ZH, R_EN, "一名男性，穿着蓝色上衣",
     "man, blue upper clothing", "低"),
    (R_ZH, R_EN, "一名穿白色连衣裙的女性",
     "woman, white dress", "低"),
]
# ======================================================

SEQ = 64


def embed(sess, tok, texts):
    import numpy as np
    vecs = []
    for t in texts:
        e = tok(t, padding="max_length", max_length=SEQ, truncation=True,
                return_tensors="np")
        ids = e["input_ids"].astype("int32")
        mask = (ids != 0).astype("int32")          # 与导出脚本同法
        v = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
        vecs.append(v)
    return np.stack(vecs)


def main():
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)

    rows = []
    for name, path in [("fp32", ONNX_FP32), ("int8", ONNX_INT8)]:
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        zs = embed(s, tok, [p[0] for p in PAIRS])       # 规则（中文）
        ze = embed(s, tok, [p[2] for p in PAIRS])       # 描述（中文）
        es = embed(s, tok, [p[1] for p in PAIRS])       # 规则（英文）
        ee = embed(s, tok, [p[3] for p in PAIRS])       # 描述（英文）
        czh = (zs * ze).sum(1)
        cen = (es * ee).sum(1)
        rows.append((name, czh, cen))

    for i, (rz, re_, dz, de, exp) in enumerate(PAIRS):
        print(f"期望{exp}  {rz} × {dz}")
        for name, czh, cen in rows:
            print(f"    {name}  cos中文={czh[i]:+.4f}  cos英文={cen[i]:+.4f}")
    # 自检：fp32/int8 向量一致性
    if len(rows) == 2:
        print(f"[自检] int8 对 fp32 的 cos 漂移上限: "
              f"{np.abs(rows[0][1]-rows[1][1]).max():.4f} / "
              f"{np.abs(rows[0][2]-rows[1][2]).max():.4f}")


if __name__ == "__main__":
    main()
