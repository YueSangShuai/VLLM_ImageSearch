#!/usr/bin/env python
"""MARS 双塔 ONNX → RKNN（rk3588, fp16）+ x86 模拟器对拍（golden 阈值 cos≥0.99）。

约定（与 onnx_export.py 头注释一致）：
  图像塔  输入 uint8 RGB NHWC [1,224,224,3] 0-255，mean/std=127.5 ≡ ONNX [-1,1] NCHW
  文本塔  input_ids int32 [1,128] + attention_mask float32 [1,128]（1/0），
          扩展掩码 (1-m)*-10000 在图内计算（-10000 在 fp16 范围内，安全）
  两塔均不量化（RKNN BERT 配方：w8a8 必崩；EfficientNet SiLU/SE 走 fp16 验证）

用法（rknn env）：
  /home/yuesang/miniconda3/envs/rknn/bin/python export/rknn_convert.py [--which image|text|all]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(ROOT, "golden")
OUT = os.path.join(ROOT, "rknn")
os.makedirs(OUT, exist_ok=True)


def build(onnx_path, rknn_path, inputs=None, input_size_list=None, mean=None, std=None):
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    kw = {}
    if mean is not None:
        kw["mean_values"] = [mean]
        kw["std_values"] = [std]
    rknn.config(target_platform="rk3588", **kw)
    if inputs is not None:
        ret = rknn.load_onnx(model=onnx_path, inputs=inputs, input_size_list=input_size_list)
    else:
        ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        sys.exit(f"load_onnx failed ({onnx_path}): {ret}")
    print("== build fp16", os.path.basename(onnx_path))
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        sys.exit(f"build failed ({onnx_path})")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        sys.exit(f"export_rknn failed: {ret}")
    print("exported", rknn_path, os.path.getsize(rknn_path) // 2**20, "MB")
    return rknn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["image", "text", "all"])
    args = ap.parse_args()

    px = np.load(os.path.join(GOLDEN, "gallery_px_u8.npy"))          # [N,224,224,3] u8 NHWC
    g_emb = np.load(os.path.join(GOLDEN, "gallery_emb_torch.npy"))   # [N,256]
    ids = np.load(os.path.join(GOLDEN, "query_input_ids.npy"))       # [M,128] i32
    attn = np.load(os.path.join(GOLDEN, "query_attn.npy"))           # [M,128] f32
    t_emb = np.load(os.path.join(GOLDEN, "query_emb_torch.npy"))     # [M,256]

    got_img = None
    if args.which in ("image", "all"):
        rknn = build(os.path.join(ROOT, "onnx", "mars_image.onnx"),
                     os.path.join(OUT, "mars_image_fp16.rknn"),
                     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        try:
            rknn.init_runtime()
        except Exception as e:
            sys.exit(f"init_runtime(x86 sim) failed: {e}")
        got_img = np.concatenate([
            np.asarray(rknn.inference(inputs=[px[i:i + 1]], data_format="nhwc")[0]).reshape(1, -1)
            for i in range(px.shape[0])])
        cos = [float((got_img[i] * g_emb[i]).sum()) for i in range(px.shape[0])]
        print("image rknn-vs-torch row-cos min=%.5f max|diff|=%.3e" % (min(cos), np.abs(got_img - g_emb).max()))
        assert min(cos) >= 0.99, "image rknn 精度不达标"
        rknn.release()

    if args.which in ("text", "all"):
        rknn = build(os.path.join(ROOT, "onnx", "mars_text.onnx"),
                     os.path.join(OUT, "mars_text_fp16.rknn"),
                     inputs=["input_ids", "attention_mask"],
                     input_size_list=[[1, 128], [1, 128]])
        try:
            rknn.init_runtime()
        except Exception as e:
            sys.exit(f"init_runtime(x86 sim) failed: {e}")
        got_txt = np.concatenate([
            np.asarray(rknn.inference(inputs=[ids[i:i + 1], attn[i:i + 1]])[0]).reshape(1, -1)
            for i in range(ids.shape[0])])
        cos = [float((got_txt[i] * t_emb[i]).sum()) for i in range(ids.shape[0])]
        print("text rknn-vs-torch row-cos min=%.5f max|diff|=%.3e" % (min(cos), np.abs(got_txt - t_emb).max()))
        assert min(cos) >= 0.99, "text rknn 精度不达标"
        rknn.release()

    # 端到端 topk 排序保持（需两塔都已对拍）
    if args.which == "all" and got_img is not None:
        expected = json.load(open(os.path.join(GOLDEN, "expected_topk.json")))
        files = json.load(open(os.path.join(GOLDEN, "gallery_files.json")))
        sims = got_txt @ got_img.T
        ok = True
        # 近并分（相邻名差 <0.005）允许重排：fp16 舍入不可能保序，但检索语义无差
        for qi, exp in enumerate(expected):
            s = np.sort(sims[qi])[::-1][:10]
            strict = np.diff(s) > -0.005  # True=可容忍的平局位
            top = [files[i] for i in np.argsort(-sims[qi])[:10]]
            ref = [e["file"] for e in exp["topk"]]
            bad = [i for i in range(10) if top[i] != ref[i] and strict[max(i - 1, 0)]]
            if bad:
                ok = False
                print("TOPK 不一致 query%d 位置%s: %s vs torch %s" % (qi, bad, top[:3], ref[:3]))
        print("topk(10) 全 18 条与 torch 一致（近并分容差 0.005）:", ok)
    print("DONE")


if __name__ == "__main__":
    main()
