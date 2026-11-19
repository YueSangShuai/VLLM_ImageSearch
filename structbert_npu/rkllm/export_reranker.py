#!/usr/bin/env python
"""Qwen3-Reranker-0.6B → RKLLM w8a8 (rk3588, 3 核 NPU)。

踩坑对齐（web_vllm_rknn/PLAN.md）：
- 必须 CWD 在 ext4 卷（/tmp）再跑，否则 export 阶段报 Not enough free space。
- 校准集 = 线上真实规则(英文短句) + 6 案中文描述，量化误差直接对准使用分布。
用法：cd /tmp/rkllm_run_rr && <rkllm-env-python> export_reranker.py
"""
import json
import os

from rkllm.api import RKLLM

MODEL = "/data/yuesang/LLM/FG-CLIP/models/.ms_cache/Qwen/Qwen3-Reranker-0.6B"
TASKS = "/data/yuesang/LLM/FG-CLIP/web_config/web_siglip_reid_rknn/data/tasks.json"
OUT = "/data/yuesang/LLM/FG-CLIP/structbert_npu/rkllm/qwen3-reranker-0.6b_w8a8_rk3588.rkllm"

CALIB_ZH = [
    "一名穿黑色上衣的男人",
    "一名男性，穿着红色上衣和深色裤子，背着一个黑色背包",
    "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包",
    "一名男性，穿着黑色上衣",
    "该人员穿黑色短袖",
    "一名男性，穿着蓝色上衣",
    "一名穿白色连衣裙的女性",
    "man, black upper clothing",
    "woman, white dress",
    "person, black short-sleeve shirt",
]


def rules_from_tasks(path, limit=14):
    out = []
    try:
        tasks = json.load(open(path))
    except Exception as e:                      # noqa: BLE001
        print("tasks.json 不可用，跳过真实规则:", e)
        return out
    for t in tasks:
        for r in t.get("rules", []) if isinstance(t, dict) else []:
            for k in ("text", "positive"):
                v = r.get(k)
                if isinstance(v, str) and len(v) > 3:
                    out.append(v)
                elif isinstance(v, list):
                    out.extend(x for x in v if isinstance(x, str) and len(x) > 3)
        if len(out) >= limit:
            break
    return out[:limit]


def main():
    calib = [{"input": s, "target": ""} for s in
             rules_from_tasks(TASKS) + CALIB_ZH]
    ds = "/tmp/rkllm_run_rr/calib.json"
    os.makedirs("/tmp/rkllm_run_rr", exist_ok=True)
    assert os.path.dirname(os.path.abspath(os.getcwd())) == "/tmp", \
        "必须 cd /tmp/rkllm_run_rr 后运行（ext4 卷）"
    json.dump(calib, open(ds, "w"), ensure_ascii=False, indent=1)
    print(f"calib samples: {len(calib)}")

    llm = RKLLM()
    assert llm.load_huggingface(model=MODEL, device="cpu") == 0, "load 失败"
    assert llm.build(do_quantization=True, optimization_level=1,
                     quantized_dtype="w8a8", quantized_algorithm="normal",
                     target_platform="rk3588", num_npu_core=3,
                     extra_qparams=None, dataset=ds,
                     max_context=512) == 0, "build 失败"
    assert llm.export_rkllm(OUT) == 0, "export 失败"
    print("DONE:", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
