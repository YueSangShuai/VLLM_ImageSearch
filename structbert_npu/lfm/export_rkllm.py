#!/usr/bin/env python
"""LFM2.5-VL 语言塔（提取后=Lfm2ForCausalLM, 1.17B）→ RKLLM w8a8 (rk3588, 3 核)。

配方对齐 gecko/export_rkllm.py：
- CWD 必须在 ext4（/tmp/rkllm_run_lfm）。
- 校准集 = LFM 宿主 bf16 基线实际输出 + 与 gecko 同款线上描述/TAG 句（对齐使用分布）。
- max_context=1024：LFM 单图 ≤256 image token + 短 prompt + 128 生成。
用法：mkdir -p /tmp/rkllm_run_lfm && cd /tmp/rkllm_run_lfm && <rkllm-env-py> export_rkllm.py
"""
import json
import os

from rkllm.api import RKLLM

MODEL = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/LFM2.5-VL-1.6B-lm-lfm2"
BASELINE = "/data/yuesang/LLM/FG-CLIP/structbert_npu/lfm/baseline_lfm_bf16.jsonl"
OUT = "/data/yuesang/LLM/FG-CLIP/structbert_npu/lfm/lfm25vl-1.6b_w8a8_rk3588.rkllm"

EXTRA = [
    "一名男性，穿着黑色上衣和深色裤子，背着一个黑色背包。",
    "一名女性，穿白色连衣裙，戴眼镜。",
    "画面中一名人员靠近车辆，身体前倾，疑似正在观察车窗或车门区域，存在可疑徘徊行为。"
    "该人员穿着深色上衣和长裤。<|TAG|>中风险",
    "该人员未戴帽子，未戴口罩，未戴眼镜，身穿黑色圆领上衣和灰色长款外套，"
    "下穿浅蓝色牛仔裤。<|TAG|>低风险",
    "该人员正手持手机进行通话，疑似查看信息或发送消息，注意防盗。"
    "该人员头戴黑色棒球帽，身穿浅粉色开衫和深灰色百褶裙。<|TAG|>中风险",
    "画面中人员无异常行为，正常路过。<|TAG|>低风险",
    "行为描述：该人员正在行走。穿着描述：穿着黑色上衣、灰色开衫、浅色牛仔裤。"
    "风险等级：低。",
]


def main():
    calib = []
    with open(BASELINE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("text"):
                calib.append({"input": rec["text"], "target": ""})
    calib += [{"input": s, "target": ""} for s in EXTRA]
    ds = "/tmp/rkllm_run_lfm/calib.json"
    os.makedirs("/tmp/rkllm_run_lfm", exist_ok=True)
    assert os.path.basename(os.getcwd()) == "rkllm_run_lfm", \
        "必须 cd /tmp/rkllm_run_lfm 后运行（ext4 卷）"
    json.dump(calib, open(ds, "w"), ensure_ascii=False, indent=1)
    print("calib samples:", len(calib))

    llm = RKLLM()
    assert llm.load_huggingface(model=MODEL, device="cpu") == 0, "load 失败"
    assert llm.build(do_quantization=True, optimization_level=1,
                     quantized_dtype="w8a8", quantized_algorithm="normal",
                     target_platform="rk3588", num_npu_core=3,
                     extra_qparams=None, dataset=ds,
                     max_context=1024) == 0, "build 失败"
    assert llm.export_rkllm(OUT) == 0, "export 失败"
    print("DONE:", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
