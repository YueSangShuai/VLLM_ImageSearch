#!/bin/bash
# host bf16 两条基线（结构化 + 线上描述）串行跑。GPU 与 conda 环境经环境变量可覆盖。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-/home/yuesang/miniconda3/envs/llama/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 基座权重（16-bit 参照）在 HF 缓存；/data/yuesang/LLM/Qwen3-VL-2B-Instruct 已被清理
MODEL=${MODEL:-/home/yuesang/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203}
ARGS=(--model "$MODEL" "$@")
$PY "$HERE/run_host_bf16.py" "${ARGS[@]}" || echo "STRUCTURED FAILED"
$PY "$HERE/run_host_bf16_desc.py" "${ARGS[@]}" || echo "DESC FAILED"
