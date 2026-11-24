#!/usr/bin/env bash
# 生成 CUHK-PEDES 的 train.json / test.json
set -e
PY=${PY:-/home/yuesang/miniconda3/envs/llama/bin/python}   # transformers 5.8，支持 Qwen3-VL；可用 PY= 覆盖
cd "$(dirname "$0")/.."
"$PY" data/make_cuhk_json.py "$@"
