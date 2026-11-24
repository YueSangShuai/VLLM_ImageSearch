#!/usr/bin/env bash
# 训练 ProjMLP（吃缓存矩阵，单卡即可）
set -e
PY=${PY:-/home/yuesang/miniconda3/envs/llama/bin/python}   # transformers 5.8，支持 Qwen3-VL；可用 PY= 覆盖
cd "$(dirname "$0")/.."
"$PY" train.py --name full "$@"
