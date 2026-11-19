#!/bin/bash
# 板端速度探查：1) 基础状态 2) SigLIP 文本塔 CPU/ORT 实测（bench_models 或独立计时）
set -uo pipefail
PY=/home/yuesang/miniconda3/envs/rknn/bin/python
S=/tmp/ssh_board.py
run() { echo "===== $1"; timeout 300 "$PY" "$S" "$1" 2>&1 | tail -${2:-40}; }

run "uname -m && head -1 /proc/meminfo && df -h / | tail -1"
run "pgrep -fa 'web_siglip|web_vllm|bench' | grep -v pgrep | head -5"
run "ls ~/output/ 2>/dev/null && ls ~/output/models/onnx/ 2>/dev/null"
run "nproc && grep -c processer /proc/cpuinfo"
run "ls ~/output/*.sh ~/output/bench_models 2>/dev/null; ~/output/bench_models --help 2>&1 | head -15"
