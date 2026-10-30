#!/bin/bash
# 停服务 → bench YOLO+BGE（NPU）→ 自动重启服务；结果落 bench_rk.log
set -uo pipefail
cd ~/output_cpp
pkill -x web_vllm_rknn || true   # 服务进程名是 ./web_vllm_rknn，-f 旧模式匹配不到
pkill -f "[w]eb_vllm_rknn_q35" 2>/dev/null   # comm 截断15字符，-x web_vllm_rknn_q35 永不匹配；-f 需括号防自匹配
sleep 2
bash run_bench_rk.sh --skip vlm --iters 8 --warmup 2 > bench_rk.log 2>&1
setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null &
echo DONE
