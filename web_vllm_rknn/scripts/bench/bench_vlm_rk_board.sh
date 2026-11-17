#!/bin/bash
# NPU 版 VLM（RKLLM w8a8）补测：与 bench_vlm_cpu_board.sh 完全同参数，跑完自动恢复服务
set -uo pipefail
cd ~/output_cpp
exec 9>/tmp/bench_vlm_cpu.lock
flock -n 9 || { echo "another bench already running" >&2; exit 1; }
pkill -x web_vllm_rknn || true
sleep 2
bash run_bench_rk.sh --skip yolo,bge --iters 2 --warmup 1 --vlm-max-tokens 120 > bench_vlm_rk.log 2>&1
echo "[$(date)] bench rc=$?" >> bench_vlm_rk_wrap.log
setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null 9>&- &
echo "[$(date)] service restarted" >> bench_vlm_rk_wrap.log
