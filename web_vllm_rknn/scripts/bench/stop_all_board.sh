#!/bin/bash
# 板端清理：停在线服务 + 任何残留 bench（进程名精确匹配，避免误伤 ssh 会话本身）
pkill -x web_vllm_rknn || true
pkill -x bench_models_cpu || true
pkill -x bench_models || true
pkill -f 'bash bench_vlm_cpu_board.sh' || true
pkill -f 'bash bench_rk_board.sh' || true
pkill -f 'bash bench_cpu_board.sh' || true
sleep 2
pgrep -ax web_vllm_rknn || echo SERVICE_STOPPED
pgrep -x bench_models_cpu || echo BENCH_STOPPED
