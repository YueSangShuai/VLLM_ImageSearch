#!/bin/bash
# CPU 版 VLM（llama.cpp f16 GGUF）板端基线：与 NPU bench 同参数（120 token / 256x256 合成图）。
# CPU f16 极慢，只测 iters=2 warmup=1；后台 nohup 跑，结果落 bench_vlm_cpu.log
set -uo pipefail
cd ~/output_cpp
# 单实例锁，防止重复启动互相挤内存
exec 9>/tmp/bench_vlm_cpu.lock
flock -n 9 || { echo "another bench already running" >&2; exit 1; }
pkill -x web_vllm_rknn || true   # 服务进程名是 ./web_vllm_rknn，必须用 -x 精确匹配
pkill -f "[w]eb_vllm_rknn_q35" 2>/dev/null   # comm 截断15字符，-x web_vllm_rknn_q35 永不匹配；-f 需括号防自匹配
sleep 2
export LD_LIBRARY_PATH=$PWD/lib:$HOME/wvc/lib
export DETECTOR_ONNX_PATH=$PWD/models/onnx/yolov8n-dyn.onnx
export EMBEDDING_ONNX_PATH=$PWD/models/onnx/bge-small-zh.onnx
export EMBEDDING_VOCAB_PATH=$PWD/models/vocab.txt
export VLM_MODEL_NAME=$HOME/gguf/qwen3vl-ft-f16.gguf
export VLM_MMPROJ_PATH=$HOME/gguf/mmproj-qwen3vl-ft-f16.gguf
export VLM_DEVICE=cpu
export VLM_GGUF_N_GPU_LAYERS=0
export VLM_MAX_NEW_TOKENS=120
echo "[$(date)] bench start" >> bench_vlm_cpu_wrap.log
./bench_models_cpu --skip yolo,bge --iters 2 --warmup 1 --vlm-max-tokens 120 > bench_vlm_cpu.log 2>&1
echo "[$(date)] bench rc=$?" >> bench_vlm_cpu_wrap.log
setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null 9>&- &
echo "[$(date)] service restarted" >> bench_vlm_cpu_wrap.log
