#!/bin/bash
# 同机 CPU 对照：web_vllm_cpp 的 aarch64 bench（ORT-CPU YOLO+BGE），与 RKNN bench 同条件
set -uo pipefail
cd ~/output_cpp
pkill -x web_vllm_rknn || true   # 服务进程名是 ./web_vllm_rknn，-f 旧模式匹配不到
sleep 2
export LD_LIBRARY_PATH=$PWD/lib:$HOME/wvc/lib
export DETECTOR_ONNX_PATH=$PWD/models/onnx/yolov8n-dyn.onnx
export EMBEDDING_ONNX_PATH=$PWD/models/onnx/bge-small-zh.onnx
export EMBEDDING_VOCAB_PATH=$PWD/models/vocab.txt
./bench_models_cpu --skip vlm --iters 8 --warmup 2 > bench_cpu.log 2>&1
setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null &
echo DONE
