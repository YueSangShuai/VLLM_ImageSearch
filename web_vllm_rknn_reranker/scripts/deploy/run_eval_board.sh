#!/bin/bash
# 板上：跑 rerank_eval（需独占 NPU，先停其它 web_vllm 服务；测完跑 restart_5003_board.sh 恢复）
set -uo pipefail
cd ~/output_cpp_reranker
chmod +x web_vllm_rknn rerank_eval run_web_rk_reranker.sh

# 停掉占用 NPU 的其它实例（scoped：不匹配本项目）
pkill -f '/home/pi/output_cpp/web_vllm_rknn' 2>/dev/null || true
pkill -f '/home/Desktop/output_cpp/web_vllm_rknn' 2>/dev/null || true
sleep 3

export LD_LIBRARY_PATH="$PWD/lib"
export RERANKER_VISION_PATH="$PWD/models/qwen3vl-reranker_vision_rk3588.rknn"
export RERANKER_RKLLM_PATH="$PWD/models/qwen3vl-reranker-2b_w8a8_rk3588.rkllm"
export RERANKER_WEIGHT_BIN="$PWD/rerank_yes_no_diff.bin"
export WEB_VLLM_MODELS_DIR="$PWD/models"
export RKLLM_NPU_CORE_NUM=3

MODE="${1:-logits}"
export RERANKER_MODE="$MODE"
echo "== rerank_eval mode=$MODE RERANKER_IMG_MID=${RERANKER_IMG_MID:-0}"
./rerank_eval --list calib/host_scores.json --out "board_scores_${MODE}.json" 2>&1 | tail -15
