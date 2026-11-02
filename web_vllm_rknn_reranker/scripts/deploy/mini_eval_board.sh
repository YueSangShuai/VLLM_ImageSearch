#!/bin/bash
# 5 条子集诊断跑（后台化由外层 ssh 负责；日志 /tmp/mini.log）
cd ~/output_cpp_reranker
export LD_LIBRARY_PATH=$PWD/lib
export RERANKER_VISION_PATH=$PWD/models/qwen3vl-reranker_vision_rk3588.rknn
export RERANKER_RKLLM_PATH=$PWD/models/qwen3vl-reranker-2b_w8a8_rk3588.rkllm
export RERANKER_WEIGHT_BIN=$PWD/rerank_yes_no_diff.bin
exec ./rerank_eval --list calib/host_scores_mini.json --out /tmp/mini.json
