#!/bin/bash
# 板上运行：四个 StructBERT RKNN 变体的前向+速度，输出 npy 待拉回。
cd "$(dirname "$0")"
LIB=/home/web_vllm_rknn/lib/librknnrt.so
for m in "sbert_tower_fp16.rknn tower tower_fp16" \
         "sbert_tower_int8.rknn tower tower_int8" \
         "sbert_pair_fp16.rknn pair pair_fp16" \
         "sbert_pair_int8.rknn pair pair_int8"; do
    set -- $m
    echo "===== $1 ====="
    python3 board_eval_structbert.py "$1" "$2" sbert_test "$3.npy" "$LIB" 30 2>&1 | tail -2
done
