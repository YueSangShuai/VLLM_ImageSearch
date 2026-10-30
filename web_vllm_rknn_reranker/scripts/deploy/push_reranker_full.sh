#!/bin/bash
# 整包推送 output_cpp -> 板子 /home/pi/output_cpp_reranker（与 5003 线上包互不干扰）。
# tar -h 解引用包内软链（reranker 双件套 / YOLO / 权重 bin）。交互输一次板子密码（pi）。
# 传完板上启动：cd ~/output_cpp_reranker && bash run_web_rk_reranker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOARD="${BOARD:-pi@10.8.18.118}"
DEST="${1:-/home/pi}"

echo "== streaming $(du -shL output_cpp | cut -f1) to $BOARD:$DEST (输入板子密码)"
tar cf - -h output_cpp | ssh $SSH_OPTS $BOARD "mkdir -p $DEST && tar xf - -C $DEST"

DIR="$DEST/$(basename output_cpp)"
echo "== verify on board"
ssh $SSH_OPTS $BOARD "cd $DIR && ls -la models/ && du -sh models/* 2>/dev/null; pgrep -x web_vllm_rknn >/dev/null && echo 'NOTE: 有同名服务在跑旧代码，注意 NPU 互斥' || true"
echo "DONE. 板上启动: cd $DIR && bash run_web_rk_reranker.sh"
