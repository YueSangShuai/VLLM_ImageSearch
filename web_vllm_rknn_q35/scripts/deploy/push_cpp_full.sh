#!/bin/bash
# 整包推送 output_cpp/ 到板上 /home/web_vllm_rknn_q35（Qwen3.5-0.8B 版平台，与线上 5003 并行）。
# tar -h 解引用软链（本地若用软链省空间）；板上拿到真实文件。权重不可压缩，不用 gzip。
# 用法: bash scripts/deploy/push_cpp_full.sh   （密码 pi；首推前确认板上 df -h 有 ~2G 余量）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOARD="pi@10.8.18.118"
DEST_BASE="${1:-/home}"
DIR="$DEST_BASE/web_vllm_rknn_q35"

ssh $SSH_OPTS $BOARD "mkdir -p $DIR"
echo "== streaming $(du -shL output_cpp | cut -f1) to $BOARD:$DIR (输入密码 pi)"
tar cf - -h output_cpp | ssh $SSH_OPTS $BOARD "tar xf - -C $DIR"

echo "== verify on board"
ssh $SSH_OPTS $BOARD "cd $DIR && ls models/ | head && pgrep -x web_vllm_rknn_q35 >/dev/null && echo 'NOTE: q35 实例仍在跑旧包，重启之' || true"
echo "DONE. 板上启动: cd $DIR && bash run_web_rk.sh   (端口 5004；与线上实例并行，勿 pkill -x web_vllm_rknn 误伤)"
