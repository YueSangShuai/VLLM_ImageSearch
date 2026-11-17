#!/bin/bash
# 清理板端 w8a8_ab 测试产物：abset/（图+prompts+结果+日志）、abset.tar.gz、ab_dump 工具。
# 先盘点再删，删后验证；只动本次测试放入的文件，服务/模型/data 一概不碰。
set -uo pipefail
R=/data/yuesang/LLM/FG-CLIP/web_vllm_rknn
SSH="python3 $R/scripts/deploy/ssh_pi.py"
BD=/home/web_vllm_rknn

echo "== before =="
timeout 60 $SSH 30 "cd $BD && du -sh abset abset.tar.gz ab_dump 2>/dev/null; pgrep -a ab_dump | head -2 || true"

echo "== delete =="
timeout 60 $SSH 30 "rm -rf $BD/abset $BD/abset.tar.gz $BD/ab_dump; echo RM_DONE"

echo "== verify =="
timeout 60 $SSH 30 "cd $BD && ls -d abset abset.tar.gz ab_dump 2>&1 | head -3; df -h /home | tail -1; pgrep -x web_vllm_rknn && echo SERVICE_OK"
