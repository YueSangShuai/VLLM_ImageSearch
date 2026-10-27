#!/bin/bash
# 复原 tasks.json 并重启服务（e2e_add_rule.sh 的逆操作）
set -euo pipefail
cd ~/output_cpp
[ -f data/tasks.json.bak_e2e ] || { echo "no backup"; exit 1; }
mv data/tasks.json.bak_e2e data/tasks.json
pkill -f 'output_cpp/web_vllm_rknn' || true
sleep 3
setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null &
echo restored-and-restarted
