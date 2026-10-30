#!/bin/bash
# 撤掉 E2E 临时规则并重启
set -euo pipefail
cd ~/output_cpp_reranker
python3 - <<'EOF'
import json
tasks = json.load(open('data/tasks.json'))
for t in tasks:
    t['rules'] = [r for r in t.get('rules', []) if r.get('id') != 'rule_e2e_black']
json.dump(tasks, open('data/tasks.json', 'w'), ensure_ascii=False, indent=2)
print('removed')
EOF
pkill -f 'output_cpp_reranker/web_vllm_rknn' || true
sleep 3
setsid nohup bash run_web_rk_reranker.sh > web.log 2>&1 < /dev/null &
echo RESTARTED
