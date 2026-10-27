#!/bin/bash
# 告警链路端到端冒烟：临时规则注入 + 服务重启（测完跑 e2e_rm_rule.sh 复原）
set -euo pipefail
cd ~/output_cpp
cp -n data/tasks.json data/tasks.json.bak_e2e || true
python3 - <<'EOF'
import json
p = 'data/tasks.json'
t = json.load(open(p))
rule = {"id": "rule_e2e_test", "name": "E2E冒烟", "text": "",
        "positive": ["一个人在画面中"], "negative": [],
        "threshold": 0.01, "score_diff": 0, "interval": 5, "enabled": True}
for task in t:
    rs = task.setdefault('rules', [])
    if not any(r.get('id') == 'rule_e2e_test' for r in rs):
        rs.append(rule)
json.dump(t, open(p, 'w'), ensure_ascii=False, indent=2)
print('injected into', len(t), 'tasks')
EOF
pkill -f 'output_cpp/web_vllm_rknn' || true
sleep 3
setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null &
echo restarted
