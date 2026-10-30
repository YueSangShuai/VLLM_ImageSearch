#!/bin/bash
# Reranker E2E 冒烟：确保有 task+设备，注入低阈值临时规则，重启 5005 服务。
# 在板上跑：bash e2e_reranker_add_rule.sh   （测完跑 e2e_reranker_rm_rule.sh 复原）
# 阈值 0.42：reranker 分数带（host 参考 pos 0.3~0.68 / cross 0.2~0.6），先低后精调
set -euo pipefail
cd ~/output_cpp_reranker
cp -n data/tasks.json data/tasks.json.bak_e2e || true
python3 - <<'EOF'
import json, os
devs = json.load(open('data/devices.json'))
assert devs, 'devices.json 为空：先通过 5005 页面或 API 添加设备'
d = devs[0]
tasks = json.load(open('data/tasks.json'))
if not tasks:
    tasks = [{"id": "task_rerank_e2e", "name": "Reranker冒烟任务",
              "device_id": d["id"], "device_name": d.get("name", ""),
              "device": {k: d.get(k) for k in ("ip", "port", "username", "password", "stream_type")},
              "rules": []}]
rule = {"id": "rule_e2e_black", "name": "黑衣服男生", "text": "",
        "positive": ["一名穿黑色上衣的男人"], "negative": [],
        "threshold": 0.42, "score_diff": 0.0, "interval": 30, "enabled": True}
for task in tasks:
    rs = task.setdefault('rules', [])
    if not any(r.get('id') == 'rule_e2e_black' for r in rs):
        rs.append(rule)
json.dump(tasks, open('data/tasks.json', 'w'), ensure_ascii=False, indent=2)
print('tasks:', len(tasks), 'device:', d.get('name'))
EOF
# scoped pkill：勿用 pkill -x web_vllm_rknn（会连 5003 线上服务一起杀）
pkill -f 'output_cpp_reranker/web_vllm_rknn' || true
sleep 3
setsid nohup bash run_web_rk_reranker.sh > web.log 2>&1 < /dev/null &
echo RESTARTED port=5005
