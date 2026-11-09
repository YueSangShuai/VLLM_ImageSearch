#!/bin/bash
# 板上执行：向所有任务的 tasks.json 注入一条「低阈值」英文 e2e 规则（近必中，用于打通
# 检测→嵌入→匹配→告警→落盘→web 展示全链路），备份原文件后重启服务。
set -euo pipefail
DIR="${1:-/home/web_siglip_reid_rknn}"
cd "$DIR"
cp -n data/tasks.json data/tasks.json.bak_e2e 2>/dev/null || true

python3 - <<'EOF'
import json
p = "data/tasks.json"
tasks = json.load(open(p))
# 兜底：任务没绑设备时挂到 devices.json 第一台（否则规则永远不会加载进 matcher）
try:
    dev0 = json.load(open("data/devices.json"))[0]["id"]
    for t in tasks:
        if not t.get("device_id"):
            t["device_id"] = dev0
except Exception as e:
    print("device bind skipped:", e)
rule = {
    "id": "rule_e2e_test",
    "name": "E2E烟雾测试(任何人)",
    "text": "person, standing, visible body",
    "positive": ["person, standing, visible body"],
    "negative": [],
    "threshold": 0.005,   # SigLIP 尺度下近似必中
    "score_diff": 0,
    "interval": 30,
    "enabled": True,
}
for t in tasks:
    rules = t.setdefault("rules", [])
    rules[:] = [r for r in rules if r.get("id") != "rule_e2e_test"]
    rules.append(rule)
json.dump(tasks, open(p, "w"), ensure_ascii=False, indent=2)
print("injected into", len(tasks), "tasks")
EOF

pkill -x web_siglip_reid || true
sleep 3
env -i HOME=/home/pi PATH=/usr/bin:/bin bash -c "cd $DIR && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &"
echo RESTARTED-with-e2e-rule
