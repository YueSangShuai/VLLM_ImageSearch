#!/bin/bash
# 板上执行：移除 e2e 烟雾规则（恢复备份）、清理 rule_e2e_test 告警/日志/图片、重启服务。
set -euo pipefail
DIR="${1:-/home/web_siglip_reid_rknn}"
cd "$DIR"

pkill -x web_siglip_reid || true
sleep 2
[ -e data/tasks.json.bak_e2e ] && mv data/tasks.json.bak_e2e data/tasks.json && echo "tasks.json restored"

python3 - <<'EOF'
import json, os, shutil, glob
def drop(path):
    if not os.path.exists(path): return
    try: data = json.load(open(path))
    except Exception: return
    kept = [x for x in data if x.get("rule_id") != "rule_e2e_test" and x.get("rule_name") != "E2E烟雾测试(任何人)"]
    if len(kept) != len(data):
        json.dump(kept, open(path, "w"), ensure_ascii=False, indent=2)
        print("purged", path, len(data) - len(kept), "entries")
drop("data/detection_log.json")
drop("data/debug_log.json")
for d in glob.glob("data/alerts_history/*/"):
    drop(d + "alerts.json") if os.path.exists(d + "alerts.json") else None
for pat in ("data/alerts_images/*/*/E2E*", "data/debug_images"):
    for p in glob.glob(pat):
        shutil.rmtree(p, ignore_errors=True); print("removed", p)
EOF

env -i HOME=/home/pi PATH=/usr/bin:/bin bash -c "cd $DIR && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &"
echo RESTARTED-clean
