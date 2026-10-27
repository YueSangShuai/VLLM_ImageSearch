#!/bin/bash
# E2E 验证后收尾：停服务 → 复原 tasks.json → bench NPU YOLO 基线 → 清测试告警 → 重启服务
set -uo pipefail
cd ~/output_cpp

pkill -f 'output_cpp/web_vllm_rknn' || true
sleep 3

if [ -f data/tasks.json.bak_e2e ]; then
  mv data/tasks.json.bak_e2e data/tasks.json && echo "tasks.json restored"
fi

echo "== bench (yolo only, NPU) =="
LD_LIBRARY_PATH=$PWD/lib ./bench_models --skip bge,vlm --iters 5 --warmup 2 2>&1 | tail -8

echo "== 清理 E2E 测试告警 =="
rm -rf "data/alerts_images/$(date +%F)/一楼电梯厅/E2E冒烟"
rm -rf "data/alerts_images/$(date +%F)/一楼电梯厅/E2E冒烟" 2>/dev/null
for d in data/alerts_images/*/一楼电梯厅/E2E冒烟; do [ -d "$d" ] && rm -rf "$d" && echo "cleaned $d"; done
python3 - <<'EOF'
import json, glob
for p in glob.glob('data/alerts_history/alerts_*.json'):
    try:
        a = json.load(open(p))
    except Exception:
        continue
    kept = [x for x in a if x.get('rule_id') != 'rule_e2e_test']
    if len(kept) != len(a):
        json.dump(kept, open(p, 'w'), ensure_ascii=False, indent=2)
        print('purged', len(a)-len(kept), 'test alerts from', p)
# detection_log 同步清
try:
    a = json.load(open('data/detection_log.json'))
    kept = [x for x in a if x.get('rule_id') != 'rule_e2e_test']
    if len(kept) != len(a):
        json.dump(kept, open('data/detection_log.json','w'), ensure_ascii=False, indent=2)
        print('purged', len(a)-len(kept), 'from detection_log')
except FileNotFoundError:
    pass
EOF

setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null &
echo "service restarted"
