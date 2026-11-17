#!/bin/bash
# 板端 W8A8 批量生成全流程（w8a8_ab）：推资产 → 停服务 → ab_dump → 轮询 → 拉结果 → 重启服务。
# 一次执行；日志即时输出。预计 ~25 分钟。
set -uo pipefail
R=/data/yuesang/LLM/FG-CLIP/web_vllm_rknn
S=$R/scripts/eval/w8a8_ab
SSH="python3 $R/scripts/deploy/ssh_pi.py"
SCP="python3 $R/scripts/deploy/scp_pi.py"
BD=/home/web_vllm_rknn
AB=$BD/abset

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 0. 规则快照（compare 用）----
timeout 60 $SCP pull $BD/data/tasks.json --dest $S/board_rules/ >/dev/null 2>&1 && say "tasks.json pulled" || say "WARN tasks.json pull failed"

# ---- 1. 推资产 ----
tar czf /tmp/abset.tar.gz -C $S images prompts.json
timeout 120 $SSH 30 "mkdir -p $AB" >/dev/null
timeout 300 $SCP push /tmp/abset.tar.gz --dest $BD/ >/dev/null 2>&1 && say "abset.tar.gz pushed" || { say "FAIL tar push"; exit 1; }
timeout 120 $SSH 60 "cd $BD && tar xzf abset.tar.gz -C abset && ls abset/images | wc -l" | tail -1
timeout 300 $SCP push $R/cpp/build/aarch64/ab_dump --dest $BD/ >/dev/null 2>&1 && say "ab_dump pushed" || { say "FAIL ab_dump push"; exit 1; }
timeout 120 $SSH 30 "chmod +x $BD/ab_dump; md5sum $BD/ab_dump" | tail -1
say "host md5: $(md5sum $R/cpp/build/aarch64/ab_dump | cut -d' ' -f1)"

# ---- 2. 停服务（与启动必须分开两条 ssh 命令，pkill 陷阱）----
timeout 120 $SSH 30 "pkill -x web_vllm_rknn || true; pkill -f '[w]eb_vllm_rknn_q3' || true; sleep 3" >/dev/null
timeout 120 $SSH 20 "pgrep -a -f web_vllm_rknn | grep -v ssh || echo NO_SERVICE" | tail -2

# ---- 3. 启动批量生成（nohup，日志 abset/run.log）----
RUNENV="cd $BD && export LD_LIBRARY_PATH=$BD/lib VLM_MODEL_NAME=$BD/models/zkteco-person-describe-ft-vl-2b-rk3588.rknn VLM_MMPROJ_PATH=$BD/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm VLM_ATTR_HEAD_PATH=$BD/models/attr_head.bin RKLLM_NPU_CORE_NUM=3 && rm -f abset/w8a8.jsonl abset/run.log && setsid nohup ./ab_dump --images abset/images --out abset/w8a8.jsonl --prompts abset/prompts.json > abset/run.log 2>&1 < /dev/null & echo LAUNCHED"
timeout 60 $SSH 30 "$RUNENV" >/dev/null
sleep 20
timeout 120 $SSH 25 "pgrep -a ab_dump | head -1; tail -3 abset/run.log" || say "check #1 failed (wrapper 假失败常见)"

# ---- 4. 轮询（100 行 = 50 图 x 2 prompt）----
for i in $(seq 1 150); do
  sleep 20
  N=$(timeout 60 $SSH 20 "wc -l < $AB/w8a8.jsonl 2>/dev/null" | grep -oE '[0-9]+' | tail -1)
  say "progress: ${N:-?}/100"
  if [ "${N:-0}" -ge 100 ]; then break; fi
done
timeout 60 $SSH 20 "tail -2 $AB/run.log" | tail -2

# ---- 5. 收结果 ----
timeout 300 $SCP pull $AB/w8a8.jsonl --dest $S/ >/dev/null 2>&1 && say "w8a8.jsonl pulled: $(wc -l < $S/w8a8.jsonl 2>/dev/null) lines" || say "FAIL jsonl pull"

# ---- 6. 重启线上服务 ----
timeout 60 $SSH 30 "cd $BD && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null & echo RESTART_CMD_DONE" >/dev/null
sleep 25
timeout 120 $SSH 20 "pgrep -x web_vllm_rknn && grep -c 'VlmDescriber(RKLLM) ready' $BD/web.log || true" | tail -2
say "DONE"
