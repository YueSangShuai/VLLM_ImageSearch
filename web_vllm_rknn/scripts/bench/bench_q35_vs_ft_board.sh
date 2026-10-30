#!/bin/bash
# Qwen3.5-0.8B vs qwen3vl-ft-2B 板端 demo 对比
# ⚠️ demo 是交互 REPL：必须喂 "<image>提示词\nexit\n" 到 stdin，否则 EOF 空 prompt 死循环
# 2026-10-26：部署已迁至 /home/web_vllm_rknn（旧 ~/output_rk3588、~/output_cpp 均不在）；
#            prompt 换为线上同款中文描述 prompt（与生产输出可比）
# 用法: bash bench_q35_vs_ft_board.sh [image.jpg] [单轮超时秒=600]
set -uo pipefail
HERE=/home/web_vllm_rknn
cd "$HERE"
exec 9>/tmp/bench_q35.lock
flock -n 9 || { echo "another bench running" >&2; exit 1; }
pkill -x web_vllm_rknn || true
pkill -f "[w]eb_vllm_rknn_q35" 2>/dev/null   # comm 截断15字符，-x web_vllm_rknn_q35 永不匹配；-f 需括号防自匹配
sleep 2
export LD_LIBRARY_PATH=./lib
IMG="${1:-}"
[ -z "$IMG" ] && IMG=$(find "$HERE/data/alerts_images" -name '*.jpg' -path '*/original/*' | sort | tail -1)
TO="${2:-600}"
PROMPT='<image>请详细描述图中行人的外观，包括性别、大致年龄段、朝向、衣着款式与颜色、携带物品等清晰可见的属性。只描述行人本身清晰可见的特征，不得猜测看不见或无法确认的特征。用连贯的中文陈述句输出，句子之间用中文句号「。」分隔，只写确定的事实，不要输出英文，不要输出多余的解释。'
run_one () {  # $1=name $2=vision $3=llm
    echo "---- $1 ----"
    local start end rc out
    out=$(mktemp)
    start=$(date +%s%N)
    printf '%s\nexit\n' "$PROMPT" | timeout "$TO" ./demo "$IMG" "$2" "" "" "$3" 512 2048 3 rk3588 > "$out" 2>&1
    rc=$?
    end=$(date +%s%N)
    # demo 的 robot: 回复与无换行的 "user: " REPL 提示同行，sed '^robot:' 匹配不到，先拆行再取
    tr -d '\r' < "$out" | grep -ao 'robot:.*' | grep -av '^robot: *$' | head -5
    [ $rc -eq 124 ] && echo "[超时 ${TO}s 未完成 → kill]" || echo "[耗时 $(( (end-start)/1000000 )) ms]"
    rm -f "$out"
}
echo "======== [$IMG] $(date) ========"
run_one "qwen3vl-ft-2B (线上在用)" models/qwen3vl-ft_vision_rk3588.rknn models/qwen3vl-ft-merged_w8a8_rk3588.rkllm
run_one "Qwen3.5-0.8B (w8a8)"    models/qwen3.5_vision_rk3588.rknn     models/qwen3.5-0.8b_w8a8_rk3588.rkllm
# 9>&- 必须：否则服务进程继承 flock 的 fd9，锁被服务永久持有，下次 bench 直接 "another bench running"
setsid nohup bash "$HERE/run_web_rk.sh" > "$HERE/web.log" 2>&1 < /dev/null 9>&- &
echo DONE_SERVICE_RESTARTED
