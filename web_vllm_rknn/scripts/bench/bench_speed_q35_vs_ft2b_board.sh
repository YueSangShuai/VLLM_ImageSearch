#!/bin/bash
# 【板上运行】Qwen3.5-0.8B vs Qwen3-VL-2B(ft) NPU 推理速度对比（2026-10-26）
# 方法：同一张告警图 + 同一中文描述 prompt；每个模型跑 max_new_tokens=32（≈prefill/固定开销）
#       与 384（decode 主导）两档；demo 自带 LLM 加载/视觉编码耗时打印；
#       decode 速率 = (T384 - T32) / (生成字数384 - 生成字数32)，CJK≈1.4字/token，两模型同口径可比。
# 用法: bash bench_speed_q35_vs_ft2b_board.sh   （建议经 launcher 后台跑，日志自落 /tmp/speed_bench.log）
set -uo pipefail
cd /home/web_vllm_rknn_q35
DEMO=/home/web_vllm_rknn/demo   # demo 二进制只在板上此路径（aarch64，勿假设在当前目录）
exec 9>/tmp/speed_bench.lock
flock -n 9 || { echo "another speed bench running" >&2; exit 1; }
# 停两个在线实例（q35 进程名被 comm 截断，必须 -f + 括号防自匹配）
pkill -x web_vllm_rknn 2>/dev/null
pkill -f "[.]/web_vllm_rknn_q35" 2>/dev/null
sleep 2
export LD_LIBRARY_PATH=/home/web_vllm_rknn/lib   # librknnrt/librkllmrt 用线上包里的（同一份）
IMG=/home/web_vllm_rknn/data/alerts_images/2026-10-15/1F\ Inside\ East\ Door/黑衣服男生/original/alert_1792020363414_363414.jpg
PROMPT='<image>请详细描述图中行人的外观，包括性别、大致年龄段、朝向、衣着款式与颜色、携带物品等清晰可见的属性。只描述行人本身清晰可见的特征，不得猜测看不见或无法确认的特征。用连贯的中文陈述句输出，句子之间用中文句号「。」分隔，只写确定的事实。'
FT_V=/home/web_vllm_rknn/models/qwen3vl-ft_vision_rk3588.rknn
FT_L=/home/web_vllm_rknn/models/qwen3vl-ft-merged_w8a8_rk3588.rkllm
Q_V=models/qwen3.5_vision_rk3588.rknn
Q_L=models/qwen3.5-0.8b_w8a8_rk3588.rkllm

run_one () {  # $1=name $2=vision $3=llm $4=max_tokens -> 打印 wall/加载/视觉/字数
    local name="$1" v="$2" l="$3" mt="$4" out s e wall
    out=$(mktemp)
    s=$(date +%s%N)
    printf '%s\nexit\n' "$PROMPT" | timeout 400 "$DEMO" "$IMG" "$v" "" "" "$l" "$mt" 2048 3 rk3588 > "$out" 2>&1
    e=$(date +%s%N); wall=$(( (e-s)/1000000 ))
    local load vis chars
    load=$(grep -ao 'LLM Model loaded in *[0-9.]* ms' "$out" | grep -o '[0-9.]*' | head -1)
    vis=$(grep -ao 'ImgEnc Model inference took *[0-9.]* ms' "$out" | grep -o '[0-9.]*' | head -1)
    chars=$(grep -ao 'robot:.*' "$out" | tail -1 | tr -d ' \n' | wc -m)
    echo "RESULT|$name|mt=$mt|wall=${wall}ms|load=${load:-NA}|vision=${vis:-NA}|chars=${chars}"
    rm -f "$out"
}

echo "======== speed bench $(date) img=$IMG ========"
for mt in 32 384; do
    for r in 1 2; do
        run_one "ft-2B"  "$FT_V" "$FT_L" "$mt"
        run_one "q35"    "$Q_V"  "$Q_L"  "$mt"
    done
done
echo "ALL_DONE"
# 拉起两个在线实例（setsid 自脱管；9>&- 防继承锁）
setsid nohup bash /home/web_vllm_rknn/run_web_rk.sh > /home/web_vllm_rknn/web.log 2>&1 < /dev/null 9>&- &
sleep 3
setsid nohup bash /home/web_vllm_rknn_q35/run_web_rk.sh > /home/web_vllm_rknn_q35/app.log 2>&1 < /dev/null 9>&- &
echo SERVICES_RESTART_SCHEDULED
