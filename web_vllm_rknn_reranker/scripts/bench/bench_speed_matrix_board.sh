#!/bin/bash
# 【板上运行】NPU 速度定量矩阵（2026-10-26 v2）：Qwen3.5-0.8B vs Qwen3-VL-2B-ft
# A) decode 拟合：mt ∈ {16,32,48,64}（均 < 两模型自然 EOS 长度 ~59/74 tok，token 数精确=mt）
#    每档 2 遍；wall(mt) 线性拟合 → 斜率=decode ms/tok，截距=prefill+视觉+REPL 固定开销
# B) 常驻斜率：同一 demo 进程连喂 k=1 / k=3 个相同 prompt → 单帧成本=(T3-T1)/2（免加载）
# 测完自动拉起 5003/5004 两个在线实例。日志建议重定向 /tmp/speed_matrix.log
set -uo pipefail
cd /home/web_vllm_rknn_q35
DEMO=/home/web_vllm_rknn/demo   # 必须绝对路径（demo 只在线上目录）
exec 9>/tmp/speed_matrix.lock
flock -n 9 || { echo "another matrix bench running" >&2; exit 1; }
pkill -x web_vllm_rknn 2>/dev/null
pkill -f "[.]/web_vllm_rknn_q35" 2>/dev/null
sleep 2
export LD_LIBRARY_PATH=/home/web_vllm_rknn/lib
IMG=/home/web_vllm_rknn/data/alerts_images/2026-10-15/1F\ Inside\ East\ Door/黑衣服男生/original/alert_1792020363414_363414.jpg
PROMPT='<image>请详细描述图中行人的外观，包括性别、大致年龄段、朝向、衣着款式与颜色、携带物品等清晰可见的属性。只描述行人本身清晰可见的特征，不得猜测看不见或无法确认的特征。用连贯的中文陈述句输出，句子之间用中文句号「。」分隔，只写确定的事实。'
FT_V=/home/web_vllm_rknn/models/qwen3vl-ft_vision_rk3588.rknn
FT_L=/home/web_vllm_rknn/models/qwen3vl-ft-merged_w8a8_rk3588.rkllm
Q_V=models/qwen3.5_vision_rk3588.rknn
Q_L=models/qwen3.5-0.8b_w8a8_rk3588.rkllm

one_shot () {  # $1 name $2 vision $3 llm $4 max_tokens
    local out s e
    out=$(mktemp)
    s=$(date +%s%N)
    printf '%s\nexit\n' "$PROMPT" | timeout 400 "$DEMO" "$IMG" "$2" "" "" "$3" "$4" 2048 3 rk3588 > "$out" 2>&1
    e=$(date +%s%N)
    echo "ONE|$1|mt=$4|wall=$(( (e-s)/1000000 ))|chars=$(grep -ao 'robot:.*' "$out" | tail -1 | tr -d ' \n' | wc -m)|$(grep -ao 'ImgEnc Model inference took *[0-9.]* ms' "$out" | grep -o '[0-9.]*' | head -1)"
    rm -f "$out"
}
resident () {  # $1 name $2 vision $3 llm $4 prompt次数
    local s e i
    s=$(date +%s%N)
    { for ((i=0;i<$4;i++)); do printf '%s\n' "$PROMPT"; done; echo exit; } \
      | timeout 600 "$DEMO" "$IMG" "$2" "" "" "$3" 64 2048 3 rk3588 > /dev/null 2>&1
    e=$(date +%s%N)
    echo "RES|resident_$1|k=$4|wall=$(( (e-s)/1000000 ))"
}

echo "======== speed matrix $(date) ========"
for mt in 16 32 48 64; do
    for r in 1 2; do
        one_shot "ft-2B" "$FT_V" "$FT_L" "$mt"
        one_shot "q35"   "$Q_V"  "$Q_L"  "$mt"
    done
done
for k in 1 3; do
    resident "ft2B" "$FT_V" "$FT_L" "$k"
    resident "q35"  "$Q_V"  "$Q_L"  "$k"
done
echo "ALL_DONE"
setsid nohup bash /home/web_vllm_rknn/run_web_rk.sh > /home/web_vllm_rknn/web.log 2>&1 < /dev/null 9>&- &
sleep 3
setsid nohup bash /home/web_vllm_rknn_q35/run_web_rk.sh > /home/web_vllm_rknn_q35/app.log 2>&1 < /dev/null 9>&- &
echo SERVICES_RESTART_SCHEDULED
