#!/bin/bash
# 结构化精度 A/B（demo REPL 版）：ft-2B W8A8 vs 线上 q35 基座，同一批告警 ROI 图、
# 同一结构化 prompt（与 run_web_rk.sh VLM_STRUCTURED_PROMPT 同属性表）。
# ⚠️ 独占 NPU：跑前停服务，跑完自动拉起。
set -uo pipefail
cd /home/Desktop/output_cpp
export LD_LIBRARY_PATH="$PWD/lib"
exec 9>/tmp/bench_q35.lock
flock -n 9 || { echo "another bench running" >&2; exit 1; }
pkill -f "[w]eb_vllm_rknn_q35" || true   # comm 截断15字符：-x web_vllm_rknn_q35 永不匹配
pkill -x web_vllm_rknn || true
sleep 2

# demo 要求 <image> 在提示词内（与 bench_q35_vs_ft_board.sh 同用法），置于开头
# ⚠️ SP 必须单行：demo REPL 逐行对话（main.cpp getline），多行会被拆成多轮、
#    第 2 轮起无图像 → 全是幻觉；且 demo 无 repeat_penalty，结构化必复读。
#    可信的结构化测量请用 ./run_bench_rk.sh --skip yolo,bge --structured --image <目录>
SP='<image> 请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。 预定义属性列表：Female,AgeOver60,Age18-60,AgeLess18,Front,Side,Back,Hat,Glasses,HandBag,ShoulderBag,Backpack,HoldObjectsInFront,ShortSleeve,LongSleeve,UpperStride,UpperLogo,UpperPlaid,UpperSplice,LowerStripe,LowerPattern,LongCoat,Trousers,Shorts,Skirt&Dress,boots,ublack,ugray,ublue,ugreen,uwhite,upurple,ured,ubrown,uyellow,upink,uorange,ubeige,ustriped_color,umulticolor,lwhite,lpink,lred,lgreen,lyellow,lpurple,lbrown,lblack,lorange,lblue,lgray,lbeige,lstriped_color,lmulticolor'

D='data/alerts_images/2026-10-19/一楼电梯厅'
IMGS=()
for folder in 戴眼镜的男生 带帽子的人 穿裙子的女生 黑衣服男生; do
    f=$(find "$D/$folder/roi" -name '*.jpg' 2>/dev/null | sort | head -1)
    [ -n "$f" ] && IMGS+=("$f")
done
echo "测试图 ${#IMGS[@]} 张："
for f in "${IMGS[@]}"; do echo "  - $(echo "$f" | sed 's|.*2026-10-19/||')"; done

run_set () {  # $1=名称 $2=vision $3=llm
    echo "======== $1 ========"
    for img in "${IMGS[@]}"; do
        rule=$(echo "$img" | sed 's|.*一楼电梯厅/||; s|/roi/.*||')
        out=$(printf '%s\nexit\n' "$SP" | timeout 180 ./demo "$img" "$2" "" "" "$3" 512 2048 3 rk3588 2>&1 | tr -d '\r' | grep -ao 'robot:.*' | grep -av '^robot: *$' | head -3)
        echo "[$rule] ${out:-<空/超时>}"
    done
}

run_set "ft-2B W8A8 (微调)" \
    models/qwen3vl-ft_vision_rk3588.rknn models/qwen3vl-ft-merged_w8a8_rk3588.rkllm
run_set "Qwen3.5-0.8B 基座 (线上当前)" \
    models/qwen3.5_vision_rk3588.rknn models/qwen3.5-0.8b_w8a8_rk3588.rkllm

setsid nohup bash ./run_web_rk.sh > web.log 2>&1 < /dev/null 9>&- &
echo DONE_SERVICE_RESTARTED
