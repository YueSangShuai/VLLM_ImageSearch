#!/bin/bash
# 视觉塔 A/B：固定 ft-merged LLM，只换 vision（ft 微调塔 vs 基座塔），判定「结构化不准」
# 是否出在 qwen3vl-ft_vision_rk3588.rknn。⚠️ 独占 NPU：跑前停服务，跑完不自动拉起（脚本外决定）。
set -uo pipefail
cd /home/Desktop/output_cpp
export LD_LIBRARY_PATH="$PWD/lib"
exec 9>/tmp/bench_q35.lock
flock -n 9 || { echo "another bench running" >&2; exit 1; }
pkill -f "[w]eb_vllm_rknn_q35" || true
pkill -x web_vllm_rknn || true
sleep 2

# ⚠️ 必须单行：demo REPL 是 std::getline 逐行对话（main.cpp:425），且仅当该行含 <image>
# 才注入图像 embedding（main.cpp:447）。多行 prompt 会被拆成多轮、第2轮起没有图像，
# 输出必然是幻觉枚举 —— 旧 bench_struct_ab_board.sh 的「垃圾枚举」结论即此测试脚本假象。
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

run_set "vision=ft微调塔 + llm=ft W8A8 (线上组合)" \
    models/qwen3vl-ft_vision_rk3588.rknn models/qwen3vl-ft-merged_w8a8_rk3588.rkllm
run_set "vision=基座塔 qwen3-vl + llm=ft W8A8 (换塔对照)" \
    models/qwen3-vl_vision_rk3588.rknn models/qwen3vl-ft-merged_w8a8_rk3588.rkllm

echo DONE_VISION_AB
