#!/usr/bin/env bash
# 多卡分片提特征启动脚本。
#
# 用法：
#   ./run_extract.sh <mode> [gpus] [shards]
#
#   ./run_extract.sh baseline                    # 基线：pa100k train + MARS val（默认 0/6 两卡，~17 分钟）
#   ./run_extract.sh full "0 6"                  # 全量，两张卡各跑一半（9~11 小时）
#   NUM_SHARDS=8 ./run_extract.sh full "0 6" "0 6"
#       # 预切 8 片，本次只跑 shard0(->卡0) 和 shard6(->卡6)，各约 2.3 小时；
#   NUM_SHARDS=8 ./run_extract.sh full "2" "2"   # 之后卡 2 空出来，只补跑 shard2
#   NUM_SHARDS=8 ./run_extract.sh full "1 3" "1 3"   # 继续补，直至 shard0..7 全部完成
#
#   ./run_extract.sh full "0" "2 3 4 5 6"   # 队列模式：一张卡串行排队跑 5 个分片
#
# 说明：
#   - shards 数 >= gpus 数即可；多出的分片在同卡上自动串行排队（轮转分配）
#   - 前台运行时进度条直接显示在终端，同时 tee 进 logs/ 留档
#   - NUM_SHARDS 一旦定了不要中途改（分片内容随总数变化，缓存不可混用）
#   - 已存在的 cache/<split>_s<k>.pt 自动跳过（断点续跑）
#   - 全部 8 片完成后即可训头：
#       python train_head.py --train-cache cache/train_s*.pt --val-cache cache/val_s*.pt \
#           --run-name full_linear
set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-baseline}"
GPUS="${2:-0 6}"
SHARDS="${3:-}"

PY=/home/yuesang/miniconda3/envs/llama/bin/python
BATCH=128          # 瓶颈在 CPU 解码，batch 大小基本不影响总耗时
WORKERS=12
SIZE="${SIZE:-448}"            # 输入边长；非 448 时缓存名自动带 _<SIZE>px 后缀防混用
SIZE_TAG=""
[[ "$SIZE" == "448" ]] || SIZE_TAG="_${SIZE}px"
mkdir -p logs cache

source ./data_files.sh
if [[ "$MODE" == "baseline" ]]; then
  TR=("${TRAIN_JSONS_PA100K[@]}")
elif [[ "$MODE" == "full" ]]; then
  TR=("${TRAIN_JSONS[@]}")
else
  echo "mode 只能是 baseline | full"; exit 1
fi

read -r -a GPU_ARR <<< "$GPUS"
if [[ -z "$SHARDS" ]]; then
  # 未指定 shards：默认 0..卡数-1，一卡一片
  SHARDS="$(seq -s ' ' 0 $(( ${#GPU_ARR[@]} - 1 )))"
fi
read -r -a SHARD_ARR <<< "$SHARDS"
# 未显式指定 NUM_SHARDS 时自动推断：取（本次请求的分片号 ∪ 已有缓存的分片号）最大值+1，
# 保证与历史缓存同一种切法（如已有 s0/s1/s7 -> 推断 8）
if [[ -z "${NUM_SHARDS:-}" ]]; then
  NUM_SHARDS=0
  for s in "${SHARD_ARR[@]}"; do (( s + 1 > NUM_SHARDS )) && NUM_SHARDS=$((s + 1)); done
  for f in cache/train_s* cache/val_s*; do
    idx=$(basename "$f" | sed -n 's/.*_s\([0-9]\+\).*/\1/p')
    [[ -n "$idx" ]] && (( idx + 1 > NUM_SHARDS )) && NUM_SHARDS=$((idx + 1))
  done
fi

if [[ ${#SHARD_ARR[@]} -lt ${#GPU_ARR[@]} ]]; then
  echo "错误：shards(${#SHARD_ARR[@]}) 数量不能少于 gpus(${#GPU_ARR[@]})"
  exit 1
fi
for s in "${SHARD_ARR[@]}"; do
  if (( s < 0 || s >= NUM_SHARDS )); then echo "错误：shard $s 超出 0..$((NUM_SHARDS-1))"; exit 1; fi
done

# 磁盘余量提示（全量特征约 26 GB）
avail_gb=$(df -BG --output=avail cache | tail -1 | tr -dc '0-9')
if [[ "$MODE" == "full" && "$avail_gb" -lt 30 ]]; then
  echo "警告：cache 所在盘仅剩 ${avail_gb}G，全量特征需约 26G，可能中途写满！"
fi

echo "mode=$MODE  GPUs=[$GPUS]  shards=[${SHARDS}]  num_shards=$NUM_SHARDS"

pids=()
# 每张卡一条串行链：名下分片按序排队跑，前一片成功才跑下一片
launch() {
  local split="$1"; shift
  local jsons=("$@")
  local nm="${split}${SIZE_TAG}" gcount=${#GPU_ARR[@]}
  for g in "${!GPU_ARR[@]}"; do
    (
      for ((k=g; k<${#SHARD_ARR[@]}; k+=gcount)); do
        s="${SHARD_ARR[$k]}"
        done_f="cache/${nm}_s${s}.done"; legacy_f="cache/${nm}_s${s}.pt"
        if [[ -f "$done_f" || -f "$legacy_f" ]]; then
          echo "[skip] ${nm} shard ${s} 已完成"; continue
        fi
        log="logs/${MODE}_${nm}_s${s}.log"
        echo "[run ] ${nm} shard ${s}/${NUM_SHARDS} -> GPU ${GPU_ARR[$g]}  ($log)"
        CUDA_VISIBLE_DEVICES="${GPU_ARR[$g]}" "$PY" extract_features.py \
          --name "$nm" --shard-id "$s" --num-shards "$NUM_SHARDS" \
          --batch-size "$BATCH" --workers "$WORKERS" --size "$SIZE" \
          --json "${jsons[@]}" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        [[ $rc -eq 0 ]] || { echo "[fail] ${nm} shard ${s} 退出码 $rc，本卡后续分片中止"; exit "$rc"; }
      done
    ) &
    pids+=($!)
  done
}

launch val   "${VAL_JSONS[@]}"
launch train "${TR[@]}"

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "本次指定的分片都已存在，无需启动"
else
  echo "已启动 ${#pids[@]} 个进程，等待完成..."
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "------------------------------------------------"
  ls -lh cache/*.pt 2>/dev/null
  [[ $fail -eq 0 ]] || { echo "有进程失败！查看 logs/（重跑本命令会跳过已完成分片）"; exit 1; }
fi

# 完成度汇总（.done 标记或旧版单文件均算完成）
is_done() { [[ -f "cache/$1${SIZE_TAG}_s$2.done" || -f "cache/$1${SIZE_TAG}_s$2.pt" ]]; }
done_tr=0; done_va=0
for s in $(seq -s ' ' 0 $((NUM_SHARDS-1))); do
  is_done train "$s" && ((done_tr++))
  is_done val   "$s" && ((done_va++))
done
echo "size=${SIZE}  进度：train ${done_tr}/${NUM_SHARDS} 片，val ${done_va}/${NUM_SHARDS} 片"
if [[ $done_tr -eq NUM_SHARDS && $done_va -eq NUM_SHARDS ]]; then
  echo "全部完成。下一步："
  echo "  $PY train_head.py --train-cache \"cache/train${SIZE_TAG}_s*_c*.pt\" --val-cache \"cache/val${SIZE_TAG}_s*_c*.pt\" --run-name ${MODE}_${SIZE}px"
fi
