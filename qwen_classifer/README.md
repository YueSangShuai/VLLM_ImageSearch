# qwen_classifer — Qwen3-VL-2B 视觉塔 + 线性头 行人属性分类（方案 A）

冻结 Qwen3-VL-2B-Instruct 的视觉塔（ViT + patch merger，输出 2048 维），
后接线性/MLP 头做 PA-100K 54 属性多标签分类。分两阶段：

1. **extract_features.py**：视觉塔冻结、只前向，把全量图片提成特征缓存到 `cache/*.pt`
2. **train_head.py**：在缓存特征上训头（秒级一个 epoch，可反复调参不重过视觉塔）

## 环境

用 `llama` conda 环境（transformers 5.8 + torch cu130，已支持 Qwen3VL）：

```bash
conda activate llama
cd /data/yuesang/LLM/FG-CLIP/qwen_classifer
# 本机 8 卡常年高负载，先 nvidia-smi 挑空卡：
export CUDA_VISIBLE_DEVICES=6
```

## 数据

- 格式实例：`/data/yuesang/LLM/MARS/train_sample_128.json`（list of dict，54 属性 + `image_path`）
- 标签取值 `-1/0/1`，**-1 = 未知**，训练损失与 mAP/Acc 评测均自动 mask，不计入
- `--json` 支持多个文件自动拼接，已核对与 MARS 配置
  （`configs/PS_cuhk_pedes.yaml`）完全兼容；额外字段（`hair_length`、
  `attributes_path`）自动忽略。各文件规模：

| 用途 | 文件 | 张数 |
|---|---|---|
| train | SYNTH-PEDES train_new_processed | 2,130,673 |
| train | MARS train/val_new_processed | 1,247,104 + 58,531 |
| train | PAB train/val_new_processed | 1,000,083 + 25,100 |
| train | pa100k train_new_processed | 79,918 |
| train | 19w train_new_relabel | 243,216 |
| train | T4/V1 train_new_relabel | 725,216 |
| train | T4/V2 train_new_relabel | 1,115,726 |
| val/test | pa100k test/val_new_processed | 9,975 + 9,998 |
| val | 19w val_new_relabel | 27,025 |
| val | T4/V1 val_new_relabel | 80,534 |

> 注意 train 全套约 **635 万张**，单卡提特征要 2~3 天。建议先用 pa100k 子集验证
> （79,918 张约 1~2 小时），多卡并行用 `--num-shards K --shard-id k` 分片，
> train_head 直接把 `cache/train_s*.pt` 一起传入即可。

## 训练流程

```bash
# ① 提特征（分片断点续跑，已完成的分片自动跳过）
./run_extract.sh baseline                  # 基线：pa100k train + MARS val，两卡 ~17 分钟

# 队列模式：shards 数可以多于 gpus 数，同一张卡上自动串行排队。
# 例：一张 0 号卡包揽所有剩余分片（约 8h）：
./run_extract.sh full "0" "2 3 4 5 6"
# 两卡分工：卡0 排队跑 2/4/6，卡6 排队跑 3/5：
./run_extract.sh full "0 6" "2 4 6 3 5"

# 多卡并行（每卡一片）：
NUM_SHARDS=8 ./run_extract.sh full "0 6" "0 6"     # shard0->卡0，shard6->卡6
NUM_SHARDS=8 ./run_extract.sh full "2" "2"         # 卡 2 空出来后补跑，以此类推

# 前台跑进度条直接显示在终端（同时 tee 进 logs/）；
# NUM_SHARDS 不指定时自动按已有缓存推断，与历史切法保持一致；
# 定了 8 就别中途改，分片内容随总数变化、缓存不可混用

# 特征按块写盘（默认 30 万张/块 ≈1.2GB）：cache/train_s0_c0000_n300000.pt ...
# - 每块写完即原子落盘，进程挂掉重跑同一命令自动从断点续提（不必重头）
# - 整个 shard 完成才写 cache/train_s0.done；run_extract.sh 以 .done 判断跳过
# - --chunk-size 0 可退回单文件模式（不推荐，中断即全丢）

# ② 训头（默认纯线性头；--hidden 2048 变两层 MLP；分块缓存全部传入）
python train_head.py --train-cache "cache/train_s*_c*.pt" --val-cache "cache/val_s*_c*.pt" \
    --run-name linear448

# ③ 评测：逐属性 AP / Acc@0.5 / 最优阈值 F1 / mAP / 平均准确率
python eval_head.py --ckpt runs/linear448/best.pt --cache "cache/val_s*_c*.pt"
```

数据集清单集中在 `data_files.sh`（与 MARS PS_cuhk_pedes.yaml 一一对应），
增删数据源改那里即可。手动分片跑法（不走脚本）：

```bash
CUDA_VISIBLE_DEVICES=$k python extract_features.py --name train \
    --num-shards 3 --shard-id $k --json <...>   # k = 0..2
```

## 指标输出

- 训练时每个 epoch 打印 val **mAP** 与 val **平均准确率**（54 类 Acc@0.5 求平均），
  逐轮记录在 `runs/<name>/history.csv`
- best epoch 的**每类准确率**自动存 `runs/<name>/per_class_acc.csv`
- `eval_head.py` 输出逐属性表（AP / Acc@0.5 / F1@最优阈值 / 最优阈值 / 正样本数），
  按 AP 升序排列（最差类在最上面），末尾给出 mAP、平均准确率、macro F1

冒烟测试（不占卡太久，先用 sample 验证链路）：

```bash
python extract_features.py --json /data/yuesang/LLM/MARS/train_sample_128.json \
    --name sample --batch-size 16
python train_head.py --train-cache cache/sample.pt --val-cache cache/sample.pt \
    --epochs 5 --run-name smoke
```

## 选项与说明

- `--use-deepstack`：提特征时额外拼接 3 个 deepstack 分支的 mean pooling，
  特征变 8192 维。多尺度信息对细粒度属性（条纹/拼色）可能有帮助，但头更大。
- `--size 448`：输入边长，必须为 32 的倍数。448 与板端 RKNN 部署的固定输入一致，
  预处理的 `expand2square(bg=127.5) → resize` 也和 `web_vllm_rknn` 部署端逐像素对齐，
  板端精度可直接对齐训练结果。
- 指标：PA-100K 惯例的逐属性 AP（无正样本的属性跳过）取均值 = mAP。
- 头很小（线性 4096×54 ≈ 11 万参数），过拟合靠 wd=0.01 + OneCycle 控制；
  加颜色扰动等在线增广时不要用水平翻转（`Front/Side` 等朝向属性会失真）。

## 部署提示

训好的头就是 1~2 个 `nn.Linear`。若要在 RK3588 单模型出结果，把
mean pooling + 头并进 `web_vllm_rknn` 的 ONNX 导出 wrapper（export_vision.py 的
`qwen3_vl_vision` 类 forward 末尾），一次 rknn.run 直接输出 54 维 logits；
注意导出用的视觉塔要与本训练用的权重一致（原始 2B 或微调后 merged 模型）。

## 文件

| 文件 | 作用 |
|---|---|
| common.py | 54 属性定义、数据加载、-1 mask、AP/mAP |
| data.py | expand2square+resize 预处理、DataLoader collate（过 Qwen3VLProcessor） |
| vision.py | 视觉塔加载（释放 LLM 权重省显存）与特征提取 |
| extract_features.py | 阶段一：特征缓存 |
| train_head.py | 阶段二：masked BCE 训头 + 按 mAP 存 best |
| eval_head.py | 逐属性 AP / F1 阈值表 |
