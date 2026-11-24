# vit_align：图文向量化网络（冻结主干 + 可训 MLP）

双塔并联的行人检索向量化网络，两个向量 cosine 直搜，衔接 web_vllm_rknn 链路：

```
图: Qwen3-VL-2B 视觉塔(冻结) -> mean-pool [196x2048→2048] -> ProjMLP(可训) -> 512 维 L2
文: BGE-small-zh-v1.5(冻结, 生产同款) -> CLS pool -> 512 维 L2
```

主干权重完全不动，唯一可训练件是图像侧 ProjMLP（2048→1024→512，LayerNorm+GELU）
与可学习温度。图像向量对齐的是 BGE **无前缀 doc 空间**；中文 query 上线时加生产
官方前缀（`为这个句子生成表示以用于检索相关文章：`）再检索图像，与 web_vllm_rknn 的
BGE 非对称检索语义一致。

板端落地：文侧向量生产已有；图侧 = 板上已有视觉塔 rknn（196×2048，mean-pool 后
2048）+ `models/vproj_head.bin`（`export_head.py` 导出，格式与加载公式在脚本头部注释，
仿 attr_head.bin 惯例）。

## 快速开始

```bash
bash scripts/make_json.sh              # CUHK-PEDES -> data/json/{train,test}.json
python text.py --selfcheck             # HF BGE vs 生产 ONNX 一致性（应 PASS）
bash scripts/extract.sh                # 冻结主干离线提特征（.done 断点续提）
bash scripts/train.sh                  # 训 MLP（吃缓存矩阵，单卡几分钟）
bash scripts/eval.sh                   # i2t/t2i R@1/5/10+mAP + 导出 bin + 零差校验
```

配置都在 `configs/default.yaml`（权重路径、维度、超参、前缀）。

## 目录

| 文件 | 职责 |
|---|---|
| `vision.py` / `extract_image_feats.py` | 视觉塔加载（移植 qwen_classifer）+ 分块提特征缓存 |
| `text.py` / `extract_text_feats.py` | BGE 文本塔（CLS+L2，sentence_bert_config 小写口径）+ caption 缓存；`--prefix` 出查询版 |
| `data/make_cuhk_json.py` | 扫 `imgs/{train,test}_query` 的 pXXXX_sYYYY.jpg + 同名 .txt 生成条目 json |
| `model.py` `loss.py` `train.py` `eval.py` | ProjMLP + 对称 InfoNCE（可开同 id 多正例）+ 训练/评估 |
| `export_head.py` | MLP → vproj_head.bin（含 `--verify` numpy 复刻零差校验） |
| `metrics.py` | Recall@K / mAP（按 id 判相关） |

## 首轮实验结果（CUHK-PEDES，2026-11-24）

| 变体 | 配置 | i2t R@1 | t2i R@1 | i2t R@5 | i2t mAP |
|---|---|---|---|---|---|
| 仅 MLP（按原方案） | `configs/default.yaml` | 0.1053 | 0.1076 | 0.2635 | 0.0856 |
| hidden2048 + lr3e-3 | `configs/big.yaml` | 0.1440 | — | — | — |
| **text_proj 开 + 800ep（当前最佳）** | `configs/tp600.yaml` | **0.2169** | 0.2022 | 0.4460 | 0.1710 |
| text_proj + wd0.1 | `configs/tp_wd.yaml` | 0.2170 | — | — | — |
| 锚点：MARS（ALBEF 主干全微调） | — | ≈0.25 | — | — | — |

- 主要增益来自文侧恒等初始化轻投影（0.105→0.21）：等价于在冻结 BGE 空间外再学一个
  512×512 线性对齐，仍不动主干。默认 yaml 保持仅 MLP（用户原方案），跑最佳版：
  `CFG=configs/tp600.yaml` + `PY=... train.py --config configs/tp600.yaml --name tproj_long`。
- 加中文前缀的英文 caption 查询 t2i 掉到 0.115——前缀对英文文本移动向量明显；
  上线中文 query 的真实表现需用中文查询实测（可用 MARS translation 中文描述重训一版）。
- 过拟合明显（train acc1 0.98，5.5k id 太少），test 峰值出现在 ~110 epoch，别信末轮。

## 口径与已知风险

- 训练预处理与板端一致：expand2square(127.5) → resize 448 → (x-0.5)/0.5，
  mean-pool 与板端 AttrHead 同口径（保证 rknn 侧可直接喂 2048 特征）。
- CUHK-PEDES 描述是英文，BGE-small-zh 英文能力弱于中文 ⇒ 指标偏低属预期。
  补强路径：用 MARS `translation/` 管线把描述译成中文再加 `text_zh` 字段训练。
- 基线锚点：MARS（ALBEF 双塔，同数据域）R@1≈0.25。
- 若纯 MLP 到顶：先加大 `head.hidden`；再不够开 `head.text_proj: true`
  （文侧恒等初始化轻投影，仍不动主干）。

## 环境

用 conda env `llama` 的 python（transformers 5.8 / torch cu130，支持 Qwen3-VL，
与 qwen_classifer 同款；scripts 已写死，可用 `PY=` 覆盖）。视觉塔权重走
HF cache（`/home/yuesang/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct`），
BGE 走 `../models/bge-small-zh-v1.5`，路径均在 yaml。
