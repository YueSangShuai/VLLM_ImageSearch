# FG-CLIP 技术文档

> 本文档描述仓库的**架构与实现细节**：模型结构、训练/评测原理、导出管线、端侧部署服务谱系。
> 用户级快速上手见 [README.md](README.md)，第三方依赖重建见 [DEPENDENCIES.md](DEPENDENCIES.md)，各推理服务另有独立 README。

---

## 目录

1. [总体架构与谱系](#1-总体架构与谱系)
2. [核心模型：fgclip 包](#2-核心模型fgclip-包)
3. [行人属性分类扩展（54 属性体系）](#3-行人属性分类扩展54-属性体系)
4. [训练管线](#4-训练管线)
5. [评测矩阵](#5-评测矩阵)
6. [模型拆分与 ONNX 导出管线](#6-模型拆分与-onnx-导出管线)
7. [端侧推理服务谱系（web* 目录群）](#7-端侧推理服务谱系web-目录群)
8. [qwen_classifer：视觉塔属性头训练工程](#8-qwen_classifer视觉塔属性头训练工程)
9. [数据目录](#9-数据目录)
10. [环境与依赖](#10-环境与依赖)
11. [已知坑与实现怪癖](#11-已知坑与实现怪癖)

---

## 1. 总体架构与谱系

本 monorepo 覆盖「**训练 → 导出 → 多端部署**」完整链路：

```
                 训练侧（GPU）                          导出                     部署侧
┌─────────────────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────────┐
│ fgclip/  FG-CLIP + 54 属性头     │    │ split_fgclip_model.py    │    │ web/            FG-CLIP 匹配 │
│ configs/ + scripts/ 训练评测     │ ─► │   → image/text 双模型    │ ─► │   （GPU PyTorch 原型）        │
│ qwen_classifer/ Qwen 视觉塔属性头 │    │ export_to_onnx.py        │    │ web_vllm*       描述+匹配    │
└─────────────────────────────────┘    │   → ONNX (含 108 维分类)  │    │   （Python → C++ → RK3588）  │
                                       └──────────────────────────┘    └─────────────────────────────┘
```

下游推理服务共享同一业务骨架：**RTSP 取流 → YOLO 行人检测 → ROI 裁剪 → 匹配（图文向量 / VLM 描述 + 文本向量 / reranker 直配）→ 告警落盘 + Web 展示**，在不同硬件上被反复移植。谱系与端口速查见 [第 7 节](#7-端侧推理服务谱系web-目录群)。

---

## 2. 核心模型：fgclip 包

### 2.1 包结构

```
fgclip/
├── model/clip_strc/
│   ├── fgclip.py              # FGCLIPModel + ClassificationHead（核心，577 行）
│   ├── modeling_clip.py       # 改造版 HF CLIP 组件（1424 行，含位置嵌入扩展）
│   └── configuration_clip.py  # CLIPTextConfig / CLIPVisionConfig / CLIPConfig
├── train/
│   ├── train_fgclip.py        # 训练主入口（682 行，含两个 Dataset）
│   ├── clean_clip_trainer.py  # CLIPTrainer：分类 focal loss + mA 指标
│   └── train_mem.py           # 入口转发
└── eval/                      # COCO 检索 / IN1K / IN-V2 / FG-OVD / 行人属性评测
```

### 2.2 FGCLIPModel 架构

`FGCLIPModel(CLIPModel)`（[fgclip/model/clip_strc/fgclip.py:54](fgclip/model/clip_strc/fgclip.py#L54)），基座 `qihoo360/fg-clip-base`：

| 组件 | 结构 | 维度 |
|---|---|---|
| 视觉塔 `CLIPVisionTransformer` | ViT-B/16，12 层 12 头，CLS 池化 + post_layernorm | hidden 768，224px / patch16 → 197 token |
| 文本塔 `CLIPTextTransformer` | 12 层 8 头，causal mask，**EOS 位池化** | hidden 512，max_position 77 |
| `visual_projection` | Linear(768→512, no bias) | |
| `text_projection` | Linear(512→512, no bias)，短文本分支 | |
| `text_filip_projection`（新增） | Linear(512→512, no bias)，长文本分支，初值从 `text_projection` 拷贝（[fgclip.py:150](fgclip/model/clip_strc/fgclip.py#L150)） | |
| 温度（3 个可学习标量） | `logit_scale`（全局 InfoNCE）/ `logit_scale_finegraind`（区域对齐）/ `logit_scale_hardneg`（硬负例），初值均 2.6592 | [fgclip.py:82-84](fgclip/model/clip_strc/fgclip.py#L82-L84) |

### 2.3 双文本分支与长文本位置嵌入扩展

每条训练样本同时有**长 caption（≤248 token）与短 caption（77 token，前缀 `"a photo of "`）两条正例**，走两套独立的投影与位置嵌入：

- **短分支**（`walk_short_pos=True`）：原始 77 行位置嵌入表 + `text_projection`。
- **长分支**（`walk_short_pos=False`）：`CLIPTextEmbeddings` 新增 `position_embedding_ori / _res` 两张 248 行表，按固定掩码拼接——前 20 个 token 查 ori 表、20 以后查 res 表（[modeling_clip.py:261-295](fgclip/model/clip_strc/modeling_clip.py#L261-L295)）；投影走 `text_filip_projection`。
- **77 → 248 插值规则**（`resize_postion_embeding()`，[fgclip.py:110-146](fgclip/model/clip_strc/fgclip.py#L110-L146)，`from_openai=True` 时执行）：`4×77 − 3×20 = 248` 行——前 20 行原样保留；第 20..75 行每对相邻行线性插值出 4 个采样；最后 4 行沿 `emb[76]−emb[75]` 方向外推。ori/res 两表初值相同，训练中被解耦微调。
- 视觉塔另有独立的 2D 位置嵌入 bicubic 插值（`interpolate_pos_encoding`，[modeling_clip.py:218-248](fgclip/model/clip_strc/modeling_clip.py#L218-L248)），支持非 224 输入（如 FG-OVD 评测用 336px）。

### 2.4 Loss 设计

`forward()`（[fgclip.py:326](fgclip/model/clip_strc/fgclip.py#L326)）按需组合四路损失：

| 损失 | 机制 | 温度 | 权重 |
|---|---|---|---|
| 对称 InfoNCE（长 + 短各一次） | `torch.distributed.nn.all_gather` 聚合全 world embedding，正例=对角线，i2t/t2i 双向 CE 取平均；总 clip loss = loss(长) + loss(短) | `logit_scale` | 1.0 |
| box 区域对齐（`add_box_loss`） | 取视觉 encoder 倒数第二层 hidden state，经 `forward_without_attn()`（只重放最后一层 V-proj/out_proj/MLP 补全，[fgclip.py:273](fgclip/model/clip_strc/fgclip.py#L273)）得 `[B,512,14,14]` 特征图；每图最多 4 框 `roi_align(1×1)`，与区域短描述做 pairwise 对比（eye 软标签双向 CE） | `logit_scale_finegraind` | 0.1 |
| 区域硬负例（`use_hard_neg`） | 数据侧每个负例区域编码 `1 正 + 10 硬负例 = 11` 条文本；文本 reshape 为 `(N,11,512)` 与 ROI 特征 einsum 得 N×11 相似度，label=0 单向 CE（[fgclip.py:571](fgclip/model/clip_strc/fgclip.py#L571)） | `logit_scale_hardneg` | 0.5 |
| 属性分类 focal loss | 见[第 3 节](#3-行人属性分类扩展54-属性体系) | — | 1.0 |

“正/负例硬样本”的准确含义：**正例 = 同一区域的长、短两条 caption；负例 = in-batch 全局负例 + 显式提供的每区域 10 条硬负例**（FineHARD 数据）。硬负例数 11 是硬编码 reshape（[fgclip.py:573](fgclip/model/clip_strc/fgclip.py#L573)），要求数据中 `short_expr_negs` 恰有 10 个 key。

---

## 3. 行人属性分类扩展（54 属性体系）

### 3.1 属性定义

[configs/classifier_config.yaml](configs/classifier_config.yaml)（顶层键拼写为 `classiffer_config`，历史 typo）：

- **54 个属性，每个 `nc: 2`**（二分类 no/yes），合计 **108 维**输出；
- 分组：人口学 4（Female / AgeOver60 / Age18-60 / AgeLess18）、姿态 3（Front/Side/Back）、配饰 6（Hat/Glasses/HandBag/ShoulderBag/Backpack/HoldObjectsInFront）、上装 8、下装 6（含 boots）、上装颜色 14（ublack…umulticolor）、下装颜色 13（lwhite…lmulticolor）；
- 每属性附一对 `text_en.yes/no` 英文句子（如 `"the person is female"`），既是文本侧对齐 prompt 也是分类头监督目标的语义来源。

### 3.2 ClassificationHead 结构与接入

```python
class ClassificationHead(nn.Module):            # fgclip.py:33
    Linear(512, 256) → ReLU → Dropout(0.1) → Linear(256, nc=2)
```

- `FGCLIPModel.__init__` 读 classifier_config，为每个属性建一个头存入 `self.classification_heads: nn.ModuleDict`（键=属性名，[fgclip.py:96-104](fgclip/model/clip_strc/fgclip.py#L96-L104)）；
- 输入是 **L2 归一化后的 `image_embeds [B,512]`**（visual_projection 输出），完全不经过文本塔；
- 前向仅在 `classification_labels is not None` 时产出 `classification_logits: Dict[attr, Tensor[B,2]]`。

### 3.3 关键张量流：图 → [B,108]

```
image [B,3,224,224] → ViT-B/16 → CLS+LN [B,768] → visual_projection → L2 归一化
    → image_embeds [B,512]
    → 54 × ClassificationHead(512→256→2) → 各自 [B,2]
    → torch.cat(dim=-1)（split_fgclip_model.py:67）→ classification_output [B, 54×2=108]
```

108 = 54 属性 × 每属性 2 个 logit（no/yes），按 YAML 中属性顺序拼接；`classification_details` 字典同时保留逐属性 `[B,2]`。

### 3.4 分类损失与指标

- **focal loss（γ=2）**，不在模型内而在 `CLIPTrainer.compute_classification_loss()`（[fgclip/train/clean_clip_trainer.py:289](fgclip/train/clean_clip_trainer.py#L289)）：2 类分支 `−[1−p_correct]²·log p_correct`；标签为 −1（未知）的样本经 valid_mask 剔除；总 loss = clip_loss + cls_loss。
- 指标为行人属性标准 **mA**：各属性 (TP+TN)/(TP+TN+FP+FN) 的均值（[clean_clip_trainer.py:21](fgclip/train/clean_clip_trainer.py#L21)）。
- 训练期预测规则为 `sigmoid(logits[:,1]) > 0.5`（取 index-1 的 sigmoid 而非 argmax，属实现怪癖，见[第 11 节](#11-已知坑与实现怪癖)）。

---

## 4. 训练管线

### 4.1 行人属性分类（本地主训练线）

```
scripts/train_classification.sh
  └─ scripts/run_train_classification.py     # YAML → deepspeed 命令行拼装器
       └─ deepspeed -- fgclip/train/train_fgclip.py --deepspeed scripts/zero2.json ...
```

- **配置**（[scripts/train_classification.yaml](scripts/train_classification.yaml)）：冻结 vision/text 主干、只训投影 + 54 分类头；bf16 + ZeRO-2 + gradient_checkpointing(`use_reentrant=False`)；输出 `./checkpoints/fgclip-classification`。configs/ 版与 scripts/ 版是两套超参（小 batch 单卡 vs 4 卡大 batch 高 lr），迁移时以实际启动的为准。
- **数据**：`PedestrianAttributeDataset`（[train_fgclip.py:370](fgclip/train/train_fgclip.py#L370)）读取 `/data/yuesang/person_data/*` 下 SYNTH-PEDES / MARS / PAB / pa100k / 19w / T4 的 `json/*_new_*.json`；每行 = 图 + 长 caption（`text_path` 文件）+ 54 个 0/1/−1 标签；分布式分片用固定 seed=42 shuffle 后按 rank 切片（非 DistributedSampler）。
- **断点续训**：output_dir 下检测 `checkpoint-*` 按最大 step 自动 resume（[train_fgclip.py:667](fgclip/train/train_fgclip.py#L667)）。

### 4.2 FG-CLIP 二阶段微调（上游线）

`scripts/train.sh`：GRIT-12M + FineHARD，`from_openai=True` 时执行位置嵌入插值 + filip 投影拷贝初始化，8 卡 per-device bs 512，开 `add_box_loss` 与 `use_hard_neg`。数据准备见 [第 9 节](#9-数据目录)。

---

## 5. 评测矩阵

| 脚本 | 数据集 | 指标 / 机制 |
|---|---|---|
| `scripts/eval_classification.sh` | 行人属性（PA-100K test 等） | mA（逐属性 acc 均值，valid_mask 过滤） |
| `scripts/eval/eval_in1k.sh` / `eval_inv2.sh` | ImageNet-1K / V2 | 模板均值 zeroshot 分类器，top-1/5 |
| `scripts/eval/eval_coco_retrieval.sh` | COCO 5K 图 × 5 caption | R@1/5/10，双向；`max_length>100` 自动切长文本分支 |
| `scripts/eval/eval_long.sh` | ShareGPT4V-1K + DCI | 1v1 检索 R@1，`max_length 248` |
| `scripts/eval/eval_fgovd.sh` | FG-OVD 区域正/负表达 | ROI 特征 vs 文本列表 softmax，正例排第一记对 → precision |

---

## 6. 模型拆分与 ONNX 导出管线

### 6.1 split_fgclip_model.py：checkpoint → image/text 双模型

[split_fgclip_model.py](split_fgclip_model.py) 把训练 checkpoint 拆成两个可独立部署的 safetensors 模型：

- **分类头自动检测**：扫描 state_dict 中 `classification_heads.<属性名>.net.3.weight` 键，用 `shape[0]` 推类别数，生成 `classifier_config`（[:144-170](split_fgclip_model.py#L144-L170)）；
- **image_model**（`ImageEncoderWithClassifier`，[:23-81](split_fgclip_model.py#L23-L81)）：vision_model + visual_projection + 54 分类头；forward 输出 `image_features [B,512]`（L2 归一化）+ `classification_output [B,108]`；
- **text_model**（`TextEncoder`，[:84-127](split_fgclip_model.py#L84-L127)）：text_model + 双投影，`use_filip` 开关切换长/短分支（输出不做 L2 归一化，由调用方处理）。

### 6.2 三条 ONNX 导出线对比

| | [export_to_onnx.py](export_to_onnx.py) | [export_onnx.py](export_onnx.py) | [tools/export_fgclip_to_onnx.py](tools/export_fgclip_to_onnx.py) |
|---|---|---|---|
| 状态 | **当前主线**（`onnx_exports/` 产物来自它） | ❌ 已失效（依赖坏符号链接 `fg_clip`） | ✅ 独立备选（零依赖 fgclip 包） |
| 输入 | `split_models/{image,text}_model` | `./fg-clip` 整模型 | 原始 HF 权重目录（自带全套模型定义） |
| image 输出 | `image_features[B,512]` + **`classification_output[B,108]`** | 仅 `image_features` | 仅 `image_features` |
| text 输入 | `input_ids`（无 attention_mask），seq 维动态 | `input_ids`+`attention_mask` | `input_ids`+`attention_mask` |
| opset | 14 | 14（可配） | 14（可配） |
| 校验 | ORT vs torch，5 次随机，max_diff < 1e-4 | checker + 10 次 allclose | checker + 2 例，失败 exit(1) |

产物消费关系：`onnx_exports/image_model.onnx` → `web/` 的属性分类；`split_models/` 是导出中间产物。
另有 [tools/fgclip_loader.py](tools/fgclip_loader.py) 用 `exec` 动态加载上游 `modeling_fgclip.py` 的兼容路线（依赖 MARS 路径），含 `get_image_dense_features` 密集特征图接口。

---

## 7. 端侧推理服务谱系（web* 目录群）

一套「行人布控」业务骨架在 6 个服务中逐硬件移植：

```
web (5001, FG-CLIP 图文匹配, PyTorch GPU)            qwen_classifer (属性头训练 → attr_head.bin)
   │ 业务骨架复用                                          │
   ▼                                                      │
web_vllm (5003, Qwen-VL 描述 + BGE, transformers/GGUF)    │
   │ 全量 C++ 移植（REST 契约不变）                         │
   ▼                                                      │
web_vllm_cpp (5003, llama.cpp + ONNX Runtime, 纯 CPU)     │
   │ 移植到 RK3588 NPU（llama.cpp → RKNN+RKLLM）            │
   ▼                                                      ▼
web_vllm_rknn (板端 5003, Qwen3-VL-2B-ft 线上版) ◄── attr_head.bin
   ├─ 克隆换 Qwen3.5-0.8B → web_vllm_rknn_q35        (板端 5004, 并行实例)
   └─ 匹配引擎换 Reranker → web_vllm_rknn_reranker   (板端 5005, 实验分支)
```

### 7.1 [web/](web/README.md) — FG-CLIP 图文匹配版（端口 5001）

最早期 GPU 单机版。RTSP → YOLOv8n 检测（ultralytics/PyTorch-CUDA）→ ROI 裁剪 224×224 → **FG-CLIP 图像塔 vs 布控规则正/负样本文本塔 embedding cosine**；同时用本仓库训练的 54 属性分类头做校验。触发条件 `best_pos ≥ threshold 且 > max_neg + score_diff`。核心五件套（DeviceManager / PersonDetector / FGCLIPMatcher / StreamProcessor / MonitoringManager）都在 [web/processor.py](web/processor.py)，是后续所有 Python 版的蓝本。

### 7.2 [web_vllm/](web_vllm/README.md) — Qwen-VL 描述 + BGE 版（端口 5003）

从「图文直接匹配」转向「**VLM 生成中文描述 → 描述过 BGE(bge-small-zh-v1.5) → 与规则文本向量匹配**」。VLM 为 Qwen3-VL-2B（基座或 LoRA 合并模型 qwen3vl-ft），进程内二选一：HF transformers 或 GGUF（llama-cpp-python，`.gguf` 后缀 + `VLM_MMPROJ_PATH` 切换），不依赖 vLLM 服务（目录名是历史）。四种匹配模式（`VLM_MATCH_MODE`）：`structured`（54 属性符号匹配）/ `description`（纯 BGE）/ `combined`（默认线上）/ `rule_direct`（规则代入 prompt 由 VLM 直判）。

### 7.3 [web_vllm_cpp/](web_vllm_cpp/README.md) — C++17 纯 CPU 移植（端口 5003）

web_vllm 的逐行 C++ 移植，全链路进程内：VLM = vendored llama.cpp（mtmd 多模态路径，默认 `models/qwen3vl-ft-20260909/qwen3vl-ft-f16.gguf` + mmproj，CPU 全层）；YOLO 与 BGE 均走 ONNX Runtime（BGE 分词为自带 vocab 实现）；ffmpeg 取流；cpp-httplib 复刻 Flask 全部 REST 路由。每个头文件注释标明对应的 Python 类（`monitor.h`↔MonitoringManager 等）。支持 aarch64 交叉编译与离线构建；`scripts/rest_ab_test.sh` 做 Python↔C++ 契约回归。

### 7.4 [web_vllm_rknn/](web_vllm_rknn/README.md) — RK3588 NPU 版（板端 5003，线上）

NanoPi R6S 上把 VLM 拆为**视觉塔 RKNN + 语言塔 RKLLM**（`RKLLM_INPUT_MULTIMODAL` 流式生成，支持三 NPU 核），YOLO 走 RKNN，BGE 保留 ORT-CPU。模型双件套：基座与 ft 微调各一对 `.rknn`+`.rkllm`（注意环境变量名沿用 `VLM_MMPROJ_PATH` 但实指 `.rkllm`）。新增**属性线性头 `attr_head.bin`**（[cpp/src/attr_head.cpp](web_vllm_rknn/cpp/src/attr_head.cpp)，magic `AHEAD1`，2048×54 线性头 CPU 前向 <1ms），由 [qwen_classifer](#8-qwen_classifer视觉塔属性头训练工程) 训练导出——combined 模式下属性由头直接输出，不再让 VLM 逐个吐属性词。x86 主机负责 rknn-toolkit2/rkllm-toolkit 模型转换、aarch64 交叉编译与推包；板上只跑推理二进制 + Web。提示词须中文，threshold 0.45。

### 7.5 [web_vllm_rknn_q35/](web_vllm_rknn_q35/README.md) — Qwen3.5-0.8B 并行实例（板端 5004）

web_vllm_rknn 的克隆：换 `qwen3.5_vision_*.rknn` + `qwen3.5-0.8b_w8a8_*.rkllm`、端口 5004、独立进程名，与线上 5003 共享 NPU（压测须 flock 互斥）。⚠️ README 声称「cpp/src 零改动」，但主项目后来新增的 attr_head 等**尚未回灌** q35，两边 cpp/src 已漂移。

### 7.6 [web_vllm_rknn_reranker/](web_vllm_rknn_reranker/README.md) — Reranker 直配实验（板端 5005）

把「VLM 描述 + BGE cosine」整体替换为 **Qwen3-VL-Reranker-2B 直接对 (ROI 图, 规则文本) 打相关性分**：同样拆 RKNN 视觉塔 + RKLLM 语言塔，取末 token **yes/no logit 差 `sigmoid(l_yes − l_no)`** 作分（[cpp/src/reranker_scorer.cpp](web_vllm_rknn_reranker/cpp/src/reranker_scorer.cpp)）。`final_score = max(positive 文本分)`，threshold/score_diff 逻辑不变，但分数分布压缩在 0.2~0.7，需按 `calib/host_scores.json` 重标定。reranker 是编译进二进制的库内模块（`VLM_MATCH_MODE=reranker`），不是独立网络服务。

---

## 8. qwen_classifer：视觉塔属性头训练工程

非部署服务，是板端 `attr_head.bin` 的**生产者**（README 见 [qwen_classifer/README.md](qwen_classifer/README.md)）：

- 思路：冻结 Qwen3-VL-2B 视觉塔（ViT + patch merger，输出 2048 维）+ 线性/MLP 头做 **PA-100K 54 属性多标签分类**；
- 流程：`extract_features.py`（冻结提特征，分片断点缓存 `cache/*.pt`）→ `train_head.py`（masked BCE，−1 标签 mask）→ `eval_head.py`（逐属性 AP/F1）→ `export_head.py` 直接导出到 `../web_vllm_rknn/output_cpp/models/attr_head.bin`；
- 与 FG-CLIP 线的 ClassificationHead 是**两套并行的属性头**：前者基于 Qwen 视觉塔 2048 维（服务 web_vllm_rknn 板端），后者基于 FG-CLIP 512 维 embedding（服务 web/ 与 ONNX 导出线）。
- 预处理 `--size 448` 与板端 RKNN 逐像素对齐，保证训练指标可迁移。

---

## 9. 数据目录

```
data/
├── data.md + get_data.sh + convert_image_name.py   # 上游 FineHARD/GRIT-12M：
#     get_data.sh 用 img2dataset 下 2100 个 parquet（21×100 目录），
#     convert_image_name.py 按 url2key_jsons 改名对齐 FineHARD region 标注
├── fgovd/          # FG-OVD 评测标注 4 个 jsonl（COCO val bbox + pos/neg 表达）
└── IN1K_val/       # imagenet2012_mapclsloc.txt 类别映射（val 图片不在库内）
```

行人属性数据不在库内，位于 `/data/yuesang/person_data/`（SYNTH-PEDES、MARS、PAB、pa100k、19w、T4），每数据集 `json/*_new_*.json`，每行含 `image_path / text_path / 54 个属性列（0/1/−1）`。

---

## 10. 环境与依赖

- **训练/导出主环境**：`pip install -r requirements_fgclip2_minimind3v.txt` —— 硬 pin `torch==2.6.0+cu124`、`torchvision==0.21.0+cu124`；`transformers>=4.53,<5.0`（pyproject 里的 `transformers==4.41.0` 是上游遗留，**以 requirements 为准**）；`numpy<2.0`；deepspeed/accelerate/sentence-transformers/flask/ultralytics 等按场景分组。
- **第三方大体积依赖**（被 .gitignore 排除，克隆后按 [DEPENDENCIES.md](DEPENDENCIES.md) 重建）：`web_vllm_cpp/third_party/`（ONNX Runtime 1.18.1 x64+aarch64 预编译、Eigen、nlohmann/json、cpp-httplib、stb）、`web_vllm_cpp/llama_buildsrc/`（llama.cpp，需 `LLAMA_BUILD_MTMD=ON`）、`web_vllm_rknn/rknn-llm/`（airockchip/rknn-llm）。
- **模型权重均不入库**：基座从 HF 下载（qihoo360/fg-clip-base 等）；部署权重（bge-small-zh/en、qwen3vl-ft GGUF+mmproj、yolov8n ONNX、RKNN/RKLLM 件）见各服务 README；`checkpoints/fgclip-classification` 为训练产物。
- 板端模型转换需要两个互斥 conda 环境（rknn-toolkit2 / rkllm-toolkit），见 web_vllm_rknn/README 与 INSTALL。

---

## 11. 已知坑与实现怪癖

| # | 问题 | 位置 / 处理 |
|---|---|---|
| 1 | `fg_clip` 是指向不存在目录的坏符号链接；根目录 `export_onnx.py` 因此失效 | 新代码一律 `import fgclip`；导出走 `export_to_onnx.py` 或 `tools/` 线 |
| 2 | `eval_classification.py` 独立运行时把 `classification_logits` 恒算成 None：`FGCLIPModel.forward` 仅在传入 `classification_labels` 时才产出 logits，而脚本没传 | [fgclip/eval/eval_classification.py:178](fgclip/eval/eval_classification.py#L178)；且 `:189` 的 `sigmoid(logits.squeeze(-1))` 对 `[B,2]` 形状处理也有误。训练内 eval（trainer `prediction_step`）不受影响 |
| 3 | 训练内 eval 的属性预测用 `sigmoid(logits[:,1])>0.5`，与 focal loss 的 softmax 语义不一致 | [clean_clip_trainer.py:438](fgclip/train/clean_clip_trainer.py#L438) |
| 4 | 硬负例数量 11 硬编码，要求数据 `short_expr_negs` 恰 10 个 key | [fgclip.py:573](fgclip/model/clip_strc/fgclip.py#L573) |
| 5 | `eval_classification.py` 按 `max_length=77` tokenize，与训练长文本 248 不一致 | [fgclip/eval/eval_classification.py:238](fgclip/eval/eval_classification.py#L238) |
| 6 | `scripts/eval/eval_coco_boxcls.sh` 模块名 `myclip.` 为旧包名残留 | 应为 `fgclip.` |
| 7 | `shareg4v_retrieval.py` 顶部 `import torch_npu`，GPU 环境需注释掉 | — |
| 8 | YAML 顶层键 typo `classiffer_config` | 改代码不如将错就错，勿改 |
| 9 | 大量配置硬编码本地绝对路径（基座模型、person_data 等） | 换机全局搜索替换 |
| 10 | q35 与主项目 web_vllm_rknn 的 cpp/src 已漂移（attr_head 等未回灌） | 同步时以主项目为准 |
| 11 | `FGCLIPModel.forward` 无条件调用 `dist.get_rank()`，依赖非分布式下返回 0 的行为 | [fgclip.py:357](fgclip/model/clip_strc/fgclip.py#L357) |

---

*本文档与代码同步维护：新增服务/导出线/属性体系变更时请一并更新。*
