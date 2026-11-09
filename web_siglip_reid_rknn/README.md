# web_siglip_reid_rknn：SigLIP2 行人图文直接匹配（RK3588 NPU）

框架复刻自 `../web_vllm_rknn`（YOLOv8n 检测 → 裁剪 → 嵌入 → 规则匹配 → web 告警/调试捕获），
推理核心把「Qwen3-VL-2B 生成中文描述 + BGE-small-zh 文本匹配」两级结构替换为
**MarketaJu/siglip2-person-description-reid 图文同空间直接匹配**：

```
RTSP 帧 → YOLOv8n(NPU) 行人检测 → ROI 裁剪(224x224)
        → SigLIP 视觉塔(NPU, RKNN fp16) → 768-d 归一化图像向量
规则(英文属性短句) → SigLIP 文本塔(CPU, ONNX Runtime + Gemma sentencepiece) → 768-d 文本向量
        → cosine × 阈值/冷却 → 告警（web 展示 ROI/原图/逐规则分数）
```

与旧框架的关键差异：

| | web_vllm_rknn | 本项目 |
|---|---|---|
| 行人理解 | VLM 生成中文描述（秒级/人） | 视觉塔直接出向量（几十 ms/人） |
| 规则匹配 | BGE 描述×规则（中文） | SigLIP 图像×规则（**英文**，同空间） |
| 分值尺度 | 0.35–0.6 | **0.10–0.22**（PETA 实测：正 ~0.15-0.22，无关 ≈0）|
| 符号/属性支线 | attr_head（绑定 Qwen 嵌入） | 已移除 |
| 分词 | WordPiece（手写 C++） | Gemma sentencepiece（官方库交叉编译）|

规则写法（`data/tasks.json`，`positive` 优先，回退 `text`；`negative` 可选，同空间编码，
`pos ≥ threshold 且 pos > max(neg)+score_diff` 才告警——web 规则页负样本框一行一条）：
```json
{"name": "黑色上装", "text": "person, black upper clothing",
 "positive": ["person, black upper clothing"],
 "negative": ["person, white upper clothing"],
 "threshold": 0.12, "interval": 60, "enabled": true}
```

## 目录

- `export/` — x86 主机：`onnx_export.py`（拆塔 ONNX + golden）、`rknn_convert.py`（视觉塔 RKNN + 模拟器对拍）
- `onnx/`、`rknn/`、`golden/` — 模型产物与 x86 golden
- `cpp/` — C++ 工程（`build_scripts/build_sentencepiece_aarch64.sh` → `build_aarch64.sh`）
- `scripts/golden/` — 板端对拍（`export_golden.py` + `rknn_check.cpp`）
- `scripts/deploy|e2e` — 推板 / 端到端验证（沿用 web_vllm_rknn 通道，pi@10.8.18.118）
- `output/` — 部署包（`run_web_rk.sh` / `run_bench_rk.sh`，scp 上板即用）

## 快速开始

```bash
# x86：产出模型（已完成可跳过）
/home/yuesang/miniconda3/envs/rknn/bin/python export/onnx_export.py
/home/yuesang/miniconda3/envs/rknn/bin/python export/rknn_convert.py
/home/yuesang/miniconda3/envs/rknn/bin/python scripts/golden/export_golden.py

# x86：交叉编译（sp 只需一次；sp 源 = third_party/sentencepiece-src v0.2.0 GitHub 克隆，
#        其 src/CMakeLists.txt 已本地修补 CMAKE_INSTALL_INCDIR→INCLUDEDIR）
bash cpp/build_scripts/build_sentencepiece_aarch64.sh
bash cpp/build_scripts/build_aarch64.sh

# 上板 + 验证
bash scripts/deploy/push_full.sh                 # 推 output/ → pi@10.8.18.118
# 板上：cd ~/output && bash run_web_rk.sh
#      ./rknn_check models/siglip_vision_fp16.rknn scripts/golden/raw   # NPU golden
#      bash scripts/e2e/e2e_add_rule.sh   # 注入近必中英文规则看全链路
#      bash scripts/e2e/e2e_finalize.sh   # 清理并恢复
```

阈值环境变量：`SIGLIP_DEFAULT_THRESHOLD`（规则未写 threshold 时生效，默认 0.15）。
调试捕获 `DEBUG_CAPTURE=1` → web `/debug` 页看每个行人的逐规则 cosine 明细。

## 已知边界

- 规则必须英文（模型训练分布 = PEDES 风格英文逗号属性短句）；中文输入走 byte-fallback
  产生乱码 token，分数自然塌缩（不漏报变误报，但会漏报）。
- SigLIP 绝对分值低，规则文本写错时几乎必然不告警（漏报友好、误报不友好，与 BGE 相反）。
- 视觉塔 w8a8 量化未做（fp16 golden cos≥0.9998 已够）；文本塔在 CPU 仅规则编辑时触发，无需转 NPU。

进展与踩坑记录见 `PLAN.md` 末尾「进展记录」。
