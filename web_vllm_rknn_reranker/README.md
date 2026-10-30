# web_vllm_rknn_reranker — 行人布控平台 RK3588 版，图文匹配 = Qwen3-VL-Reranker-2B（板端端口 5005）

[web_vllm_rknn/](../web_vllm_rknn/README.md) 的平行实验分支：业务层（设备/规则/告警/前端）完全复用，
把匹配引擎从「VLM 描述 + BGE cosine」替换为 **Qwen3-VL-Reranker-2B**：
YOLO 裁剪 ROI 后，(ROI 图, 规则文本) 直接过 reranker（RKNN 视觉塔 + RKLLM 语言塔，NPU 全链路），
取末 token yes/no logit 差 `sigmoid(l_yes − l_no)` 作相关性分；`final_score = max(positive 文本分)`，
threshold / score_diff / interval / 告警落盘逻辑不变。

## 与主项目的差异

- `match_mode=reranker`（新增第五种模式；改回 `combined` 即回退旧链路，需同时配 VLM/BGE 模型路径）
- reranker 模式下不加载描述 VLM 与 BGE（省内存），也不做二次核验
- 新增环境变量：`RERANKER_VISION_PATH` `RERANKER_RKLLM_PATH` `RERANKER_MODE`(logits|hidden)
  `RERANKER_INSTRUCTION`(改了必须重跑 host_ref) `RERANKER_MAX_POS_TEXTS`(默认2) `RERANKER_NEG`(默认0)
  `RERANKER_IMG_MID`(图片占位文首/`<Document>:`后) `RERANKER_YES_ID/NO_ID` `RERANKER_WEIGHT_BIN`
- 阈值语义不变但**分布不同**：reranker 分数全局压缩在 ~0.2~0.7，须按 `calib/host_scores.json` 重标

## 所需模型

| 模型 | 用途 | 获取 |
|---|---|---|
| Qwen3-VL-Reranker-2B | 匹配引擎源权重 | `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-VL-Reranker-2B --local-dir ../models/Qwen3-VL-Reranker-2B` |
| qwen3vl-reranker_vision_rk3588.rknn + qwen3vl-reranker-2b_w8a8_rk3588.rkllm | 板端双件套 | 本仓库转换产物 `models/rk3588/`（流程见下） |
| yolov8n_640_rk3588.rknn | 行人检测 | 软链复用主项目 |
| rknn-llm v1.3.1 | 转换工具链 | 软链 `rknn-llm -> ../web_vllm_rknn/rknn-llm` |

## 转换流程（x86，两个互斥 conda 环境，细节见主项目 INSTALL.md）

```bash
# 0) 主机参考分数 + prompt/yes-no-id/权重导出（Qwen env: transformers>=4.57, torch 2.8, qwen-vl-utils>=0.0.14）
python scripts/host_ref_rerank.py                     # → calib/host_scores.json 等
# 1) 视觉塔（rknn env，工作目录 export_reranker/，export/ 为 demo 软链）
python export/export_vision.py --path <reranker ckpt> --model_name qwen3-vl --height=448 --width=448
python export/export_vision_rknn.py --path onnx/qwen3-vl_vision.onnx --model_name qwen3-vl
# 2) 量化校准：真实行人 ROI(448) + reranker prompt（Qwen env, GPU）
python scripts/make_reranker_calib.py                 # → export_reranker/data/llm_inputs*
# 3) 语言塔（rkllm env；CWD 必须在 ext4 如 /tmp/rerank_export，校准 pkl 需实体拷贝）
python export/export_rkllm.py --path <reranker ckpt> --target-platform rk3588 --num_npu_core 3 \
    --quantized_dtype w8a8 --device cpu
```

## 构建与部署

```bash
bash cpp/build_scripts/build_aarch64.sh               # 组装 output_cpp/（data 非空不覆盖；模型软链）
bash scripts/deploy/push_reranker_full.sh             # 首次整包 → 板 pi@<IP>:/home/pi/output_cpp_rknn...
bash scripts/deploy/push_reranker_update.sh           # 小更新：二进制×2 + run 脚本，带 md5
# 板上：cd ~/output_cpp_rknn && bash run_web_rk_reranker.sh   （端口 5005）
# ⚠️ 与 5003 服务共用 NPU：压测/跑分时先停另一个（pkill 写成脚本推上去跑）
```

## 验证工具

```bash
# 板端对 273 张 calib ROI 重打分（logits 可行性 + 延迟 + 一致性一箭三雕）
RERANKER_VISION_PATH=... RERANKER_RKLLM_PATH=... ./rerank_eval --list calib/host_scores.json --out board_scores.json
python scripts/compare_scores.py calib/host_scores.json board_scores.json   # spearman + 规则级分离 + 分位数
```

## 已知要点

- prompt 必须与 host_ref 渲染逐字一致（`calib/prompt_sample.txt`）；instruction 改了分数全部作废
- runtime 若不支持多模态 GET_LOGITS：`RERANKER_MODE=hidden` + `rerank_yes_no_diff.bin`（与 HF 数学等价）
- qwen-vl-utils <0.0.14 时官方脚本会静默无图打分（分数全 ~0.46），务必核对版本
