# web_vllm_rknn_reranker 变更日志

> 项目目标：复制 [web_vllm_rknn](../web_vllm_rknn/)（RK3588 行人布控平台），把图文匹配从
> 「VLM 描述 + BGE cosine」整体替换为 **Qwen3-VL-Reranker-2B** 直接对 (ROI 图, 规则文本) 打分。
> 板端全链路（reranker 转 .rknn/.rkllm 跑 NPU）、复用 C++ 业务层、端口 5005。
> 板端基础设施/固件/部署的历史变更见原项目 BUILD_CHANGES.md。

## 2026-10-29

- 新建项目：从 web_vllm_rknn 复制 cpp/（排除 build/）、scripts/、文档；rknn-llm/、YOLO 权重软链复用；
  校准集 `calib/alerts_images/`（273 张线上告警 ROI，即 match() 实际输入）+ `detection_log.json` + `alerts_history/`（旧管线分数，可做同图新旧对照）。`output_cpp/data/` 起空 tasks。
- 下载 Qwen3-VL-Reranker-2B（hf-mirror，4.3GB）→ `FG-CLIP/models/Qwen3-VL-Reranker-2B/`。
  架构 = Qwen3VLForConditionalGeneration（与板上 Qwen3-VL-2B 相同），分数 = sigmoid(logit_yes − logit_no)，
  yes=9693 no=2152。官方脚本 prompt：system「Judge whether the Document … only "yes" or "no"」+
  user「<Instruct>: … <Query>:文本 <Document>:图像」+ assistant 生成前缀（渲染样例 `calib/prompt_sample.txt`）。
- `scripts/host_ref_rerank.py`（A800, Qwen env）：产出 `calib/host_scores.json` 268 对参考分 +
  `calib/yes_no_ids.txt` + `calib/rerank_yes_no_diff.bin`（W_yes−W_no，hidden 兜底模式用）。
  结论：正例 vs 跨规则负例分离方向正确（黑衣服规则 mean 0.51 vs 0.29），但 sigmoid 压缩明显
  （全局 0.2~0.7），板端阈值必须按新分布重标，且 w8a8 量化后会进一步偏移——一致性验证只看排序。
  ⚠️ 坑：Qwen env 的 qwen-vl-utils 0.0.2 缺 `image_patch_size` 参数，官方脚本会静默退化为**无图打分**
  （分数全 ~0.46）；升级到 0.0.14 后正常。
- 视觉塔导出（rknn env）：`export/export_vision.py --model_name=qwen3-vl --path <reranker>` → ONNX
  （与旧基座 ONNX 同构，1.63GB）→ `export_vision_rknn.py` → 855MB .rknn（与板上现有基座塔同大小，正常；
  33MB 那是 rkllm，勿再误读）。归档 `models/rk3588/qwen3vl-reranker_vision_rk3588.rknn`。
  工作目录 `export_reranker/`（export/ 软链到 rknn-llm demo，不污染旧项目产物）。
- 量化校准（`scripts/make_reranker_calib.py`）：真实行人 ROI 缩 448×448 + reranker 格式 prompt，
  hook language_model 前向捕获 inputs_embeds，21 条 ~267 token → `export_reranker/data/llm_inputs*`。
  ⚠️ 首版 bug：deepstack_visual_embeds 列表内层张量未搬 CPU，rkllm build 报
  "Attempting to deserialize object on a CUDA device"；已改递归 to_cpu。
- 语言塔 .rkllm 导出（rkllm env，CWD /tmp/rerank_export 避开 xfs）：
  `export_rkllm.py --target-platform rk3588 --num_npu_core 3 --quantized_dtype w8a8 --device cpu`，
  校准集需实体拷贝到 /tmp（软链会带 xfs/设备问题）。首次构建在校准加载步失败（上述 CUDA 张量问题），修后重跑。
- C++（进行中）：新增 `reranker_scorer.{h,cpp}`（仿 vlm_describer，run 时
  `infer.mode=RKLLM_INFER_GET_LOGITS`，取末 token yes/no logits 差出 sigmoid；兜底 hidden 模式用
  diff 向量点积，tie_word_embeddings 下数学等价）；matcher 增加 match_mode=reranker 分支
  （max over pos 文本 → final_score，threshold/score_diff/interval/告警字段不动，跳过描述/BGE/二次核验）；
  config 新增 RERANKER_* 环境变量组。图片占位默认沿用「文首 `<image>`」既有约定，
  `RERANKER_IMG_MID=1` 可试 `<Document>:` 后插（板上 demo 决定）。

- C++ 全部接线完成并通过 x86 `-fsyntax-only` 检查：
  - `reranker_scorer.{h,cpp}`：GET_LOGITS 取末 token yes/no 两 logit（回调里即拷，防回收）；
    hidden 兜底 = 末 hidden · (W_yes−W_no)（bin 自动探测 exe 同目录/RERANKER_WEIGHT_BIN）；
    帧级视觉缓存（resize 后 RGB memcmp 命中免跑视觉塔）；`RERANKER_IMG_MID` 控图占位文首/文中。
  - `matcher.cpp::matchReranker`：pos 文本逐条打分取 max→final_score（≤RERANKER_MAX_POS_TEXTS 条），
    负样本默认关；threshold/score_diff/interval/告警字段与向量模式同语义；单条打分失败整帧放弃等下帧。
  - `monitor.cpp` 按 mode 惰性加载（reranker 模式不建 VLM/BGE）；`config` 新增 RERANKER_* 组。
  - `tools/rerank_eval.cpp`：喂 host_scores.json 板端重打分（同图连续打分吃缓存，延迟只计纯 prefill）；
    `scripts/compare_scores.py`：spearman + 规则级 pos/cross 分离 + 分位数。
  - `build_aarch64.sh`：新增 rerank_eval 组装；**data 非空不再被 web_vllm_cpp/data 覆盖**（旧坑硬性修复）；
    reranker 双件套 + 权重 bin 软链进包。`push_reranker_full.sh`（板端独立目录
    /home/pi/output_cpp_reranker，不碰 5003）/`push_reranker_update.sh`（二进制+run 脚本带 md5）。
  - `run_web_rk_reranker.sh`：端口 5005，VLM_MATCH_MODE=reranker，instruction 默认与 host_ref 一致。

## 待办
- [ ] rkllm 导出完成 → 上板 logits demo（GET_LOGITS + 多模态可行性，不行切 hidden）
- [ ] 板端 rerank_eval vs host_scores.json 排序一致性 + 单条 prefill 延迟
- [ ] E2E（5005）加规则验证告警 + 阈值按板端分布重标
