# BGE 升级计划：bge-small-zh-v1.5 → BAAI/bge-large-zh-v1.5（非对称指令版）

> 目标：把平台文本向量模型从 bge-small-zh-v1.5（512 维）升级到 bge-large-zh-v1.5（1024 维），
> 并按 BGE 官方**非对称检索**用法适配：query 侧加指令前缀，passage 侧不加。
> 范围：先只动 web_vllm_rknn（RK3588 版）；x86 版 web_vllm_cpp 验证通过后按需跟进。

## 0. 现状（已探明）

| 项 | 现状 |
|---|---|
| 模型 | `output_cpp/models/onnx/bge-small-zh.onnx`，ONNX 输入 input_ids/attention_mask/token_type_ids（dynamic batch/seq），输出 last_hidden_state 512 维 |
| 推理端 | `cpp/src/text_embedder.cpp`：WordPiece + CLS pooling + L2；`dim_` 从模型输出动态读（text_embedder.cpp:197），**升 1024 维代码零改动**（.h 里 `int dim_ = 512` 只是初值） |
| 向量缓存 | 规则正/负样本向量**不持久化**，refreshRules 时内存重建；tasks.json 只存文本；alerts_history 里存的分数是历史快照，不受影响 → **无需重建向量库** |
| 编码双方 | matcher：帧描述 → `buildEmbeddingText()`（关键词过滤）→ 编码 1 次/帧；规则 positive/negative 短句 → 编码后常驻内存 |
| 分词坑 | sentence-transformers 读 `sentence_bert_config.json` 的 do_lower_case（会覆盖 tokenizer_config.json）——导出/对分以 ST 为准，不用 AutoTokenizer（见记忆 [[bge-st-do-lower-case]]） |
| 工具链 | HF 下载可走代理 10.8.18.5:7890；导出环境用 llama env |

## 1. 非对称指令机制设计

BGE 中文系列的标准用法：**查询加前缀、文档不加**。本平台语义映射：

- **query 侧 = 图像描述**（VLM 输出、经 buildEmbeddingText 过滤后的文本）：
  编码时加前缀 `为这个句子生成表示以用于检索相关文章：`
- **passage 侧 = 规则正/负样本文本**：不加任何前缀（现状不变）

实现落点（只加在 query 侧，一处开关）：
- `cpp/src/matcher.cpp` `match_rules()` 内：`image_emb = emb_->encode_one(BGE_QUERY_PREFIX + filtered_text)`
- 前缀字符串进 `config.h/cpp` 成环境变量项 `BGE_QUERY_INSTRUCTION`（默认上面那句；设空串=关闭，便于 A/B）
- 不在 TextEmbedder 内部加——它不该知道业务语义；bench_models 走同一入口自然覆盖

## 2. 模型获取与导出

1. 下载 `BAAI/bge-large-zh-v1.5`（代理走 hf hub；同时落 `web_vllm_cpp/models/` 一份作两版共用）
2. **分词对齐验证**（防 do_lower_case 坑）：ST encode 一组中文样例 vs HF tokenizer，确认 small 时代结论对 large 同样成立，样例扩进 `golden/bge_golden.json`
3. ONNX 导出（FP32，opset 17，dynamic_axes batch/seq，输入输出签名与 small 完全一致）：
   `bge-large-zh-v1.5.onnx`（FP32，~1.2G）
4. INT8 动态量化（onnxruntime.transformers `--use_int8_quantization` 或 optimum CLI，per-channel， Bert 算子融合）：
   `bge-large-zh-v1.5-int8.onnx`（~350M）
5. **双版本都要**：上板实测后择一定稿（延迟差 ~3-4x，余弦漂移预期 <1e-3）

## 3. 正确性验证（主机侧，先行）

- golden 对齐：用现有 `golden/bge_golden.json` 的文本样例，Python ST(bge-large, 带/不带前缀各一组) 产出新 golden →
  x86 版 bench_models（如可跑）或直接 ONNX vs ST 逐位比对（cos > 0.9999 视为管线对齐）
- 抽样语义冒烟：拿 tasks.json 里 3 条规则 × 描述样例，打印新旧模型正/负样本得分分布，肉眼确认排序合理

## 4. 板端集成与基准

1. 替换文件：`output_cpp/models/onnx/bge-large-zh-v1.5{,-int8}.onnx` 进包；`run_web_rk.sh` /
   `run_bench_rk.sh` 的 `EMBEDDING_ONNX_PATH` 默认改指 int8 版（定稿后改最终名）
2. `push_cpp_full.sh` 推板，`bench_rk_board.sh` 分别测 FP32/INT8 的 BGE 单条编码延迟
   （预期 small 7ms → large-int8 40-100ms；监控节奏 15s/次，占用可忽略）
3. golden 样例在板上 C++ TextEmbedder 输出 vs Python golden cos ≥ 0.999（FP32）/ ≥0.998（INT8）

## 5. 阈值重新校准（关键，不能跳过）

现状 17 条规则 threshold 0.5-0.65、score_diff 0.02 全是按 small 校准的；large 的余弦分布整体偏移。

1. 素材：`output_cpp/data/detection_log.json` + alerts_history 的历史「描述↔命中规则」对
2. 方法：新模型（带前缀 query）对历史正/负样本-描述对全量打分，看正样本分位（P10/P50）与负样本分位（P90/P99）分布
3. 产出一份「旧阈值 → 建议新阈值」对照表写入本文档；先按全局偏移平移，个别敏感规则（低质量类）单独看
4. **过渡策略**：新模型先以 `threshold 全局 -0.05` 宽上线上跑一天观察误报，再逐规则收紧
5. INT8 vs FP32 分别给阈值建议（量化主要压缩高分段）

## 6. 上线路径与回滚

- 回滚保障：small 的 onnx 先在主机改名保留（`bge-small-zh.onnx.bak`），板上只换文件+环境变量即可回退；
  `BGE_QUERY_INSTRUCTION=""` 可单独回退指令侧
- 观察指标：误报率（negative_higher 占比）、漏报用户反馈、BGE 编码延迟
- 板上空间：净增 ~1.0G（14G 余量，够）

## 7. 任务清单（执行顺序）

- [ ] 1. 代理下载 bge-large-zh-v1.5，ST 分词对齐验证（do_lower_case 坑）
- [ ] 2. 导出 FP32 ONNX + INT8 量化版，签名对齐检查
- [ ] 3. 扩 golden 样例（含加前缀 query 组），Python↔ONNX 对齐 ≥0.9999
- [ ] 4. C++ 改动：config 加 BGE_QUERY_INSTRUCTION；matcher.cpp query 侧加前缀（1 处）+ 交叉编译
- [ ] 5. run_web/run_bench 脚本 EMBEDDING_ONNX_PATH 改向；push_cpp_full.sh 上板
- [ ] 6. 板端 bench：FP32/INT8 延迟 + golden 对齐 → 定稿版本，删除未选版本
- [ ] 7. 阈值对照表（历史数据重打分）→ 更新规则 → 观察一天收紧
- [ ] 8. 本文档补记录；PLAN.md 追加条目；（可选）x86 版同步

## 风险与对策

| 风险 | 对策 |
|---|---|
| 1024 维 × 帧循环点积成本 | 规则向量少（每规则 2-10 条短句），点积成本微秒级，可忽略 |
| INT8 漂移致边界规则误报翻转 | 第 6 步双版本实测 + 阈值按版本分别给 |
| 阈值一刀切平移踩坑 | 分位数对照 + 宽阈值过渡观察期 |
| 描述侧加前缀后属性词匹配受影响 | 符号匹配路径（attr_pos/attr_neg）不走向量，天然不受影响 |
