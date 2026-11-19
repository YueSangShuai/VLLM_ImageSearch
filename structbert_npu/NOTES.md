
## SigLIP2-reid 文本塔做「文本×文本」判定：不可行（2026-11-13，服务器 ORT）
用 web_siglip_reid_rknn/onnx/siglip_text(_int8).onnx（Gemma 分词、pad64、mask=ids≠0，
线上同款口径）对同一 6 案做 规则×描述 向量 cosine，中英文双口径：
- 中文：正 {0.736,0.791,0.817} vs 负 {0.717,0.764,**0.825**}，全重叠；
  最错的「白色连衣裙」(0.825) 反而全场最高。
- 英文（线上规则格式）：正 {0.803,0.875,1.000} vs 负 {0.781,**0.980**,0.870}，同样重叠
  （「蓝色上衣」0.980 > 稀释正样本 0.875）。
- int8 与 fp32 漂移 ≤0.039，量化不是原因。
原因：SigLIP 文本塔只被训练为「投影到图像空间」的编码器，文本↔文本几何未校准：
所有文本挤在一个窄锥里（cos 全 >0.7），既无蕴含方向也无属性绑定。
**结论：文本塔只能干它线上干的事（图像×规则）；文本×文本匹配请走 Qwen3-Reranker 路线。**
脚本 scripts/siglip_text_score.py。至此对照系齐：StructBERT(稀释崩) / bge-base(4/6) /
Qwen3-R-0.6B(6/6) / SigLIP文本塔(无判别力)。

## Qwen3-Reranker-0.6B 转 RKLLM w8a8 上板实测（2026-11-18）
转换：rkllm-toolkit 1.3.1（rkllm env），rk3588/3核/w8a8/optimization_level=1，
校准集=线上真实规则+6案中文（structbert_npu/rkllm/export_reranker.py），max_context=512，
产物 930MB，CWD 须放 /tmp（ext4，PLAN.md 老坑）。全程约 3 分钟。
板端 bench：rkllm/rerank_bench.cpp（RKLLM_INPUT_TOKEN 直灌主机预渲染 token，绕开
runtime 模板；GET_LOGITS 取末位 logits softmax(yes/no)；perf 直报 prefill_ms）。
运行文件 /home/pi/rrbench/，librkllmrt 1.3.1 复用板上服务目录的。
速度（RK3588，注意：web_vllm_rknn 服务全程在跑，CPU 限核 4-7，NPU 共享——偏保守）：
- 加载 3.2s（一次性）；首对 prefill 1012ms（冷），稳态 **146~276ms/对（均值 231ms）**，
  每对 ≈110 token，mem 909MB。
- 对照 SigLIP 文本塔（板 CPU 实测 bench_models --skip yolo,vision）：
  fp32 73.0ms/条、int8 68.6ms/条（64 token；ORT int8 对 transformer 几乎无收益）。
  → **Qwen3-R w8a8 ≈ SigLIP 文本塔的 3.2~3.4 倍耗时，仍在百毫秒级**。
- 服务器 CPU 参考：bf16 ~2s/对，w8a8 NPU 提速 ~9x。
**量化保真：6/6 判定不变**，P(yes)={0.046,0.9947,0.9989,0.5762,0.0255,0.0009}
vs 主机 bf16 {0.090,0.9979,0.9995,0.7058,0.0059,0.0007}；正最低 0.576 略降但仍 >0.5，
负最高 0.046，阈值 0.5 依旧干净（最脆的「短袖」案余量从 0.71 收窄到 0.58，留意）。
板端服务未受影响（NPU 可多上下文共存，无需 pkill 生产服务）。

## Qwen3-Embedding-0.6B 边界案例（2026-11-18，同 6 案）
口径：官方 query 指令式 last-token pooling + L2 归一化，cos(query带指令, doc)。
正 {0.706(短袖), 0.762(稀释案), 0.884(裸句)} vs 负 {0.363, 0.639(红), 0.644(蓝)}
→ **6/6 可分，阈值 ∈ (0.644, 0.706)，取 0.675**。
但 margin 只有 0.062（对比 Reranker 的 P=0.576 vs 0.046 近乎两个数量级）：
红/蓝颜色冲突案（0.639/0.644）只比阈值低 0.03~0.04，比最弱正样本「短袖」0.706
低 0.06 —— 新案例一翻车就整片翻。且 0.644 是这批负样本的 max，样本一多必重叠。
工程优势：规则向量可预计算缓存（每帧只需编码描述 1 次，NPU 上 ≈231ms/人，
是 Reranker 每对都要过一遍的 1/2），还免费带检索。
结论：检索粗筛可用 Embedding（阈值 0.675 起步、随告警数据校准），
**终判仍应上 Reranker**；单模型二选一的话选 Reranker（margin 决定泛化）。
脚本 scripts/qwen3_embed_score.py。至此五方对照闭环。

## BGE 初筛 + Reranker 复筛级联成本（2026-11-18）
bge-small-zh 同 6 案（scripts/bge_prefilter_check.py）：正 {0.606,0.626,0.774}
负 {0.376,0.609,0.632}，正负重叠（当终判不可），但**作召回筛合格**：
阈值压到 0.576 以下即 6/6 不漏，本批边界负样本只杀 1/3——
它的价值在砍掉「与描述无关」的规则长尾（白裙级 cos≈0.37），边界案照放。
成本模型（板端实测值）：BGE 编码 1 条描述 7.2ms/人（CPU，规则向量缓存）
+ Reranker 231ms×候选数 K（NPU）。R 条规则纯 Reranker 需 231×R；
6 规则场景 BGE 一般放 K≈2~3 → 470~700ms/人，比纯 Reranker（≈1.4s）省 50~65%。
BGE 在 CPU、Reranker 在 NPU，天然并行，BGE 那 7ms 完全被 NPU 排队遮蔽。
阈值定法：取「校准集 TP 最低分 − 安全边距」，宁可放错不可漏（漏检不可逆，误放行只多一次复筛）。
