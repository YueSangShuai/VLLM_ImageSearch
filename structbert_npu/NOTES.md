
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

## LFM2.5-VL-1.6B vs gecko-guard-1b 三方对比（2026-11-24，宿主 A800）
同源 5 张 A/B 行人图、desc/guard 双 prompt、greedy、max_new_tokens=128（lfm/ 目录）：
- **质量-desc（线上口径）**：LFM 胜出。输出简洁无复读，还能多抓属性（002 眼镜、001 靴子）；
  gecko 在 desc prompt 下退化成 `<|TAG|>` 复读循环（老问题，见结构化罚分条目）。
  LFM 唯一硬伤：004 把浅紫短裙说成"白衬衫+深色裤子"（gecko 对）。
- **质量-guard**：gecko 胜。严格 `<|TAG|>` 格式 + 风险分层（低/中）；LFM 无 `<|TAG|>` token、
  输出"行为描述/穿着描述/风险等级"自然格式，且 5/5 全判"中等"——风险无区分度，不能直接替。
  共同亮点：都发现 003 打电话。
- **RKLLM w8a8 大小**（rk3588/3核/normal/max_context=1024，LM 塔拆塔直转）：
  gecko（Qwen2.5-0.5B）788MB vs LFM（Lfm2ForCausalLM 1.17B）**1454MB（1.35GB），1.84×**。
  意外收获：rkllm-toolkit 1.3.1 loader.so 支持列表含 **Lfm2ForCausalLM**（strings 实锤），
  拆塔（safetensors 键 model.language_model.*→model.*，lm_head 绑定 embedding）后直转成功，
  全程 ~12 分钟，产物 lfm/lfm25vl-1.6b_w8a8_rk3588.rkllm。校准集同款配方。
- **宿主机速度**（同图 001+desc，warmup 后 greedy 64tok，bf16）：
  prefill LFM 33ms vs gecko 48ms；**decode LFM 7.0ms/tok(142 tok/s) vs gecko 23.6ms/tok(42 tok/s)，3.4×**；
  e2e 0.48s vs 1.53s（LFM 的 LFM2 混合 conv 架构红利）。
  ⚠ 宿主机 GPU 数≠板端 NPU：LFM2 conv 层在 NPU 效率未知；且视觉塔是 SigLIP2 **NaFlex 变分辨率**
  （tile 切图+缩略图，动态 shape 转 RKNN 是大坑），gecko 的 InternVL 448 固定分辨率已板端验证。
  板端实测待做。脚本：lfm/host_baseline_lfm.py、lfm/bench_host.py、lfm/export_rkllm.py。

## 板端真实 ROI 7 图 gecko vs LFM2.5-VL 对照（2026-11-25，宿主 bf16）
按规则去重各取最新 roi（1F Inside East Door/货梯口/一楼电梯厅共 7 条规则），
desc/guard 双 prompt，产物 gecko/baseline_gecko_board.jsonl、lfm/baseline_lfm_board.jsonl。
肉眼对照结论（拉取时按 日期__规则__文件 命名，规则名即人肉标注）：
- **desc**：LFM 7/7 无复读、20~40 字短句；多抓包/手推车/衣服图案等细节，但「戴眼镜的女生」
  漏了眼镜。gecko 2/7 复读退化（Reranker验证、黑衣服男生），不退化时对且颜色更准（酒红 vs 泛称红）。
- **guard**：gecko 依旧占优：格式稳、风险有低/中分层；LFM 7/7 全「中等」且理由牵强
  （"穿着不太正式→中风险"），行为描述空泛"似乎正在移动"，128 token 常被打满截断。
- gecko guard 也有一处幻觉（戴眼镜女生"疑似携带宠物"，实际手里是食物盒+包）。
- 板上 alerts_history 未拉到（scp 目录失败），用文件夹名代替人肉标注，够用。
- 板端图拉取坑：fetch_board_images.sh 的远端 find|awk 会被二次展开炸掉，
  改成远端纯 find + 本地 awk 去重；含空格路径 scp 拉不动，先板上 cp 到 /tmp/roipull 再拉。

## 板端 LLM 塔三模型测速：gecko-LM vs LFM-LM vs Qwen3-VL-2B（2026-11-25，RK3588 3核 w8a8）
lfm/gen_bench.cpp（PROMPT 模式+主机预渲染 ChatML、greedy、64 token 上限、GENERATE 模式读 perf）。
三塔同款 desc 文本 prompt（token 数不同：76/57/39），在线服务全程在跑（NPU 共享，偏保守口径，
同 Qwen3-Reranker 基准做法）。板上文件 /home/pi/rrbench/。
| 模型 | 参数量 | 模型文件 | prefill | decode | mem |
|---|---|---|---|---|---|
| gecko-LM (Qwen2.5-0.5B) | 0.5B | 788MB | 453ms/76tok | **101ms/tok (9.9 tok/s)** | 657MB |
| LFM-LM (Lfm2 1.17B) | 1.2B | 1454MB | 271ms/57tok | **95ms/tok (10.4~10.9)** | 1222MB |
| Qwen3-VL-2B | 2B | 2376MB | 202ms/39tok | **92ms/tok (10.9)** | 2056MB |
结论：
- **decode 三者几乎一样（9.9~10.9 tok/s）**。宿主机 A800 上 LFM 快 gecko 3.4× 的红利在 NPU 上
  完全消失——LFM2 的 conv 线性注意力靠带宽/融合取胜，NPU 上按算力摊，反而没吃到；
  2B 也只比 0.5B 慢 10%（说明共享 NPU 下瓶颈是排队不是模型本身）。
- prefill 按 prompt token 数近似线性，参数量影响很小（在线场景加 256 视觉 token 后差距会拉开）。
- 与官方空闲 NPU 表（benchmark.md，opt_level=0：Qwen2-0.5B 41.6 tok/s、Qwen3-VL-2B 15 tok/s、
  LFM2.5-VL-450M 62 tok/s）差 3~4×，主要是服务共抢；若独占测速需先停 web_vllm_rknn。
- 三模型可共存加载无需停服务（mem 峰值 2GB/8GB 总），2B 权重走 mmap。
- ⚠ 本测速只测语言塔；端到端还要各加视觉塔（gecko=InternViT RKNN 未上板、LFM=NaFlex 未解决、
  Qwen3-VL=已上线同款）。
