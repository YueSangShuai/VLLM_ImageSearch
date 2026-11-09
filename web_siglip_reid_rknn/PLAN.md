# web_siglip_reid_rknn：SigLIP2 行人图文直接匹配（RK3588 NPU）

框架复刻自 ../web_vllm_rknn（YOLO→裁剪→嵌入→匹配→web 告警），推理核心替换：

| 旧（web_vllm_rknn） | 新（本工程） |
|---|---|
| Qwen3-VL-2B（rkllm）生成中文描述 | 不再描述：裁剪图直接进 SigLIP 视觉塔 |
| BGE-small-zh 文本嵌入 + 描述/规则 cosine | SigLIP 文本塔嵌入 + 图/文嵌入 cosine（同一空间） |
| 符号/属性支线（attr_head，绑定 Qwen 嵌入） | v1 移除 |

- 模型：MarketaJu/siglip2-person-description-reid（SigLIP2-base-patch16-224 微调，
  CUHK/ICFG-PEDES、PETA 等训练；safetensors 即纯 SiglipModel 权重，repo 里的
  ReSiPVD fc 降维头**未发布权重**，不用它——直接 768-d 归一化 cosine）
- 规则语言：英文逗号分隔属性短句（训练分布即如此），如
  "person, black upper clothing, carrying bag"
- 分值标定（2026-11-05，PETA 40 图实测）：正确属性 0.15–0.22，错误属性 ≈0 上下，
  与 BGE 的 0.35/0.55 尺度完全不同 → DEFAULT 阈值降到 0.10–0.15，逐规则校准。

## 模块映射（沿用原框架缝）

| 原模块 | 处理 |
|---|---|
| detector/frame/json_store/util/monitor/stream_processor/server | 原样复用（YOLO RKNN 已上板） |
| vlm_describer / rknn_image_enc / attr_head / attributes.h | 删除 |
| text_embedder（BGE+WordPiece） | 改写：SigLIP 文本塔 ONNX（ORT-CPU，板端有先例）+ sentencepiece（Gemma）分词 |
| matcher（BGE 描述×规则 + 符号支线） | 简化：视觉嵌入 × 规则文本嵌入 cosine + threshold/score_diff/interval 保留 |
| 新增 | siglip_vision.{h,cpp}：视觉塔 RKNN 封装（仿 detector.cpp 的 rknn C API 模式，224×224，input norm mean/std=127.5） |

分工理由：视觉塔每帧每人都要跑 → NPU；文本嵌入只在规则保存/变更时算 → 板端 CPU
（ORT fp32，12 层 seq64 毫秒级），256k 词表 embedding 表完全不进 NPU。

## 阶段

### A 主机侧（x86）
- [x] A1 摸底：模型/tokenizer 下载（hf-mirror）；PETA 实测判别信号正常
- [x] A2 ONNX 导出：vision(pixel[1,3,224,224]→emb[1,768]) / text(input_ids+attn[1,64]→emb[1,768])，
      L2 归一化烘进图；golden npy；ORT vs torch ≤1.5e-06
- [x] A3 闸门：rknn-toolkit2 2.5.0 fp16 转换成功（183MB）；x86 模拟器 cos≥0.9998、8×4 排序保持；
      w8a8 未做（fp16 已达标）；文本塔不转 NPU（仅规则编辑时算，板端 ORT-CPU int8=269MB，
      golden cos≥0.992；fp32 1077MB 板端 available 2.8GB 放不下）

### B C++ 改造
- [x] 拷贝 cpp/，删 VLM 三件套
- [x] SiglipVision（RKNN C API）+ 预处理（裁剪直接 resize 224×224，RGB）
- [x] sentencepiece aarch64 交叉编译（sdist 源码，自含依赖），tokenize golden 对拍
- [x] text_embedder 重写（SigLIP 文本塔，padding=max_length 64，pad_id=0，无 bos——以
      transformers 实测 ids 为准）
- [x] matcher 重写为纯 cosine；symbolic/attr 支线删除；debug 快照字段调整
- [x] server/static：debug 页展示 cosine 明细；规则页文案改英文示例
- [x] config：SIGLIP_* env（模型路径、默认阈值、NPU core）

### C 板端
- [x] aarch64 交叉编译（沿用原 toolchain/rknn 头库）
- [x] bench：yolo + siglip_vision(rknn) + siglip_text(ort) 延迟基线
- [x] 部署 + 英文规则 e2e + 阈值校准（初标完成；「背包」规则精度差，待规则文本再调）

## 风险
- 视觉塔 SigLIP2 注意力池化头（probe/attention pooling）算子可转性 → A3 闸门定生死
- 分值尺度小（0.1 量级）→ 阈值/冷却参数全部重标；负样本 ≈0 分意味着规则文本写错时
  几乎必然不告警（漏报友好、误报不友好，与 BGE 行为相反）
- Gemma 分词 byte-fallback 对中文规则的行为：英文路线下基本不触发，中文输入会走
  byte-fallback 出乱码级 token（分数自然塌缩，不会误匹配）

---

## 进展记录

### 2026-11-05
- 建档。模型下载至 models/（reid 权重 1.5GB fp32 + google/siglip2-base-patch16-224 取
  tokenizer/preprocessor；两个 safetensors key 集合相同、hash 不同 → reid 权重确实微调过）
- 关键发现：repo 的 modeling_resipvd.py 引用 text_fc/visual_fc 但权重未发布 → 按纯
  SiglipModel 加载即可，README 的 usage 亦然
- PETA 实测（rkllm env, transformers 5.8）：REID 排序正确、BASE 近随机；分值 0.1x 量级
- 环境坑：`conda activate rknn && python` 会被 PATH 劫持到 py3.9 系统 python，
  必须用 /home/yuesang/miniconda3/envs/rknn/bin/python（py3.10.22, toolkit2 2.5.0 OK）

### 2026-11-05（B/C 收尾）
- B/C 全部完成：三交叉编译产物（web_siglip_reid / bench_models / rknn_check）已部署
  pi@10.8.18.118:/home/web_siglip_reid_rknn，服务常态运行。sp 源用 GitHub v0.2.0 克隆
  （src/CMakeLists.txt 本地修补 CMAKE_INSTALL_INCDIR→INCLUDEDIR）
- bench 基线（板上空 NPU，iters=5）：YOLO 79.2ms(12.6fps) / SigLIP 视觉 144.7ms(6.9fps)
  / SigLIP 文本 68.1ms/条(CPU)；线上每人 ≈ 79+145 = 224ms（对比旧 VLM 链路秒级/人）
- e2e 烟雾规则（threshold=0.005）14:40 起全天在跑：命中 31/75 人，链路通（告警图/日志/web 全对）。
  已 finalize 清理恢复
- 真实场景分值标定（debug_log 75 行人）：分值比 PETA 压缩约一半——「黑色上装」p50=0.055
  max=0.166、「背包/提包」p50=0.024 max=0.137、烟雾规则（泛描述）max 也才 0.173。
  阈值 0.12 在负样本 p50(≈0.05) 之上、仍可达 → 维持 0.12，不建议再降
- 告警人工核验（11 条 / 9 人）：「黑色上装」8 中 7 真（1 条条纹衫误报，光照偏暗）；
  「背包/提包」3 条全不成立（手持手机/背面视角）——低分辨率监控裁剪下 bag 概念不可分，
  该规则要改用更显著的表述（如 wearing backpack）或弃用
- bench 清理旧残留：汇总标签「BGE 文本向量」→「SigLIP 文本向量」；单条计时样本中文→英文
  （中文走 byte-fallback 会测出偏高的分词耗时）
- 部署通道备忘：ssh_pi.py/scp_pi.py 需要 pexpect，只有 py11/Qwen env 有；
  板上中文路径 scp glob 会挂死，先远端 tar 再 pull

### 2026-11-05（同步平台新功能：负样本门）
- ../web_vllm_rknn 平台 11-04 上了负样本门（`pos ≥ threshold && pos > max(neg)+score_diff`），
  本工程 matcher 移植时漏掉——stream_processor/告警日志字段一直在读 negative_* 但恒为空。
  逐文件比对确认其余共享模块（server/stream/monitor/detector/debug.html）fork 时已带上
  平台最新改动，唯 matcher 缺此项
- matcher.{h,cpp}：updateRules 同口径编码 negative（SigLIP 对称空间、无 query 前缀）；
  打分加负样本门与 neg_blocked 判定；调试明细补 negative_list/negative_scores/
  max_negative_score、新 status `negative_higher`；告警 json 补 negative_higher（web 规则页
  负样本输入框本就在共享模板里，自此真正生效）。无 negative 时行为与旧版逐位一致
- 重编译上板（md5 bf905115）重启，服务正常、电梯厅设备启动加载 2 条规则；
  PUT /api/tasks → refreshRules → updateRules 链路带 negative 往返验证无副作用；
  FP 压制实录待相机恢复（三路 RTSP 全天读超时，验证窗口内无任何行人帧）
- 顺带修活一处暗伤：上一轮 e2e_finalize.sh 从 tasks.json.bak_e2e 还原的规则快照 device_id
  是空串（备份早于建任务写 device_id 的版本）→ 设备 matcher 0 规则、/debug rule_details
  全空、web 规则页看着有规则但永不打分。已修板上 tasks.json（device_id 补回
  a4cfe0d3-…）并 curl 探针证明 PUT 本身无损。**教训：finalize 之后必须核对 tasks.json**
- 待办：相机恢复后在规则页给「黑色上装」加 white upper clothing 类负样本，
  观察 /debug 出现 status=negative_higher 即闭环
