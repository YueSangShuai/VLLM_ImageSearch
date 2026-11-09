# 行人布控系统端侧移植 试验报告与试验进展

> 覆盖范围：`web_vllm`（Python/GPU 版）→ `web_vllm_cpp`（C++ 纯 CPU 版）→ `web_vllm_rknn`（RK3588 NPU 版）三阶段移植试验。
> 撰写日期：2026-11-05。事实依据：各项目内 PLAN.md / PROGRESS.md / BUILD_CHANGES.md / TECH_REPORT.md / golden 数据 / data/ 告警落盘，以及 [TECHNICAL.md](TECHNICAL.md) §7 谱系。

---

## 1. 试验总体概述

### 1.1 试验对象与业务目标

一套「行人布控」业务链路的三阶段端侧移植试验：

```
RTSP 取流 → YOLOv8n 行人检测 → ROI 裁剪 → VLM 生成中文外观描述
→ 与布控规则（中文正/负样本文本）匹配打分 → 告警落盘 + Web 展示
```

匹配触发条件（全谱系统一）：`best_positive_score ≥ threshold 且 > max_negative_score + score_diff`；每条规则带 `interval` 告警冷却。

### 1.2 试验目的与演进动因

| 阶段 | 动因 |
|---|---|
| V1 → V2 | 去除 Python + GPU + 外部依赖（transformers/torch/ultralytics），改为**全链路进程内 C++ 推理、纯 CPU 可跑**，实现"拷仓库→装 cmake/g++/ffmpeg→一条命令编译运行"的换机能力 |
| V2 → V3 | 进一步下沉到 **RK3588 NPU 边缘设备**（NanoPi R6S），验证国产 NPU 工具链（RKNN/RKLLM）承载 VLM 布控全链路的可行性与性能收益 |

### 1.3 三版本谱系位置

```
web (5001, FG-CLIP 图文直匹配, PyTorch GPU)          qwen_classifer (属性头训练 → attr_head.bin)
   │ 业务骨架复用                                          │
   ▼                                                      │
web_vllm (5003, Qwen-VL 描述 + BGE)  ←── 本报告 V1        │
   │ 全量 C++ 移植（REST 契约不变）                         │
   ▼                                                      │
web_vllm_cpp (5003, llama.cpp + ONNX Runtime, 纯 CPU) ←── 本报告 V2
   │ 移植到 RK3588 NPU（llama.cpp → RKNN + RKLLM）          │
   ▼                                                      ▼
web_vllm_rknn (板端 5003, 线上) ←── 本报告 V3 ◄── attr_head.bin
   ├─ 克隆换 Qwen3.5-0.8B → web_vllm_rknn_q35        (板端 5004, 并行实例)
   └─ 匹配引擎换 Reranker → web_vllm_rknn_reranker   (板端 5005, 实验分支)
```

V1 的关键方法论转向（相对更早的 web 版）：从「FG-CLIP 图文向量直接匹配」改为「**VLM 生成中文描述 → BGE(bge-small-zh-v1.5) 编码 → 与规则文本向量匹配**」，把视觉理解交给 VLM、文本匹配留给轻量 embedding。

---

## 2. V1：web_vllm（Python 版，GPU 宿主）

目录：[web_vllm/](web_vllm/)，端口 5003。

### 2.1 技术栈

| 组件 | 实现 |
|---|---|
| Web | Flask + flask-cors，页面 `/ /devices /rules /alerts` |
| 检测 | ultralytics YOLOv8n（COCO person，conf 0.5，帧缩 640） |
| VLM | Qwen3-VL-2B，进程内推理（**不依赖 vLLM 服务，目录名为历史遗留**）；双后端：HF transformers bf16 或 GGUF（llama-cpp-python + mmproj）；默认权重为 LoRA 微调合并模型 `saves/Qwen3-VL-2B-Instruct/lora/train_2026-09-09/checkpoint-17254/lora` |
| Embedding | BGE bge-small-zh-v1.5（sentence-transformers，384 维归一化 cosine） |
| 取流 | 每帧一个 `ffmpeg -vframes 1` 子进程（绕开 OpenCV H.264 解码问题），指数退避重连 |
| 数据 | 纯 JSON 文件（devices/tasks/detection_log/alerts_history/alerts_images），原子写 + 全局锁 |

### 2.2 四种匹配模式（VLM_MATCH_MODE）

1. **structured**：VLM 按 54 个预定义属性（CUHK-SYSU 风格）只输出命中的属性名；规则文本若全由属性名组成则走符号匹配。
2. **description**：VLM 中文自由描述 → BGE → 与规则正/负样本 cosine。
3. **combined**（默认线上配置）：每帧两次 VLM 生成——结构化属性（符号匹配 + 界面展示）+ 自由描述（BGE 向量匹配）。
4. **rule_direct**：启用规则填入 prompt 由 VLM 直判，只输出命中规则名或「不匹配」，不走 BGE（score 恒 1.0）。

生成参数：temperature 0.0、repetition_penalty 1.2、max_new_tokens 320/128/60；tasks.json 实配 threshold 0.55、interval 2s。

### 2.3 试验结果（data/ 告警落盘为唯一量化证据）

| 日期 | 模式 | 告警数 | score 范围 | 备注 |
|---|---|---|---|---|
| 2026-09-23 | combined | 96 | 0.5503–0.6727（均值≈0.599） | 「黑衣服男生」85 条 |
| 2026-09-24 | combined | 346 | 0.55–0.693 | 黑衣服男生 246、长发女生 70 |
| 2026-09-29 | rule_direct | 98 | 恒 1.0 | 黑衣服男生 75、长发女生 20 |

### 2.4 暴露的问题（成为后续版本的试验课题）

- **贴阈值 FP**：告警集中在 0.55–0.60 低分区；典型误报如描述明写「浅紫色短袖」仍被「黑衣服男生」以 0.5524 命中；「长头发的女生」命中的样本属性里派生出 Male。
- **BGE 语义分区分度差**：性别/颜色近义文本 cosine 分差极小，正负样本分都挤在 0.55–0.69，`score_diff` 形同虚设——这是后来 rknn 线引入 attr_head 属性头与 Reranker 直配实验的直接动因。
- **小目标 FP**：09-24 告警中 205/346 为 low_quality ROI（宽 39–58px），VLM 在近不可辨的 ROI 上编造描述。
- **描述幻觉**：推测句（"她看起来像是在购物"）需靠提示词大段禁令压制 + `_clean_vlm_description` 硬剥套话。
- **架构局限**：多设备共用一把 VLM 锁串行推理；无去重/跟踪（仅 interval 冷却）；rule_direct 无分数不可调阈值。
- 无正式 eval 脚本、无 FPS 数据；run.log 空。

---

## 3. V2：web_vllm_cpp（C++17 纯 CPU 版，x86 + aarch64 交叉）

目录：[web_vllm_cpp/](web_vllm_cpp/)，端口 5003（与 Python 版不可并跑）。前端页面原样复用，REST 契约与 Python 版逐字段一致。

### 3.1 组件映射（Python → C++）

| Python 组件 | C++ 方案 |
|---|---|
| Qwen3-VL 描述 | vendored **llama.cpp** libllama + libmtmd 直接链接（f16 GGUF 3.4G + mmproj 819M，CPU 全层，生成加锁） |
| YOLOv8n | **ONNX Runtime CPU**，动态 H/W 导出 ONNX，letterbox/NMS 手写（复刻 Python int 截断语义） |
| BGE | torch 导出 ONNX；C++ 手写 WordPiece 分词 + **CLS pooling**（关键修正，非 mean）+ L2 归一化 |
| cv2 取帧 | ffmpeg 子进程（fork + 超时 kill）+ stb 编解码，不引 OpenCV |
| Flask | cpp-httplib，同路径同 JSON 字段 |
| 存储 | nlohmann/json + tmp 写后 rename 原子替换 |

匹配器四模式 `structured | description | combined | rule_direct`，combined 为默认。

### 3.2 一致性验证结果（与 Python 版对齐）

| 项目 | 方法 | 结果 |
|---|---|---|
| BGE 向量 | `golden/bge_golden.json`（9 条文本 × 512 维） | **9/9 cos=1.000000** |
| YOLO 检测 | `golden/yolo_golden.json`（真实告警帧，bbox [279,1,318,113] conf 0.67495） | **IoU=1.0000，bbox 精确一致，conf 差 0.010** |
| VLM 行为 | rule_direct 冒烟 4 张真实 ROI | 判定全对（含历史 FP 图正确拒判、背身图保守不判）；GPU temp=0 边缘小图非确定翻转属 CUDA 归约核特性（Python 同有），对齐结论以 CPU 冒烟为准 |
| REST 契约 | 静态逐路由对照 + `scripts/rest_ab_test.sh` 动态 A/B 脚本 | 静态全路由逐字段一致；**动态 A/B 未闭环**（PLAN 标 🔄，且 09-29 后 C++ 侧 BGE 短句化修复未回移 Python，跑 A/B 前需先同步） |
| 端到端 | 2026-09-29 实机：3 路 RTSP → YOLO → rule_direct VLM → 告警落盘 | 全通；ROI/original 与原帧像素 MSE 对照证明裁剪/缩放与 Python 一致（排除"旋转 90°"疑云） |

对齐过程中修掉的两个关键数值 bug：YOLO dyn 导出已内置 DFL（坐标解码曾差 8 倍）；BGE 分词 ASCII 累加器漏 flush + `do_lower_case` 实际来源是 `sentence_bert_config.json`。

### 3.3 构建体系演进时间线（BUILD_CHANGES.md）

| 日期 | 里程碑 |
|---|---|
| 09-24 | 12 模块全部完成；CPU + CUDA 双构建通过；Docker 路线受阻（拉不到镜像 + 宿主机 /usr/bin/nvcc 是 11.5）→ 宿主机直编 |
| 09-29 | 端到端实机部署通过（3 路 RTSP）；**同日转向纯 CPU 交付**；BGE FP 修复（matcher 只对描述中的衣着短句做 embedding） |
| 10-13 | **删除全部 Docker 构建线与 GPU 构建线，转 CPU-only**；ONNX Runtime v1.18.1 改源码编译（确立"全源码编译、零预编译二进制"政策）；脚本目录化 `build_scripts/{cpu,aarch64,common,offline}`；aarch64 离线物料方案（~300MB，目标机免联网原生编译） |
| 10-14 | **aarch64 交叉编译完成**（免 root 解包 jammy 交叉 deb，glibc 2.35 对齐 NanoPi R6S；qemu 验证产物依赖零缺失、最高 GLIBC 符号 2.34 ≤ 2.35）；新增 `bench_models.cpp` 模型耗时基准 |

### 3.4 性能与遗留

- 仓库内无实测吞吐数字，只有量级预期：CPU 上 f16 VLM 单帧全链路十秒级（combined 每帧 2~3 次生成）；实测数字在 V3 板端补齐（见 §4.3）。
- **aarch64 交叉产物从未真机验证**：仅到 ELF/GLIBC/qemu 依赖自检层面（`output/aarch64/` 留有 10-15 的 qemu core dump）；该路线随后被 V3 的 rknn 移植取代。
- 与 Python 版数据解耦后两侧 data/ 已漂移；REST 动态 A/B、中间产物清理未收尾。

---

## 4. V3：web_vllm_rknn（RK3588 NPU 版，NanoPi R6S 线上）

目录：[web_vllm_rknn/](web_vllm_rknn/)，板端 NanoPi R6S @RK3588（3 核 NPU，8G RAM），板端端口 5003，当前线上。

### 4.1 平台与环境准备

- 原固件 kernel 5.10 / rknpu 驱动 0.8.2 无法跑 RKLLM w8a8 → 原地只换 eMMC boot-chain 4 分区（不动 rootfs），升至 **kernel 6.1.141 / rknpu 0.9.8**。
- 磁盘长期紧张（根盘 ~5.7G，模型占 6.1G），多次因盘紧受限；11-04 板卡清空后 free 17G。
- x86 侧双 conda env 做模型转换：`rknn`（rknn-toolkit2 2.5.0）与 `rkllm`（rkllm-toolkit 1.3.1）互斥；repo 选型纠正为 **airockchip/rknn-llm v1.3.1**（曾走错 rknn_model_zoo / rknn3-model-zoo）；⚠️ RKLLM 导出目录必须在 ext4（xfs 卷上 O_TMPFILE 预检查误报空间不足）。

### 4.2 推理栈替换

| 组件 | V2（web_vllm_cpp） | V3（web_vllm_rknn） |
|---|---|---|
| YOLOv8n | ORT CPU | **RKNN NPU fp16**，640 固定输入（7MB .rknn；转换须用未裁剪的 6 输出 ONNX，运行时须请求全部 n_output 且 want_float=1，输入 uint8 NHWC + mean/std 烘焙在 convert 期） |
| VLM | llama.cpp GGUF f16 CPU | **RKLLM w8a8，3 NPU 核**：Qwen3-VL-2B 视觉塔 ONNX→.rknn + 语言塔→.rkllm；`vlm_describer` 同名类接口级替换；模型含基座与 ft 微调合并双件套 |
| BGE | ORT CPU | **保留 ORT CPU**（动态 seq 需分桶才能上 NPU，且 7ms/条非瓶颈） |
| 属性 | VLM 生成 | **10-30 新增属性线性头 `attr_head.bin`**（qwen_classifer 训练：冻结视觉塔 + 线性 2048→54，PA-100K 多标签，val mAP 0.7006 / Acc 0.9385；C++ mean-pool + matmul，CPU <1ms），取代 VLM 结构化生成，省约 10s/帧 |

### 4.3 核心性能试验：同板 CPU vs NPU 对照

（脚本 `scripts/bench/bench_rk_board.sh` / `bench_cpu_board.sh` / `bench_vlm_*`，均 flock 互斥 + `pkill -x/-f` 清场后测）

| 组件 | CPU（8 核） | NPU（3 核） | 结论 |
|---|---|---|---|
| YOLOv8n | 加载 224ms / 推理 **145.1ms**（6.89fps） | 加载 104ms / **114.7ms**（8.72fps） | 推理 ×1.27、加载 ×2.2；2.9M 参数不撞算力瓶颈，int8 量化预期还能 ~2× |
| BGE | 单条 ~7ms / batch5 46ms | 未移植 | <1% 链路占比，非瓶颈 |
| Qwen3-VL-2B VLM | llama.cpp f16：**31.7s/120tok**（≈3.8 tok/s），加载 3.8s | RKLLM w8a8：**8.16s/120tok**（≈14.7 tok/s），加载 12s | **端到端 ×3.9**（decode ×3.8）；CPU 加载快 3×（mmap vs 拷 NPU 内存） |
| 端到端单帧 | — | 旧 combined 双生成 ≈16s；属性头上线后（省一次 128-token 生成）**常驻 ≈5.1~5.4s** | VLM 生成一直是唯一瓶颈 |

帧成本拆解（速度矩阵 v2）：ft-2B decode 29.0 tok/s；视觉塔编码 ft ~1.95s vs q35 ~0.85s（×2.3）；文本 prefill 全量仅 **328ms**——"prefill 固定税 5-6s"旧账被证伪，真凶是图像 embedding prefill ~3s + 视觉塔 1.9s + decode。

### 4.4 精度试验

1. **YOLO golden 对齐**：板上 yolo_probe vs `web_vllm_cpp/golden/yolo_golden.json`：RKNN 检出 [279,2,318,113]@0.6709 vs golden [279,1,318,113]@0.6750 —— **±1px / conf ±0.004，通过**（fp16 NPU 漂移，接受）。
2. **量化未破坏 LoRA**：ft w8a8 输出保留训练分布的短描述风格；同 prompt 基座则长篇分点作文。
3. **结构化输出 A/B 定案**（`ISSUE_structured_acc_20261028.md`，六组实验）：板上三个 W8A8 模型（ft-2B / 基座 / q35）都达不到 host bf16 的干净属性短列表；视觉塔 A/B 输出逐字相同→与视觉塔无关；宿主 bf16 同 prompt 也全枚举→**非量化问题，是 prompt 尾部 57 属性列表诱发复读**；去列表更糟（词表锚定丢失）；penalty 扫描：1.2 时 3/6 枚举、**1.4 清零**、1.6 无增益。最终结论：W8A8 结构化必须 penalty ≥1.4（后续该路径整体被属性头取代并于 11-04 删码）。
4. **模型选型对比（ft-2B vs 基座 vs Qwen3.5-0.8B）**：q35-0.8B 基座速度快、遵 prompt（零环境句）、视觉编码 ×2.3 快；10-19 曾误判 q35"线性注意力不吃 NPU、5min/轮"，10-26 证伪（系 demo REPL 空 prompt 死循环假象，喂对 prompt 后 13.5s vs ft 21.5s）。**10-29 起线上默认切为未微调基座**（ft 的中文描述优势可由 prompt 工程替代）。
5. **检索匹配试验**：10-20 "检测到行人但不报警"根因 = 英文 prompt 输出 + 中文规则 BGE 仅 0.33 分（教训：prompt 必须中文）；随后上线非对称检索（规则加 query 前缀）+ 分句噪声过滤。10-30 GTE-ModernColBERT A/B 定案不可用（中文被 BPE 拆碎，top1 16/273 vs BGE 102/273）。11-04 匹配口径改为「规则短句 vs 描述整句单向量」（去分句 max），阈值统一 0.55，并新增 `scripts/eval/text_match_score.py` 评测脚本。
6. **元 token 坍缩**（11-03）：问答/选择式描述 prompt（"判断…输出匹配的规则"）使基座 W8A8 只输出「有匹配/无匹配」三字，BGE 恒 0.27，few-shot 救不回；必须用祈使式生成 prompt。Prompt Cache 路线验证后判废（文本 prefill 仅 328ms 砍无可砍；多模态 save_prompt_cache 静默 no-op；`<image>` 占位符必须在文首否则图像被静默丢弃）。

### 4.5 端到端功能验证

- **10-16 报警链 E2E 打通**：注入必过规则 → 真实摄像头 RTSP → YOLO 检出 → RKLLM 中文描述 → BGE 命中 → 告警落盘，全通（板上验证以 data/ 文件 + REST 为准，web.log 因块缓冲不可信）。
- 11-03「VLM描述为空」事故定位与修复：换部署目录后基座模型未推送 → rkllm_init 启动即败、describer 永久 not-ready、全帧空描述（YOLO/Web 不受影响故 watchdog 不报）；修复后健康判据 = web.log 出现 `VlmDescriber(RKLLM) ready` 且描述有多样性。

---

## 5. 试验时间线总表

| 日期 | 版本 | 事件 |
|---|---|---|
| 09-09 | — | Qwen3-VL-2B LoRA 训练产出 checkpoint-17254（后续三版共用的微调源） |
| 09-22 | V1 | web_vllm 部署修复（LLaMA-Factory v5 导出元数据不兼容，需回贴基座 config）；combined 模式定稿（阈值 0.55） |
| 09-23~29 | V1 | 三轮实跑落盘 96/346/98 告警；确认贴阈值 FP、小目标 FP、描述幻觉三类问题 |
| 09-24 | V2 | C++ 12 模块完成，双构建通过，BGE/YOLO golden 对齐 100% |
| 09-29 | V2 | 端到端实机通过（3 路 RTSP）；转纯 CPU 交付；BGE 短句 embedding 修复 |
| 10-13~14 | V2 | CPU-only + 全源码构建政策确立；aarch64 交叉编译产物通过 ELF/GLIBC/qemu 自检 |
| 10-14~15 | V3 | 立项；基座 + ft 双 RKNN/RKLLM 产物转换全通；板端刷内核（rknpu 0.9.8）后 **RKLLM w8a8 端到端跑通** |
| 10-16 | V3 | C++ 移植板上首跑（5003）；YOLO→RKNN golden ±1px；**报警链 E2E 验证通过** |
| 10-19 | V3 | CPU vs NPU 对照表补齐（VLM ×3.9）；q35-0.8B 初判"不可用"（后证伪） |
| 10-20 | V3 | 英文 prompt 静默杀手定位；非对称检索 + 分句噪声过滤上线 |
| 10-26~28 | V3 | q35 翻案 + 并行平台（5004）；结构化 A/B 全案（penalty 1.4 定案）；pkill/env 泄漏系列运维陷阱修净 |
| 10-29 | V3 | 线上切未微调基座；关 VLM 结构化 |
| 10-30 | V3 | **属性线性头上线**（省 ~10s/帧）；规则条件化描述实验；GTE-ColBERT 判废 |
| 11-02~03 | V3 | 调试捕获上线；「VLM描述为空」事故修复；Prompt Cache 判废；元 token 坍缩教训 |
| 11-04 | V3 | 删结构化死码（match mode 只剩 description/combined）；板卡清空重建部署；描述 prompt 精简为 34 字属性清单版；匹配改整句口径、阈值 0.55 |

---

## 6. 阶段性结论与经验教训

### 6.1 技术结论

1. **NPU 化收益集中在 VLM**：VLM（RKLLM w8a8）端到端 ×3.9 是移植的全部理由；YOLO fp16 仅 ×1.27（2.9M 参数不撞算力瓶颈，留 int8 空间）；BGE 7ms 不值得动。
2. **量化本身不破坏微调能力**（w8a8 保留 LoRA 输出分布），但**小模型 W8A8 的行为高度依赖 prompt 风格**：语言风格、指令形式（生成式 vs 判断式）、语言（中/英）任一等价性被破坏都会导致链路静默失效而非报错。
3. **BGE 单向量匹配已到能力边界**：否定词语义同点（戴/不戴眼镜不可分）、性别反义同点、贴阈值 FP 无法靠阈值调节解决；已定案的替代路径 = **属性线性头**（已上线）+ **Reranker 直配**（5005 实验中）。
4. **结构化属性由 VLM 生成不可靠**（复读陷阱、枚举退化），由 <1ms 的专用线性头接管后既快又稳——方向上验证了"VLM 只做描述生成、判别交给小头"的分工。

### 6.2 工程/运维教训（各花费了一轮以上调试）

- 测量必须在服务清场后进行：pkill 三种坑（comm 15 字符截断、`-f` 匹配自身 cmdline、同一命令里目录路径字面量误杀 shell）曾致测量虚高 20 倍、服务重启半途中断。
- 脚本环境变量会经 `${VAR:-default}` 泄漏进启动脚本，加载错模型无感知——服务重启一律 `env -i`。
- scp 推包丢执行位、板端 stdout 块缓冲、板端时钟落后宿主 ~19 天驱动告警目录命名，均为远程板卡环境的固定成本。
- 结论性测量须走生产路径（`bench_models --structured`），demo REPL 的多行拆轮/无 penalty 使早期测量系统性失真（q35 "5min/轮"与"结构化尚可"两个假结论皆源于此）。

---

## 7. 当前状态与下一步

### 7.1 线上态（截至 2026-11-04）

- 板端 `/home/web_vllm_rknn`，端口 5003：**基座 Qwen3-VL-2B W8A8 + combined + 属性头 + 无条件全量描述（34 字属性清单 prompt）+ 调试捕获开**；YOLO RKNN fp16 上 NPU，BGE 仍 ORT-CPU；18 条规则 threshold 0.55。
- 并行实验：`web_vllm_rknn_q35`（5004，代码与旧产物保留）、`web_vllm_rknn_reranker`（5005，Reranker yes/no logit 打分实验，分数分布 0.2–0.7 需重标定）。

### 7.2 遗留问题（按优先级）

1. **属性头符号门未接入告警判定**：当前 FP≈67%（属性错绑、性别反义同点、人物基线分 0.45–0.52 贴阈值）；头输出干净，预计可拦约半数 FP。
2. **阈值重标定**：整句口径切换后实测分 0.33–0.51，0.55 偏高、告警将稀少；且属性头阈值在宿主 bf16 特征上标定，板端 fp16 分布差待重标（`VLM_ATTR_HEAD_DUMP` 钩子已留）。
3. 帧延迟再压缩：图像 token 196→更少（需重标头）以压 ~3s 图像 embedding prefill，或试 n_batch。
4. **q35 微调**（压弃权长文/环境段/他她混用）：10-26 确认的投入方向，尚未开始。
5. 工程收尾：板端服务无自启；「一楼电梯厅」流读取超时重连中；10-26 一次线上无 OOM 迹象死亡未定位；V2 侧 REST 动态 A/B 与 Python→C++ 的 BGE 短句修复回移仍未闭环；YOLO int8 量化未做。

---

*详细过程记录见各项目内文档：[web_vllm_cpp/PROGRESS.md](web_vllm_cpp/PROGRESS.md)、[web_vllm_cpp/BUILD_CHANGES.md](web_vllm_cpp/BUILD_CHANGES.md)、[web_vllm_rknn/PLAN.md](web_vllm_rknn/PLAN.md)（全时间线）、[web_vllm_rknn/TECH_REPORT.md](web_vllm_rknn/TECH_REPORT.md)、[web_vllm_rknn/ISSUE_structured_acc_20261028.md](web_vllm_rknn/ISSUE_structured_acc_20261028.md)；谱系总览见 [TECHNICAL.md](TECHNICAL.md) §7。*
