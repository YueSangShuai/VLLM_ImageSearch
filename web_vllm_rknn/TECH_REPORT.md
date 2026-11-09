# 技术报告：web_vllm_rknn 板端行人布控平台（RK3588）

> 依据 `output_cpp/run_web_rk.sh`（板端唯一启动入口）展开，交叉引用 README.md、BUILD_CHANGES.md、cpp/src/ 源码。
> 撰写日期：2026-11-03。当前线上部署目录：板上 `/home/web_vllm_rknn`，端口 5003。

---

## 1. 系统概述

本系统是 `web_vllm_cpp` 行人布控平台向 **NanoPi R6S（RK3588，三核 NPU）** 的边缘移植：在单板离线环境下接入 IPC 摄像头 RTSP 流，完成

**行人检测（YOLOv8n）→ ROI 裁剪 → VLM 中文外观描述（Qwen3-VL-2B W8A8）→ 属性线性头 / BGE 向量匹配 → 规则打分 → 告警落盘 + Web 展示**

全链路无需云端。业务层（Web、规则、告警存储）与 x86 C++ 版一致，差异全部收敛在推理后端与启动配置里——而所有推理后端的选型与提示词策略，都固化在 `run_web_rk.sh` 这一个脚本中。

## 2. 启动脚本解析

脚本采用「脚本与二进制同目录」的自包含包布局，核心机制是**全部配置走 `${VAR:-默认值}` 环境变量**，任何一项可在不改脚本的情况下覆盖（也由此埋下 env 泄漏的坑，见 §8）：

```
output_cpp/
├── web_vllm_rknn          # 主服务二进制（aarch64 交叉编译产物）
├── run_web_rk.sh          # 本脚本；必须 bash（dash 不认 pipefail）
├── lib/                   # LD_LIBRARY_PATH 注入：librknnrt / librkllmrt 等板端 runtime
├── models/                # 全部权重，软链 → rknn 仓库 models/rk3588/
├── data/ static/          # 运行时数据 / 前端页面（真源在 cpp/static/，构建时拷入）
```

### 2.1 运行时目录与端口

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LD_LIBRARY_PATH` | `$PWD/lib` | 板端 RKNN/RKLLM runtime 随包分发，不依赖系统安装 |
| `WEB_VLLM_DATA_DIR` / `STATIC_DIR` / `MODELS_DIR` | `./data` `./static` `./models` | 相对包根目录 |
| `WEB_VLLM_PORT` | **5003** | 与 x86 版（5000 系）并行；q35 试验平台占 5004 |

### 2.2 VLM：RKNN 视觉塔 + RKLLM 语言塔

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLM_MODEL_NAME` | `models/qwen3-vl_vision_rk3588.rknn` | 视觉塔（ViT+merger），448×448 固定输入，输出 196 token × 2048 维 |
| `VLM_MMPROJ_PATH` | `models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm` | 变量名沿用历史，**实指语言塔 `.rkllm`**（W8A8 量化） |
| `RKLLM_NPU_CORE_NUM` | 3 | 占满三核 NPU |

**模型身份沿革（重要）**：2026-10-29 起线上默认从 LoRA 微调双件套（文件名含 `qwen3vl-ft`）切换为**未微调基座 Qwen3-VL-2B-Instruct**。原因：板端 W8A8 上 ft 与基座的结构化输出均达不到 host bf16 质量（复读枚举属性表），而 ft 的中文描述优势可由 prompt 工程替代；切回 ft 只需改这两个变量。`models/` 内软链随默认同步替换，ft 文件保留在板可随时回退。

### 2.3 检测与文本向量

| 变量 | 默认值 | 后端 |
|---|---|---|
| `DETECTOR_ONNX_PATH` | `models/yolov8n_640_rk3588.rknn` | RKNN（NPU），fp16，640 输入 |
| `EMBEDDING_ONNX_PATH` | `models/onnx/bge-small-zh.onnx` | **ONNXRuntime-CPU**（NPU 已被 VLM/YOLO 占满，BGE-small 91MB CPU 足够） |
| `EMBEDDING_VOCAB_PATH` | `models/vocab.txt` | BGE 中文词表 |

### 2.4 属性线性头（2026-10-30 上线）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLM_MATCH_MODE` | **combined** | `combined` / `description` 两态；`description` = 纯描述向量匹配（原 `structured` 态随属性头上线已从代码移除） |
| `VLM_ATTR_HEAD_PATH` | `models/attr_head.bin` | `=off` 禁用头并回退旧预处理 |

`attr_head.bin` 来自 [qwen_classifer/](../qwen_classifer/) 项目：冻结 Qwen3-VL 视觉塔 + 线性头（2048→54，PA-100K 多标签，val mAP 0.7006 / Acc 0.9385），`export_head.py` 导出 W/b + 逐属性最优 F1 阈值 + 属性名块。**不重导视觉 rknn**：C++ 直接对 run_imgenc 产出的 embedding 段 0（merger 主输出，196×2048）做 mean-pool + matmul（CPU <1ms）。

启用头后 combined 模式的关键收益：**描述生成一次即同时拿到 54 维属性，省掉旧 combined 整次 128-token 结构化生成（约 −10s/帧）**。加载时 magic/in_dim/属性名与 `attributes.h` 逐字校验，任何失败仅日志禁用、不拖垮服务。头启用时预处理切换为 `expand2square(bg=127.5)+floor 偏移+resize448`，与训练端 `data.py` 及板端 RKNN 导出逐像素对齐。

### 2.5 提示词（单路径，全中文）

- **描述路径 `VLM_DESCRIPTION_PROMPT`**（当前=无条件全量描述，10-20 加强版：详述外观+禁猜测+禁疑问句 hedging+强制中文陈述句）：告警走「描述 vs 规则」纯向量匹配。
  规则条件化路线（`{rules}` 注入 + 「无匹配」哨兵）10-30 上线、11-03 回退：基座 W8A8 在「判断/选择/输出规则名」式 prompt 下坍缩成只输出 有匹配/无匹配 元 token。代码路线未删：prompt 含 `{rules}` 仍会注入；弃权判定=整段 ≤6 字哨兵 或 去空白全文 contains「无匹配」→ 向量得分置 0，符号属性路径不受影响。
  旧「结构化路径 `VLM_STRUCTURED_PROMPT`」（54 属性名列表 + 复读守卫/penalty 1.4）已随属性头取代其职能而从代码删除，结论与教训见 BUILD_CHANGES 10-28 条目。

两条铁律，各来自一次线上事故：
1. **prompt 必须中文**——英文长描述 vs 中文规则 BGE 仅 ~0.33 分，永不越阈（10-20「检测到行人但不报警」根因）；
2. **prompt 内不得自带 `<image>`**——vlm_describer 会自动前置，叠加成双占位符后行为异常（11-03 修正）。

### 2.6 调试捕获

`VLM_DEBUG_CAPTURE=1`（当前默认开）：**所有** YOLO 检出的行人（无论是否告警）落盘 ROI+画框原图至 `data/debug_images/`，VLM 描述/属性/命中分数写 `data/debug_log.json`，Web `/debug` 页可筛选回看。这是 10 月底以来 FP 归因的主要数据来源。回退置 0 重启即停，`data/debug_*` 可直接删。

## 3. 单帧推理链路与耗时

```
RTSP 帧 ──YOLOv8n/RKNN──► ROI 裁剪(224) ──视觉塔 rknn──► 196×2048 emb
                                                    ├─ mean-pool+attr_head ──► 54 属性 +阈值判定（<1ms）
                                                    └─► RKLLM 生成中文描述 ──► 分句过滤(「。」→「，」两级,
                                                        isNoiseClause 黑名单去 hedge 句) ──► BGE query 取 max
                                                                                            │
                        规则触发：positive ≥ threshold 且 > max_neg + score_diff，interval 冷却 ◄┘
```

实测拆解（板端停服务，bench_speed_matrix，脚本已清理、数值为历史实测）：视觉塔 ~1.9s，图像 embedding prefill ~3s，文本 prefill 328ms，decode ~29 tok/s（ft-2B），常驻单帧端到端约 5.1s。已验证判废的优化：librkllmrt 1.3.1 文本 prompt cache 可用但 prefill 占比太小；图像 embedding 无法存 cache。下一杠杆 = 降图像 token 数或 n_batch 调参。

## 4. 匹配策略沿革与当前形态

| 阶段 | 策略 | 结论 |
|---|---|---|
| BGE 单向量对称匹配 | 描述全文 vs 规则 | 基线 0.45~0.52 与阈值重叠，长描述分句取 max 虚高 |
| 非对称检索 | 规则 query 加 BGE 指令前缀 | 衣着 margin 拉开；**否定词负样本向量同点，救不了 FP**（10-20） |
| 规则条件化描述 | prompt 注入 {rules} + 无匹配哨兵 | 描述变短变准，弃权判定已放宽为「全文 contains 无匹配」 |
| combined + 属性头（当前） | 描述一次生成 + 符号属性顺带产出 | 省一次生成；**头属性尚未参与告警判定**，仅展示 |

**当前主要矛盾（11-03 误判分析）**：抽样 12 次命中仅 4 次真（FP≈67%），机制有三：①属性错绑——BGE 不区分部位，「黑色长裤」顶高「黑衣服男生」；②性别不敏感——反义短语在向量空间同点；③「话痨式弃权」绕过哨兵（已修）。共同的解法方向：**用 attr_head 符号输出做门控**（性别、u/l 颜色逐属性校验），头在本批数据上输出全部干净，可拦下约半数 FP——这是下一步工作。

## 5. 模型产物与构建部署

- 模型清单与下载：见 README「所需模型与下载」表；转换在 x86 双 conda 环境完成（`rknn`：rknn-toolkit2 2.5.0；`rkllm`：rkllm-toolkit 1.3.1，互斥勿同装，CWD 须 ext4）。
- 固件前提：kernel ≥6.1.141 / rknpu ≥0.9.x（老 5.10 的 0.8.2 跑不了 RKLLM w8a8），换 boot-chain 不动 rootfs，物料与救砖流程见 `firmware/`、`RECOVERY.md`。
- 构建：`cpp/build_scripts/build_aarch64.sh` 是唯一入口，组装 output_cpp/（**⚠️ 会用 web_vllm_cpp/data 覆盖 output_cpp/data，push 前先备份**）。
- 部署：日常只 push 单文件（二进制/run 脚本/tasks.json）+ md5 核对 + `scripts/deploy/start_svc_board.sh` **纯净 env** 拉起；板上数据热改（换流、规则阈值）走 scp data 文件 + 重启，勿整包覆盖。
- 板上告警落盘 `data/alerts_images/<日期>/<设备>/<规则>/{roi,original}/` + `alerts_history/alerts_YYYY-MM-DD.json`。**告警图含真实行人 PII，勿外传。**

## 6. 运维铁律（血泪条目速查）

1. **启停分离**：`pkill -f "[w]eb_vllm_rknn"; <启动命令>` 写在同一条 ssh 命令里，pkill 会匹配启动串里的目录字面量自杀，服务静默不起——必须分两条命令。comm 15 字符截断：q35 实例须 `pgrep/pkill -f "[.]/web_vllm_rknn_q35"`。
2. **纯净 env 启动**：测试脚本 export 的 `VLM_MODEL_NAME` 等会经 `${VAR:-default}` 被 run 脚本照收，导致服务加载错模型（曾造成「结构化突然不准」假象）。
3. **scp 丢执行位**：推 .sh 后补 `chmod +x`，否则 nohup Permission denied。
4. **VLM 静默失败形态**：模型文件缺失时 detector/web 全部正常，唯描述恒空、attrs 恒 `[Male]`（空属性派生指纹）、日志刷「VLM描述为空」。换部署目录后必须核对 run 脚本默认路径的文件真实存在。
5. **内存**：板端手动跑评测前先停服务——rerank 评测撑爆内存触发 OOM killer 杀掉了 2.7G shmem 的在线服务。
6. **板端时钟落后宿主约 19 天**，告警日期目录以板端时间为准；排查时先 `date`。

## 7. 现状与下一步

**当前线上态**（截至 2026-11-04）：基座 Qwen3-VL-2B W8A8 + combined + 属性头 + **无条件全量描述**（11-03 从规则条件化回退，见 §2.5）+ 调试捕获开启；18 条规则 threshold 0.45/0.55、negatives 已清空。11-04 板卡清空后整包重建部署到 /home/web_vllm_rknn，清理版二进制（5baf46c2，删 VLM 结构化生成路径/judge prompt/旧工具后重编）已在板；本次未带 ft 双件套（需要时从 npu_models/ 补推）。

**待办（按优先级）**：
1. 头属性符号门接入告警判定（治属性错绑/性别不敏感 FP，数据已证明可拦 ~5/12）；
2. 头阈值板端 fp16 特征重标定（现值标定于 bf16，`VLM_ATTR_HEAD_DUMP` 钩子已留）；
3. 手持物品等规则改属性名符号直判（`HoldObjectsInFront`），「长头发的女生」类无头属性可校验的规则提阈或禁用；
4. 图像 token 数下调（196→更少，需重标头）压视觉 prefill 的 ~3s；
5. 板端时钟同步（NTP 或开机校时）。
