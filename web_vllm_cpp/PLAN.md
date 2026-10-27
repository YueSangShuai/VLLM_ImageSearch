# web_vllm_cpp 移植计划表

> 配套 [PROGRESS.md](PROGRESS.md)（架构决策与踩坑记录）。本文件只做进度对账，随开发实时更新。
> 图例：✅ 完成 | 🔄 进行中 | ❌ 未开始
> 更新日期：2026-09-24

## 一、源码模块（src/）

| 模块 | 对应 Python | 状态 | 验证 |
|---|---|---|---|
| config.h/.cpp | env 变量 / 默认值 | ✅ 已对齐 rule_direct、judge 提示词、dyn ONNX 默认路径 | -fsyntax-only 通过 |
| util.h/.cpp | 码点级字符串/时间/uuid | ✅ | 通过 |
| frame.h/.cpp | ffmpeg 取帧 + stb 编解码 | ✅ 修过 resize v2 API | 通过 |
| detector.h/.cpp | PersonDetector（YOLO ONNX） | ✅ | 通过 |
| text_embedder.h/.cpp | BGE（WordPiece+CLS） | ✅ | 通过（待黄金样例数值对齐） |
| json_store.h/.cpp | JSON 读写/告警/日志 | ✅ | 通过 |
| vlm_describer.h/.cpp | vlm_search/vlm_describer.py GGUF 后端 | 🔄 代码已写完 | 语法过，待真实模型运行验证 |
| matcher.h/.cpp | VLMTextMatcher（四模式） | 🔄 代码已写完 | -fsyntax-only 通过，待行为对照 |
| stream_processor.h/.cpp | StreamProcessor | ✅ | -fsyntax-only 通过 |
| monitor.h/.cpp | MonitoringManager + DeviceManager | ✅ | 通过 |
| server.h/.cpp | app.py 全部路由 + 页面 + 心跳 | ✅ | 通过 |
| main.cpp | 启动序列（预加载+自动开监控） | ✅ | 通过 |

## 一.5 编译与数值对齐（2026-09-24 实测）

| 项目 | 结果 |
|---|---|
| 全 12 模块 -Wall -fsyntax-only | ✅ 全过 |
| 本地链接（宿主机 llama.cpp CPU build，llama-common+mtmd） | ✅ CMake 正式配置一次通过（build-local/） |
| rule_direct 冒烟（CPU 推理，4 张真实 ROI） | ✅ 判定与解析全部正确 | interval=2s 为生产规则配置值，非 bug；历史告警文件夹名是向量模式 FP，不能当 ground truth |
| **BGE 黄金样例** | ✅ 9/9 cos=1.000000 |
| **YOLO 黄金样例** | ✅ IoU=1.0000，conf 差 0.010 |

修复的两个关键数值 bug：
1. **YOLO 解码**：该 dyn 导出已内置 DFL，前 4 通道是 letterbox 像素坐标系绝对 xywh，
   不能再乘 stride / 加 anchor（原按 cxcywh+stride 解码坐标差 8 倍）
2. **BGE 分词**：两处
   - ASCII 词累加器漏 flush，含英文文本整词丢失
   - **SentenceTransformer 实际读 `sentence_bert_config.json` 的 `do_lower_case: true`**
     （覆盖 tokenizer_config.json 的 false）→ 生产行为是小写化。
     golden 的 tokenized 记录（cased AutoTokenizer 产物）对含 ASCII 文本不可信，
     golden 向量（ST 实际输出）才是基准。**PROGRESS.md 原「do_lower_case=False」结论已修正**

## 二、构建与部署

| 事项 | 状态 | 备注 |
|---|---|---|
| CMakeLists 源文件列表 | ✅ 已含 llama-common 链接 + common/vendor include（Dockerfile 传参对齐） | 本地 cmake 验证中 |
| Dockerfile llama 编译目标 | ✅ 已含 llama-common mtmd 目标 | 待实跑 |
| build.sh (Docker) | ⛔ 不可用 | 宿主机连不上 docker.io，`nvidia/cuda:12.2.0-devel` 镜像拉取失败（base 镜像有缓存但缺 devel/nvcc）；保留待有网环境 |
| build_host.sh | ✅ 产出 deploy/（CUDA 全量编译 BUILD_EXIT=0） | 坑：`/usr/bin/nvcc` 是 11.5（gcc≥11.3 编 .cu 必炸），脚本内显式选 `/usr/local/cuda-12.2/bin/nvcc` |
| run_web.sh env 对齐 | ✅ rule_direct 默认 + judge 提示词导出已写入 | - |
| 前端 templates/static 拷贝 | ✅ build.sh 内实现 | 部署时用 |
| 本地真实编译链接 | ✅ CPU(build-local) + CUDA(build) 双通过 | 均一次通过 |
| Docker 全量编译 | ⛔ 宿主机无外网拉不到镜像 | 见上，走 build_host.sh |
| 端到端部署验证 | ✅ 2026-09-29 实机通过 | deploy/run_web.sh 常驻运行：3 路 RTSP→YOLO→rule_direct VLM→告警落盘全通；API 5 条告警、detection_log 持续增长、ROI/original 像素级与 Python 路径一致（MSE 对照排除旋转疑云）、页面 200 |
| 可移植打包 | ✅ | scripts/package_deploy.sh → web_vllm_cpp_deploy_*.tar.gz（自带 CUDA 运行库+libstdc++，闭包自检）；目标机需 sm_80+驱动535+ffmpeg+glibc≥2.35 |

## 三、对齐验证（移植质量门槛）

| 项目 | 状态 | 基准 |
|---|---|---|
| BGE 向量 vs 黄金样例 | ✅ cos=1.0 | golden/bge_golden.json（以 vectors 为准） |
| YOLO 检出 vs 黄金样例 | ✅ IoU=1.0 | golden/yolo_golden.json bbox 精确一致 |
| VLM 描述输出 | ✅ rule_direct 冒烟通过（2026-09-24，CPU llama） | 4 张真实 ROI：全黑着装→命中「黑衣服男生」；条纹上衣/红衣（历史向量模式 FP）→正确拒「不匹配」；背身俯拍戴眼镜→保守不判（符合提示词约束） |
| REST 契约逐字段 | 🔄 静态对照完成（全部路由字段/消息文本一致；告警·日志字段一一对应） | 待部署后动态 A/B：同数据快照分别起 Python(5004)/C++(5005) diff 响应 |

## 四、行为对齐备忘（PROGRESS.md 第四节为准，此处只列易漏项）

- [x] rule_direct 模式（python 脚本当前默认！）+ {rules} 注入 + 长名优先回认
- [x] Female 缺失派生 Male；combined 双生成
- [x] 判定式 `best_pos >= threshold && best_pos > max_neg + score_diff`
- [x] POST/PUT /api/rules 追加/更新所有 tasks.json 任务 + refresh_rules
- [x] 帧预览 1 秒缓存；告警图路由带 1 小时 Cache-Control
- [x] 心跳 5s、重连退避 min(delay*2,30)、单实例 pkill 限定路径（2026-09-24 核实：server.cpp heartbeat_loop / stream_processor.cpp 退避 / run_web.sh pkill 均已实现）
- [x] detection_log 头插截断 1000 / alerts 头插（stream_processor.cpp:199 append_detection_log、monitor.cpp:47 prepend_alert 已接线）
- [x] CORS 预检 OPTIONS 兜底应答（server.cpp，对齐 flask-cors 自动应答；待下次编译带上）
- [x] 已接受差异：占位图无 cv2 文字（纯黑 640x480）；GPU 边缘小图 temp=0 下非确定翻转（Python 同特性）

## 五、收尾

- [ ] 全模块语法检查 → 本地链接 → 跑通
- [ ] 中间文件清理（llama_buildsrc、/tmp 草稿、build 产物按需保留）
- [ ] PROGRESS.md 同步收尾记录
