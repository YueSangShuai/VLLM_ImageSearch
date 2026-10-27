# web_vllm C++ 移植工作记录

> 目标：将 `/data/yuesang/LLM/FG-CLIP/web_vllm`（Flask + transformers 行人布控系统）整体用 C++ 重写；
> Docker 仅作可复现编译环境，产物二进制直接在宿主机运行；前端页面原样复用，REST 契约与 Python 版逐字段一致。
> 整理日期：2026-09-24

## 一、架构决策（已与用户确认）

| Python 组件 | C++ 方案 |
|---|---|
| Qwen3-VL 描述（transformers / llama-cpp-python） | llama.cpp libllama + libmtmd 直接链接，单进程、生成加锁；模型用微调 GGUF |
| YOLOv8n（ultralytics） | 导出动态 H/W 的 ONNX，ONNX Runtime CPU 推理，letterbox/NMS 手写 |
| BGE bge-small-zh（sentence-transformers） | torch 导出 ONNX；C++ 手写 WordPiece 分词 + CLS pooling + L2 归一化 |
| cv2 取帧/画框/编码 | ffmpeg 子进程取帧 + stb 编解码/缩放，**不引入 OpenCV** |
| Flask + flask-cors | cpp-httplib，同路径同 JSON 字段 |
| JSON 文件读写 | nlohmann/json + mutex + tmp 写后 rename 原子替换 |
| 前端 | templates/*.html（确认零 Jinja）+ static/ 原样拷贝 |
| GPU 推理 | 不装 NVIDIA Container Toolkit：Docker 编译、宿主机跑二进制，.so 随 deploy/ 打包（rpath=$ORIGIN/../lib） |

数据目录与 Python 版**共用** `/data/yuesang/LLM/FG-CLIP/web_vllm/data/`，同一时间只跑一套（沿用 pkill 单实例约定，且限定项目路径避免误杀）。

## 二、模型资产（已完成 ✅）

### 1. 微调模型转 GGUF —— 已实图验证
- 源：`/data/yuesang/LLM/saves/Qwen3-VL-2B-Instruct/lora/train_2026-09-09/checkpoint-17254/lora`（safetensors）
- 产物：
  - `/data/yuesang/LLM/FG-CLIP/models/qwen3vl-ft-20260909/qwen3vl-ft-f16.gguf`（3.4GB）
  - `/data/yuesang/LLM/FG-CLIP/models/qwen3vl-ft-20260909/mmproj-qwen3vl-ft-f16.gguf`（819MB）
- 工具：`/data/yuesang/LLM/llama.cpp`（2026-07 版，含 `conversion/qwen3vl.py` 转换器与已编译的 mtmd）
- 元数据校验：`general.architecture`、`tokenizer.ggml.*`、`tokenizer.chat_template` 齐全
- 实图验证：`llama-mtmd-cli` 对真实告警 ROI 跑结构化提示词，输出
  `Age18-60,Front,HoldObjectsInFront,ShortSleeve,Trousers,ublack,lblack`
  与人工看图（黑短袖✓ 正面✓ 持物✓ 成年✓）一致；`lblack`（裤子实际米白色）是**模型自身幻觉**
  ——Python 版对同图也输出过"黑色七分裤"的同样幻觉，证明 GGUF 转换忠实、无需修。

**踩坑记录（重要）**：转换 mmproj 时稳定报 `OSError: Not enough free space to write 33554432 bytes`
（errno=None），两次精确停在同一字节偏移 605292704，但磁盘实际剩余 400GB+，numpy 直接 tofile 复现不了。
根因：`~/.local` 里的 numpy 1.26.4 对 lazy tensor 的 `tofile` 路径有伪 ENOSPC bug。
**解法**：monkey-patch `gguf.lazy.LazyNumpyTensor.tofile`，改用 `fout.write(array.tobytes())` 绕开（脚本 `/tmp/fix_convert.py`，重建时可用）。
排查中另发现：**`conda activate FGCLIP2` 失效，python 实际落到 Qwen 环境**（其 numpy 2.0.2/torch2.8）；
真实解释器是 `/home/yuesang/miniconda3/envs/FGCLIP2/bin/python`（numpy 1.26.4 来自 `~/.local`）。

### 2. YOLOv8n ONNX —— 已完成
- 产物：`/data/yuesang/LLM/FG-CLIP/models/onnx/yolov8n-dyn.onnx`
- 关键决策：**动态 H/W**。生产路径是"先整帧缩到≤640，再交给 ultralytics letterbox"，
  ultralytics 默认 `auto=True` 会把输入 pad 到 stride-32 对齐的**非方形**（如 640x480），
  固定 640x640 导出与 ultralytics 像素布局不同，会导致坐标对不齐，故导出动态形状并在 C++ 复刻 auto letterbox。

### 3. BGE ONNX —— 已完成
- 产物：`/data/yuesang/LLM/FG-CLIP/models/onnx/bge-small-zh.onnx`
- 接口：input_ids / attention_mask / token_type_ids → last_hidden_state（动态 batch/seq）
- 关键修正：**bge-small-zh-v1.5 用 CLS pooling**（`1_Pooling/config.json`），且 **do_lower_case=False**
  （"World"→[UNK]、"hello" 原样保留）——不是常见认知里的 mean pooling。

### 4. 黄金样例（C++ 对齐基准）—— 已完成
- `/data/yuesang/LLM/FG-CLIP/web_vllm_cpp/golden/bge_golden.json`：9 条文本的
  ST 归一化向量(512维) + HF tokenizer 精确 input_ids（含中文标点逐字切分、WordPiece 行为样例）
- `/data/yuesang/LLM/FG-CLIP/web_vllm_cpp/golden/yolo_golden.json`：真实告警帧上生产路径
  （scale=min(640/max(h,w),1) resize 后 ultralytics）检出
  `bbox [279,1,318,113] conf 0.675`

## 三、C++ 工程（进行中）

目录 `/data/yuesang/LLM/FG-CLIP/web_vllm_cpp/`：

```
CMakeLists.txt      已完成（依赖 LLAMA_LIB_DIR / LLAMA_INC_DIRS / ORT_DIR 三个缓存变量）
Dockerfile          已完成（nvidia/cuda:12.2.0-devel-ubuntu22.04；sm_80 only；BUILD_SHARED_LIBS）
build.sh            已完成（rsync llama.cpp→上下文 → docker build → 拷出产物/CUDA 库 → deploy/）
run_web.sh          已完成（环境变量与 use_vllm_web.sh 对齐；LD_LIBRARY_PATH 指向 deploy/lib）
.dockerignore       已完成
third_party/        已下载：httplib.h(v0.18.3) json.hpp(v3.11.3) stb_image/_write/_resize2 onnxruntime 1.18.1(预编译)
src/
  config.h/.cpp     已完成：env 变量解析、strip_image_placeholder、默认路径（GGUF/onnx/vocab）
  util.h/.cpp       已完成：UTF-8 码点级 trim/substr/split_any、uuid4、now_isoformat、sanitize_name
  frame.h/.cpp      已完成：fork+ffmpeg 取帧(带超时 kill)、stb JPEG 解码/编码、stbir 缩放、draw_rect、
                      url_encode(对齐 quote(safe=''))、build_rtsp_url、占位图
  detector.h/.cpp   已完成：letterbox(auto, stride32) + 动态 ONNX 推理 + 3 网格解码(仅 class0)
                      + 贪心 NMS(IoU0.7) + 坐标还原（含 Python int 截断语义）
  text_embedder.h/.cpp 已完成：WordPiece（不分 lowercase、中文逐字、标点独立、词表外 [UNK]）
                      + ORT 推理 + CLS pooling + L2 归一化
  json_store.h/.cpp 已完成：Paths、load/save 原子写、append_detection_log(头插截断1000)、prepend_alert
```

### 已编译验证（2026-09-24）
以上 6 个模块 g++ 11.4 `-Wall -fsyntax-only` 全部通过。首次检查修掉 5 处错误：
config.cpp 缺 `<unistd.h>`、`exe_dir()` 自由函数与成员同名冲突；
frame.cpp 的 `stbir_resize_uint8_generic` 是 resize v1 API，改用 v2 medium API `stbir_resize`
（STBIR_TYPE_UINT8 即线性色彩空间）；新增 `third_party/nlohmann/json.hpp → ../json.hpp` 软链
（json_store.h 按 `<nlohmann/json.hpp>` 引用）；json_store.cpp `json::resize` 改为
`get_ref<json::array_t&>().resize`。httplib/nlohmann 独立 TU 编译通过。

### 2026-09-24 补充：全部 src 模块已写完并通过黄金样例
`vlm_describer`（llama+mtmd，仿 mtmd-cli 单轮路径）、`matcher`（四模式含 rule_direct）、
`stream_processor`、`monitor`、`server`、`main` 均已落地；BGE cos=1.0、YOLO IoU=1.0（见 PLAN.md 一.5）。
原「三处行为对齐」中的 rule_direct/心跳/重连/json_store 接线已逐项核实。

### 2026-09-24 补2：构建路径切换 + rule_direct 冒烟结论
- **Docker 编译路径不可用**：宿主机连不上 docker.io，`nvidia/cuda:12.2.0-devel` 拉取失败
  （base 镜像有缓存但无 nvcc）。宿主机自带 CUDA 12.2 + gcc 11.4 → 新增 `build_host.sh`
  直接编 `llama.cpp/build-cuda`（sm_80）+ 应用链接，deploy/ 布局与 build.sh 完全一致。
- **rule_direct 冒烟**（CPU llama，真实 GGUF，4 张真实 ROI）：
  - 全黑着装 ROI → 输出「1. 黑衣服男生、2. 长头发的女生」，编号+顿号格式解析正确，双规则命中
  - 条纹上衣/红衣 ROI（历史告警文件夹是**向量模式 FP**，文件夹名≠ground truth）→ 正确「不匹配」
  - 背身俯拍戴眼镜 ROI → 保守不判（符合提示词「只依据清晰可见内容」约束）
  - **教训**：历史告警 ROI 存的是 `cv2.resize(224,224)`，VLM 生产输入是原始尺寸 crop；
    冒烟要用 bbox 从 original 图重裁，别拿 224 存档图当输入
  - `interval` 生产配置为 2 秒（tasks.json），不是默认 60——测试间隔判断前先查这里
- **REST 契约静态对照完成**：全部 /api/* 路由逐字段一致；Flask 2.3+ jsonify 默认按键排序，
  nlohmann dump 同为键序，响应字节序亦对齐；monitoring start/stop/refresh 消息文本逐字一致。
  已接受差异：占位图无 cv2 文字（纯黑 640x480 JPEG）。
  动态 A/B 脚本 `scripts/rest_ab_test.sh`：数据快照副本上 Python(5004, GGUF 后端) vs C++(5005)
  逐端点归一化 diff（剔除 id/时间戳/状态等易变字段，先停监控+清空告警再读）。

### 2026-09-24 补3：GPU 路径观察（非移植 bug，记录以免误判）
- deploy 冒烟（CUDA 全 offload）：32x101 全黑背身 ROI 在 rule_direct 下**连跑三次两种输出**
  （命中「黑衣服男生」/「不匹配」），同参数 CPU 构建则稳定命中。
  根因：CUDA 归约核非确定 + temp=0 平票翻转 + 低分辨率边缘图本身歧义。
  Python+GPU 走同一套 CUDA 核，行为特性相同——对齐结论以 CPU 冒烟 + 静态对照为准，
  别拿单张边缘图的单次 GPU 输出当回归基准。
- 另见 `common_fit_params: failed to fit params...` 警告：GPU 显存紧张（他人进程占用）时
  llama 尝试自动降 offload 层数失败后按原样继续；生产如显存紧张关注此警告。

### 尚未完成
- REST A/B 实跑 + 端到端部署验证
- 中间文件清理（llama_buildsrc、/tmp 草稿）

### 2026-09-29：与 Python 版解耦 + combined 默认 + Docker 跨平台交付
- **解耦**：data/、static/、templates/ 已复制进本仓库；config.cpp/run_web.sh 的
  `WEB_VLLM_DATA_DIR` 默认改为 `<仓库>/data`；build*.sh 前端拷贝改本地副本。
  Python web_vllm/ 不再是依赖（仅 rest_ab_test.sh 按设计指向它）。
  注意：两边 data 从此**各写各的**，web_vllm/data 的告警历史不再反映 C++ 服务状态。
- **匹配模式默认改 combined**（结构化属性+行人描述同时开）：run_web.sh、config.cpp、
  server.cpp 状态页三处同步；非法值回退仍是 structured（与 Python 一致）。
- **容器交付（新路径，与宿主机 deploy 包路径并行）**：
  - `Dockerfile.multi`：多阶段，builder 沿用旧 Dockerfile 容器内路径
    （/src/llama-build/bin、/src/web_vllm_cpp/build/web_vllm_cpp，旧 build.sh 的
    docker cp 段对新文件也可用）；runtime=ubuntu22.04+ffmpeg/libssl3/libgomp1/tzdata，
    ENTRYPOINT 直接复用 run_web.sh（其 ${VAR:-} 默认让位于镜像 ENV）。
  - `build_image.sh`：x86_64↔aarch64 / CUDA on|off 策略矩阵；aarch64 默认 CPU 推理
    （GGML_CUDA=OFF + 下载 aarch64 ONNX Runtime）；模型 cp -al 暂存进 models/ 后
    COPY 进镜像（--no-models 则运行侧 -v 覆盖 /opt/models）。不做交叉编译。
  - `scripts/make_devel_base.sh`：本机无 docker.io 时的 devel 基座——本地
    cuda-base 镜像 + 宿主机 /usr/local/cuda-12.2（tar 流剔除 nsight/静态库）commit 成
    `web-vllm-cpp-builder-base:12.2`，nvcc 自检通过。
  - `docker-compose.yml` / `docker-compose.gpu.yml`：CPU 基础 + GPU 叠加；data bind
    mount；TZ=Asia/Shanghai（告警日期目录依赖 localtime）。
  - 待验证：镜像构建、容器内 ldd、GPU overlay、告警落盘。ARM 路径需真实 ARM 机验证。
- **2026-09-29 转向：放弃 Docker 路线（文件保留但不再推进），改纯 CPU 交付**：
  - `build_cpu.sh`：从**仓库自带** llama_buildsrc/ 编 llama.cpp CPU 共享库
    （GGML_CUDA=OFF，无 nvcc/无 GPU 的机器也能编）+ 链接应用 + 组 deploy-cpu/
    （与 GPU 版 deploy/ 并存）；ONNX Runtime 用预编译 .so，httplib/nlohmann/stb
    header-only。产物自带 ldd 自检。
  - `run_web_cpu.sh`（拷成 deploy-cpu/run_web.sh）：VLM_DEVICE=cpu（代码内强制
    n_gpu_layers=0）、无 CUDA 环境变量、pkill 只匹配 deploy-cpu 路径。
  - 踩坑：vendored llama_buildsrc 缺 common/build-info.cpp.in / build-info.h
    （历史 rsync 遗漏），已从外部树补回。
  - CPU 性能预期：4B F16 VLM 单帧全链路十秒级（combined 两次 VLM 推理）；
    高吞吐场景用 rule_direct 或 Q4 量化 GGUF。

### 2026-09-29：BGE 向量匹配「白上衣黑短裤 → 黑衣服男生」FP 修复
- **实测根因**（用线上同一 ONNX 复算）：长叙事描述整体向量是"话题相似度"，
  同域两句基线就 0.4~0.55。FP 描述对正样本「穿黑色上衣的男人」cos=0.5613
  ≥ threshold 0.55，负样本最高仅 0.5470、score_diff=0 → 误报。
  真 TP 同通道也只有 0.5852——**真假分差 0.024，调阈值不可分**。
- **修复**（matcher.cpp `buildEmbeddingText`）：进 BGE 的匹配文本从整段描述改为
  **只含衣着/携带关键词的句子**（按"。"分句过滤，摘不到回退全文）。
  实测衣着短句下 FP pos0.5758<neg0.5768（自动出局）、TP pos0.6129 vs neg0.5091
  （margin+0.104）。英文属性码串 BGE 不识别（0.44 无区分度），勿直接 embed。
- **规则参数**：黑衣服系 3 条 score_diff 0→0.02（FP margin 仅 -0.001，留余量）。
  threshold 维持 0.55 即可。自然语言规则想更稳优先加硬负例（如"穿白色上衣黑短裤的男生"）
  而非抬阈值。
- **坑**：Bash 工具里 `pkill -f "<含路径模式>"` 会匹配到工具包装 shell 自身的
  cmdline 把会话打死（exit 144）。用行尾锚定 `pattern$` 规避。

## 四、行为对齐要点备忘（移植时勿丢）

1. `match()` 判定式：`best_positive >= threshold && best_positive > max_negative + score_diff`；
   combined 模式下自然语言规则走 BGE 向量匹配（这正是"黑裤子匹配黑上衣"误报的根源），
   属性规则（如 `ublack`）走符号匹配；Female 缺失时派生 Male（processor.py 本次会话已加）。
2. `use_vllm_web.sh` 默认 `VLM_MATCH_MODE=combined`、`VLM_RULE_DESCRIPTION_ENABLED=0`（二审默认关闭）。
3. 日志/告警字段名、`/api/*` 路径、POST /api/rules 会把规则追加进**所有** tasks.json 任务（保持 parity）。
4. 重连退避：sleep 后 delay=min(delay*2,30)；取帧命令与 processor.py 完全一致（含 `-stimeout 5000000`）。
5. 前端 `rules.html` 有 10fps 帧预览，`/api/video/frame/<id>` 有 1 秒缓存，服务端需实现。

## 五、2026-09-29 会话总结（变更清单 + 操作手册）

### 5.1 与 Python 版彻底解耦
- `data/`、`static/`、`templates/` 已复制进本仓库，全部脚本/代码不再引用
  `../web_vllm/`（唯一保留：`scripts/rest_ab_test.sh` 的 PY_SRC，按设计指向 Python 版）。
- 生效点：`src/config.cpp` data 默认 `<仓库>/data`；`run_web.sh`/`deploy/run_web.sh`
  `WEB_VLLM_DATA_DIR=$HERE/../data`；build*.sh 前端从本地副本拷。
- **后果**：两份 data 从此各写各的；两边同时跑 = 各自独立服务，注意端口错开。
- 模型也已硬链进 `models/`（GGUF×2 + ONNX×2 + vocab，同分区零空间；目录结构镜像
  `FG-CLIP/models/`）。两个 run_web*.sh 的模型默认路径已改指 `$HERE/../models/...`；
  环境变量仍可覆盖。**跨机拷贝注意**：硬链在 scp/rsync 后会变全量副本（4.3G），
  或用 `--links`+同构目录、或干脆让新机复用外部模型路径。

### 5.2 匹配模式默认 combined
结构化属性 + 行人描述同时开。三处同步：`run_web.sh`、`config.cpp`（非法值回退仍
structured，与 Python 一致）、`server.cpp` 状态页。切回：`VLM_MATCH_MODE=rule_direct ./run_web.sh`。

### 5.3 纯 CPU 交付（当前生产方式，不用 Docker）
- `build_cpu.sh`：一条命令 = llama.cpp CPU 库（从自带 llama_buildsrc/，**无 CUDA/nvcc/GPU 也能编**）
  → 链接应用 → 组装 `deploy-cpu/`（lib 仅 28M，无 libcuda 依赖）→ ldd 自检。
  三方库：ONNX Runtime 预编译 .so 直接用；httplib/nlohmann/stb header-only。
- `run_web_cpu.sh` → 构建后为 `deploy-cpu/run_web.sh`：VLM_DEVICE=cpu（代码强制 ngl=0）、
  pkill 只匹配 deploy-cpu。
- **换 x86_64 Linux 机器三步**：拷仓库目录 →（装 cmake/g++/ffmpeg）→ `./build_cpu.sh && ./deploy-cpu/run_web.sh`。
- 性能预期：4B F16 单帧全链路十秒级；高吞吐上量前：rule_direct / Q4 量化 / 降帧率。

### 5.4 Docker 跨平台路线（文件已备、**未推进**）
`Dockerfile.multi`、`build_image.sh`、`docker-compose.yml(+.gpu.yml)`、
`scripts/make_devel_base.sh`（本机无 docker.io 时用本地 cuda-base+宿主工具链合成 devel 基座，已验证可用）。
目标机只需 docker(+nvidia-container-toolkit)、模型打进镜像。若日后启用：先修 build_image.sh 里
`--runtime-base`/`--base` 指本地镜像，ARM 路线从未真机验证。

### 5.5 BGE 误报修复（matcher 匹配文本 + 规则参数）
- 根因：长描述整段向量=话题相似度，FP/TP 分差仅 0.024，调阈值不可分（实测数据见 09-29 FP 修复节）。
- **修复 1（代码）**：`matcher.cpp::buildEmbeddingText` —— 进 BGE 的匹配文本改为
  描述中**含衣着/携带关键词的句子**（摘不到回退全文；告警展示的描述不变）。
- **修复 2（参数）**：黑衣服系规则 `score_diff` 0→0.02（现由 PUT /api/rules 写入）。
- 遗留：Python 版仍是整段匹配，与 C++ 不再逐字节对齐（跑 rest_ab_test 前需先回移）。

### 5.6 规则文本优化方法论（实测得出，适用所有自然语言规则）
- 公式：`负例 = 同句式异色短语（灰/蓝/白上衣的男人）+ 异性别同特征（黑上衣的长发女生/黑上衣的男人）`，
  **一律短语式、5~8 条封顶**；正例用句子式「一名穿黑色上衣的男生」（比裸短语「穿黑色上衣的男孩」
  颜色权重高得多，单此一项消掉最强 FP）。
- 禁忌：整句式负例（结构骨架撞车，杀自家 TP，实测 margin -0.1）；颜色枚举过多
  （负例"蓝色上衣"会撞真 TP 的"蓝色牛仔裤"下装）。
- 验收标准：拿真实误报描述回测，FP 不再命中且所有真样本 margin ≥ 0.05。
- 现行配置（已生效）：
  - 黑衣服男生 pos=[一名穿黑色上衣的男生, 一名穿黑色上衣的男人] neg=8条(原6+灰上衣男人+黑上衣长发女生)
  - 黑衣服的女生 pos=[一名穿黑色上衣的女生, 穿黑色上衣搭配短裙的女人, 一名穿黑色上衣的女人] neg=8条(原6+灰/蓝上衣女人)

### 5.7 修复的坑（防止复发）
1. **build.sh rsync 排除 `'build*'` 误伤** `common/build-info.cpp.in`、`cmake/build-info.cmake`
   → 容器内 cmake 必炸。已改目录级 `'build*/'`；llama_buildsrc/ 已补齐缺失文件。
2. **Bash 里 `pkill -f "<含路径的模式>"` 会匹配工具包装 shell 自身 cmdline**，把会话打死
   （exit 144）。用行尾锚定 `模式$` 规避。
3. data/ 目录当前属主/内容有漂移（alerts 图片目录曾被清空过），需要历史告警时以
   `web_vllm/data`（Python 版）为存档参考。

### 5.8 当前运行状态与速查
- **2026-09-29 清库**：按用户要求只保留源码，已删 build*/、deploy/、deploy-cpu/、
  打包 tar（~480M）。要跑先重建：CPU `./build_cpu.sh && ./deploy-cpu/run_web.sh`；
  GPU `./build_host.sh && ./deploy/run_web.sh`。服务当前**未运行**。
- 起/停 CPU 版：`./deploy-cpu/run_web.sh` ／ `pkill -f "deploy-cpu/web_vllm_cpp$"`
- 切 GPU 版：`pkill -f "deploy-cpu/web_vllm_cpp$"; ./deploy/run_web.sh`
- 改规则（免重启，CPU 上锁竞争约等 1 帧）：
  `curl -X PUT :5003/api/rules/<id> -H 'Content-Type: application/json' -d '{...}'`
- 服务日志：`/tmp/claude-*/…/tasks/*.output`（本会话起的）或 nohup 目标文件
