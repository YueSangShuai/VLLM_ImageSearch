# web_vllm_rknn：RK3588 NPU（rkllm）路线

目标：把 web_vllm_cpp 的推理栈迁到 RK3588 NPU（RKNPU2），服务功能对齐现 CPU 版。
进度记录统一写在本文件末尾「进展记录」一节（简单条目，一条一事）。

## 事实与约束
- llama.cpp 无 RKNN 后端，rkllm 路线 = 换推理栈（VlmDescriber 需按 rkllm C API 重写）
- 模型从 HF safetensors / ONNX 转换，GGUF 用不上；**微调模型目前只找到 GGUF，HF 权重待定位**
- 微调模型为 Qwen3-VL-2B 级（F16 GGUF 3.4GB），尺寸适合 3588 NPU
- 第一道闸门（决定路线存亡）：rknn-toolkit2 能否转换你的微调 Qwen3-VL（视觉塔算子覆盖）

## 阶段计划

### Phase 0 可行性（阻塞项，最先做）
- [x] 定位/重导出微调模型 HF 权重 → web_vllm_rknn/models/qwen3vl-ft-merged/
- [x] x86 主机安装 rknn-toolkit2 + rkllm-rt + 参考 rknn_model_zoo 的 Qwen2-VL/Qwen3-VL demo
- [x] 基座 Qwen3-VL-2B 试转 .rkllm，3588 上跑通（2026-10-15：换内核 rknpu 0.9.8 后端到端输出连贯 ✅）
- [x] 微调权重试转（转不动则回退 Vulkan/继续 CPU 方案）→ x86 侧双产物均成功，路线存活

### Phase 1 模型转换线
- [x] LLM → .rkllm（实际 w8a8）；视觉塔 ONNX → .rknn（基座/微调双产物均已上板验证 2026-10-15）
- [ ] YOLOv8n、BGE 一并转 RKNN 上 NPU —— YOLO ✅（2026-10-16 golden 对齐）；BGE 仍 ORT-CPU（动态 seq 需分桶）
- [ ] golden 数值对齐（沿用 web_vllm_cpp/golden 的做法，CPU 输出为基准）

### Phase 2 C++ 移植
- [x] 从 web_vllm_cpp 拷贝通用模块（2026-10-16 全量复用，matcher 等零改动）
- [x] VlmDescriber → RKLLM 后端（同名类同签名接口级替换，回调流式 + per-run sampling）
- [ ] Detector / TextEmbedder 的 ORT 调用换成 RKNN C API —— Detector ✅（2026-10-16 golden ±1px）；TextEmbedder 仍 ORT-CPU（BGE 动态 seq 需分桶）
- [x] CMake aarch64 交叉线跑通（x86 线做 -fsyntax-only 全量通过；rkllm 库仅 aarch64，链接只在交叉线）

### Phase 3 平台上板
- [x] output 产物部署 3588，端到端告警链路验证（2026-10-16：E2E 规则注入实测，describe+落盘全通）
- [x] bench 程序移植，出 NPU 耗时基线（VLM 7.85s/生成、YOLO-RKNN 114.7ms；CPU GGUF 对照因板盘不足暂缓）
- [ ] 量化导致的行为漂移 → 重调 matcher 规则/embedding 阈值（吸取 GPU 非确定性教训）

## 风险
- 微调架构转不动（Phase 0 定生死，失败成本最低时止损）
- 2B Q4/Q5 输出质量下降 → 影响规则匹配与检索阈值
- rkllm 单实例串行；3 核 NPU 多实例并行留作后续优化

---

## 进展记录

### 2026-10-14
- 创建 web_vllm_rknn/，定下本计划
- 环境摸底：本机未装 rknn-toolkit2；onnx 1.17 已在；yolov8n-dyn.onnx / bge-small-zh.onnx 现成
- 微调权重定位：基座 /data/yuesang/LLM/Qwen3-VL-2B-Instruct，LoRA 在
  LLaMA-Factory/saves/Qwen3-VL-2B-Instruct/lora/train_2026-07-16-09-44-09/checkpoint-15500
  ⚠️ 假设 15500 即线上 qwen3vl-ft-20260909 GGUF 的来源（训练实际只跑到 15510/29830 步），若不符请换 checkpoint
- LoRA merge 完成：llamafactory-cli export（CPU）→ models/qwen3vl-ft-merged/（2 shard, ~4GB）
- 环境就绪：conda env `rknn`(py3.10) 装好 rknn-toolkit2==2.5.0 + torch==2.4.0+cpu，`from rknn.api import RKNN` 通过
  ⚠️ 装 toolkit 前先手动装 `torch==2.4.0+cpu`（--index-url download.pytorch.org/whl/cpu），否则会连带拉 ~2.5GB CUDA 栈（本机用不上）
  代理：http://10.8.18.5:7890
- 下一步：拉 rknn_model_zoo Qwen3-VL demo，先用基座 /data/yuesang/LLM/Qwen3-VL-2B-Instruct 试转验证可转性

### 2026-10-14（repo 选型纠正 + 转换流程确认）
- ❌ 走错两次：`rknn_model_zoo`（仅 CV，无 VL）；`rknn3-model-zoo`（用 rknn3_* API，面向 **RK1828 外置协处理器**，非 RK3588 内置 NPU）——均已移出目录
- ✅ 正解 = `airockchip/rknn-llm` **v1.3.1**：官方支持 **Qwen3-VL**，且有现成 RK3588 2B 示例
  （qwen3-vl-2b-instruct_w8a8_rk3588.rkllm / qwen3-vl-2b_vision_rk3588.rknn）。正好我们的模型=Qwen3-VL-2B、平台=RK3588
- 转换分工（两条线，版本互斥 → **两个 conda env**）：
  - 视觉塔：`rknn-toolkit2`(≥2.3.2) + `transformers==4.57.0` + onnx==1.18.0
    → export_vision.py --model_name=qwen3-vl --height=448 --width=448 再 export_vision_rknn.py
  - LLM：`rkllm-toolkit` 1.3.1（torch==2.6.0 + transformers==5.8.0）
    → export_rkllm.py --target-platform rk3588 --num_npu_core 3 --quantized_dtype w8a8 --device cpu
- 关键：rknn-llm 无 --lora 选项 → 必须喂 **合并后模型**（已就绪 models/qwen3vl-ft-merged/）✓
- 视觉输入固定分辨率(448×448)，线上任意 bbox ROI 需 resize 到该尺寸（反而更确定）
- 板侧：rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so + rknpu-driver；
  C++ 改造 = 用 deploy/ 的 vision-rknn + rkllm 生成封装，替换 web_vllm_cpp 的 VlmDescriber(mtmd)，
  matcher/monitor/server 全部复用
- 待办 env：给 rknn env 装 transformers==4.57.0；新建 rkllm env(装 1.3.1 wheel + torch2.6.0cpu)

### 2026-10-14（两 env 装齐，pip 源改用清华镜像）
- rknn env 曾重建（py3.10.22 不变），旧 toolkit2 丢失后已按 requirements_rknn.txt 重装：torch 2.4.0+cpu + rknn-toolkit2 2.5.0 + transformers 4.57.0，`from rknn.api import RKNN` 通过
- ⚠️ requirements_rknn.txt 的 `tokenizers>=0.20,<0.22` 与 transformers 4.57 冲突，已改为 `>=0.22,<0.23`
- rkllm env 补装 rkllm_toolkit-1.3.1 wheel（torch 2.6.0+cpu / transformers 5.8.0 原已就绪），`import rkllm` 通过
- pip 统一走 https://pypi.tuna.tsinghua.edu.cn/simple；仅 torch +cpu 版走 download.pytorch.org/whl/cpu
- ⚠️ pip 依赖检查提示 fgclip2/ultralytics 缺依赖 → 用户 site（~/.local/lib/python3.10）会串进这两个 py3.10 env，转换若有诡异包冲突先想到它
- 两 env 均自检通过，可开跑 Phase 0 视觉塔 ONNX 导出闸门

### 2026-10-15（Phase 0 基座转换双产物落地 → 微调模型转换进行中）
- 视觉塔闸门通过：基座 export_vision.py → onnx/qwen3-vl_vision.onnx(1.6G) → export_vision_rknn.py → rk3588 .rknn(816M)，输出 [196,2048]@448×448
- ⚠️ RKNN 转换需 `setuptools<81`（rknn-toolkit2 依赖 pkg_resources，setuptools 82+ 已移除）
- LLM → RKLLM 踩大坑：build/量化全过、**export 阶段报 "Not enough free space to write 103743488 bytes"**，磁盘/inode/swap 均排除（swap 假设经实测证伪）
  - 根因：toolkit 用 `O_TMPFILE` 在 **CWD** 建匿名临时文件并做 statfs 预检查，在 **/data 的 10T xfs 卷**上算错空间（官方 issue #297/#98 同症状，报错者全是大容量/非 ext4 卷）
  - ✅ 解法：**CWD 换到 ext4 卷**（/tmp）再跑 export_rkllm.py 即成功；产物再 mv 回 /data 不影响（仅转换预检查受影响）
  - 诊断过程留档：write() 全成功→非真写盘失败；fd6=F_DUPFD(O_TMPFILE "./")；报错字节恒为 99MB（模型结构决定）
- 量化校准数据：make_input_embeds_for_quantize.py 生成 data/llm_inputs.json（README 写 inputs.json，需 `ln -s llm_inputs.json inputs.json`）
- 产物：models/rk3588/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm(2.3G) + qwen3-vl_vision_rk3588.rknn(816M)
- 下一步：models/qwen3vl-ft-merged/（LoRA 合并版）走同流程转微调模型，CWD 务必放 ext4

### 2026-10-15（微调模型双产物完成 → Phase 0 x86 侧全部通过）
- 微调视觉塔：ONNX(1.6G) → rk3588 .rknn(816M)，输出 [196,2048] 与基座一致
- ⚠️ 坑2：LLaMA-Factory 导出 config 用 transformers≥5 新字段 `text_config.rope_parameters`，rknn env 的 4.57 只认 `rope_scaling` → 建 models/qwen3vl-ft-merged-compat/（safetensors 等全部绝对路径软链，config 补 `rope_scaling` 别名）喂给转换脚本，训练产物零改动
- ⚠️ 坑3：sed 重定向写进悬空软链会报 No such file（写前先 rm 目标）
- 微调 LLM：校准数据 20 条 + export_rkllm.py（CWD=/tmp/rkllm_run_ft ext4）→ .rkllm(2.3G) 一次通过，再次验证 xfs 根因结论
- 产物入库 models/rk3588/：基座/微调各 vision .rknn(816M) + w8a8 .rkllm(2.3G)，共 6.1G
- Phase 0 完成；下一步 Phase 2/3：aarch64 交叉编译 deploy demo，上 3588 跑通基座端到端（infer.py → C++ demo），再对比微调输出质量（golden 对齐）

### 2026-10-15（deploy demo aarch64 交叉编译通过 + 板上包就绪）
- 复用 web_vllm_cpp 的交叉工具链（third_party/aarch64-cross/wrap + aarch64-linux-gnu.toolchain.cmake，gcc-11 cross，免 root deb 解包）
- ⚠️ 坑4：web_vllm_cpp/setup_cross_toolchain.sh 解包时把 libgomp.spec/libgomp.a 连带改名成 *.unused → 交叉 -fopenmp 直接 fatal；已恢复原名（影响面仅 cross sysroot，web_vllm_cpp 自身不用 OpenMP，无感）
- demo/imgenc/audioenc 三个 aarch64 ELF 编译通过（RK3588 依赖：opencv 3.4.5 prebuilt + librknnrt + librkllmrt，仓库自带）
- 板上包 output_rk3588/：demo + imgenc + lib/{librknnrt,librkllmrt}.so + models/（4 个真实文件副本，勿用软链防 rsync 拍平）+ run_base.sh / run_ft.sh（demo 参数：image vision.rknn "" "" llm.rkllm 512 2048 3rknn核 rk3588）
- 待板上验证：rknpu 驱动版本需与 librknnrt 2.5.0 兼容、librkllmrt 需 rkllm-runtime 1.3.1 配套；板上直接 ./run_base.sh 即可

### 2026-10-15（板上实测：视觉塔跑通，LLM 被旧驱动卡死 → 需刷固件）
- 板：NanoPi R6S @10.8.18.118（pi/pi），Ubuntu 22.04，8G RAM，kernel 5.10.110（2022-10 FriendlyElec 固件），根盘 overlay 仅剩 ~5.7G（模型已占 6.1G，别再往板上堆东西）
- 板上包：~/output_rk3588（6.1G 已传齐，tar-over-ssh 流式传的，未占双倍空间）
- ✅ 视觉塔在驱动 0.8.2 上完全正常：448×448 单帧推理 基座 1952ms / 微调 2246ms，输入输出 tensor 元数据均符合预期
- ❌ RKLLM w8a8 生成阶段报 `E rkllm: matmul(w8a8) run failed`（LLM 加载 init 成功、vision 正常，纯执行挂）——runtime 自带警告 "driver too low, upgrade to 0.9.7"
- ⛔ **rknpu 驱动 0.8.2 是 builtin 编译进 kernel 的**（`modinfo rknpu` → filename: (builtin)），换 .ko 无门；板上无内核 headers、apt 源只有 jammy-ports（无 FriendlyElec kernel 包）、外网不通
- **结论：唯一出路 = 刷 FriendlyElec 2023+ 固件**（新镜像 kernel 带 rknpu ≥0.9.x；rknn-llm/rknpu-driver/ 里自带的 0.9.8 包是内核源码，同样指向上游内核而非热补丁）
  - 刷机是破坏性操作（重刷系统盘），需要用户决定：sd-fuse_r6s 重做镜像 or 官方整站镜像重写；刷完把 ~/output_rk3588 重新拷入即可直接复跑 run_base.sh / run_ft.sh
- 待验证（板刷完固件后）：./run_base.sh → 交互输入 1（"What is in the image?"）应输出基座描述；再 run_ft.sh 对比微调输出
- 其他实测经验（补）：转换 env 全程加 PYTHONNOUSERSITE=1；板上无需装任何依赖（opencv 已静态进 demo，libgomp 系统自带）

### 2026-10-15（原地换 eMMC kernel 分区方案就绪，不刷机不动 rootfs）
- 前提变化：主机代理可用（10.8.18.5:7890）→ 板上 SSH 正常、旧系统其实完好；用户无 SD 卡 → 定路线：**只换 eMMC boot-chain 分区（dtbo/resource/kernel/boot），rootfs 与 6.1G 模型不动**
- 镜像源：官网 NanoPiR6S 页只有百度网盘 ❌ → 正解 `downloads.friendlyelec.com/os-images/rk3588/images/`（全量镜像 tgz）
  - 已取 `ubuntu-jammy-minimal-arm64-images-20280806.tgz`（sha256 校验 ✓）→ kernel **6.1.141**，KRNL 格式与现板 kernel 分区一致；parameter.txt 分区表与板上 GPT 完全吻合
- ⚠️ 坑5：rootfs.img 是 Android sparse 且 chunk type 非标准（0xcac1=RAW / 0xcac3=DONT_CARE）；pip 无 simg2img、apt 包是空壳 → 手工按该变体解析转 raw（3G ext4）✓
- 新内核 **r8125（PCIe RTL8125 网卡，eth1 就是它）不在内核里**，只换 kernel 不补模块 → 重启即失联。已从 raw rootfs 用 debugfs rdump 提取 `/lib/modules/6.1.141`（120M，含 extra/r8125.ko）打成 modules-6.1.141.tar.gz
- 已就绪（板上）：
  - `~/firmware/{dtbo,resource,kernel,boot}.img`（与主机 sha256 一致）
  - `~/modules-6.1.141.tar.gz`
  - `~/board_backup.sh`（备份旧分区→ /data/part-backup/，已跑）
  - `~/board_flash.sh`（前置校验备份完整 → 写 4 分区 → 回读校验，失败自动回滚 → 装 modules + depmod）
  - 旧分区备份双份：板上 /data/part-backup/ + 主机 firmware/backup/（sha256 双方核对一致）
- 执行（用户自行操作）：
  1. `ssh pi@10.8.18.118 'echo pi | sudo -S bash /home/pi/board_flash.sh'` → 看到 "ALL DONE, safe to reboot"
  2. `ssh pi@10.8.18.118 'echo pi | sudo -S reboot'`（等 ~1-2 分钟）
  3. 验证：`uname -r`=6.1.141；`dmesg | grep -i rknpu` driver_version ≥0.9.x；然后 cd ~/output_rk3588 && ./run_base.sh
  - 救砖预案见 RECOVERY.md（USB-C maskrom，无需 SD）
- 主机侧工具留档：ssh_pi.py / scp_pi.py（pexpect 密码自动化，无 sshpass；scp push 可用、pull 有 "Is a directory" 怪癖改用 sftp）；多变量循环命令易被多层 shell 转义吃掉，一律写脚本文件推上去跑

### 2026-10-15（原地换内核成功 → RKLLM 端到端跑通，Phase 0/3 闸门全过）
- board_flash.sh 全流程通过（写 4 分区 + 回读校验 + 装模块），重启后 kernel 6.1.141、rootfs/overlay/6.1G 模型原封不动
- `run_base.sh` 实测：**rkllm-runtime 1.3.1 + rknpu driver 0.9.8 + RK3588 3核**，LLM 加载 8.2s，视觉塔推理 1877ms（与 0.8.2 时代持平），基座 w8a8 中文描述输出连贯、细节完整（宇航员/可乐箱/地球全对）
- 附带收益：新内核自带 rknpu 0.9.8，无需任何驱动热补丁；旧 boot-chain 双份备份（板上 /data/part-backup + 主机 firmware/backup）
- ⚠️ 注意点（板上 demo 打印）：img_start/img_end 等 special token 用的是默认占位符，C++ 移植 RkllmDescriber 时需按 Qwen3-VL 实际 chat template 设置
- 下一步：run_ft.sh 对比微调输出质量 → golden 数值对齐 → Phase 2 C++ 移植（VlmDescriber → RkllmDescriber）
- run_ft.sh 实测：微调 w8a8 输出**训练分布短描述风格完整保留**（"在月球表面，一位宇航员…坐在一个绿色的箱子上…背景是地球和星空。"），对比基座同图长篇分点作文 → LoRA 效果未被 W8A8 量化破坏；两模型 vision 推理均 ~1.85s@448×448，加载 ~8s
- **Phase 0 彻底完成（x86 + 上板双闭环）**。遗留技术点：demo 的 special token 占位符、视觉输入固定 448×448（线上 ROI 需 resize）、max_context 4096

### 2026-10-16（Phase 2 C++ 移植完成 → 全服务板上跑通真实摄像头）
- 新项目 `cpp/`：web_vllm_cpp 通用模块全量复用（matcher/monitor/server 等**零改动**），
  `vlm_describer.{h,cpp}` 整体重写为 RKLLM 后端（同名类同签名，接口级替换）：
  - `rknn_image_enc.{h,cc}` = demo image_enc 精简版（deepstack 交错拼接布局）
  - describe()：ROI→resize_bilinear 448×448 → run_imgenc → RKLLM_INPUT_MULTIMODAL（prompt 文首拼 `<image>`，
    special token 用 env 可覆盖 VLM_IMG_START/END/CONTENT，默认 Qwen3-VL 的  vision_start/ vision_end/ vision）
  - 回调流式累加 + mu_ 串行；per-run 采样用 `RKLLMInferParam.sampling_params`（top_k/temp/repeat 随 describe 参数走）
  - config 字段不换名：vlm_model_path→.rknn、mmproj_path→.rkllm（env 同名 VLM_MODEL_NAME/VLM_MMPROJ_PATH）
- 交叉编译：复用 web_vllm_cpp 工具链，**新坑6：sysroot 缺 libstdc++.so.6 运行时**（setup_cross_toolchain.sh 当年漏下
  libstdc++6-arm64-cross deb；ORT 静态 C++ 所以 x86 线没暴露）→ 手动 apt-get download + dpkg-deb -x 进 root/，已修复
  - `cpp/build_scripts/build_aarch64.sh` 一条命令出 `output_cpp/`（bin+lib+data+static+models+run_web_rk.sh）
- **板上首跑即通**（~/output_cpp，服务端口 5003）：
  - `VlmDescriber(RKLLM) ready: vision=448x448 tokens=196` ✓ YOLO/BGE(ORT-CPU) ✓
  - 自动拉起 3 设备监控，"10楼食堂" 真实摄像头行人检测持续出帧 ✓；另两路读取超时=板子网段问题，与移植无关
  - 冒烟注意：run_web_rk.sh 要用 bash 跑（板上 sh=dash 不认 pipefail）
- 待办：告警链路端到端验证（规则命中→描述→推送；日志里还没出现 describe 调用行）、bench 移植、
  板时钟比主机慢 ~35h（建议 chrony 或手动 date 同步，TLS 场景会踩雷）

### 2026-10-16（bench 移植完成 → RK3588 三模型基线）
- bench_models 移植到 cpp/（改文案即可，接口全同）；`run_bench_rk.sh` 启动；⚠️ 跑前先停在线服务（NPU 独占）
- **RK3588 基线（W8A8 微调模型 / 8核 ORT-CPU）**：
  | 模型 | 加载 | 单次推理 | 备注 |
  |---|---|---|---|
  | YOLOv8n (ORT CPU) | 188ms | 146.5ms | 6.8 fps |
  | BGE (ORT CPU) | 550ms | 单条 6.9ms / batch5=44ms | 512 维 |
  | Qwen-VL (NPU) | 8062ms | **7853ms**（120 token 生成） | 含视觉 1.9s；48.5 字/秒 |
- 观察：NPU 生成 ~48 字/秒是链路瓶颈；线上 combined 每帧 ~2 次生成 → 单帧告警路径 ~16s，
  降 token / rule_direct 模式 / 视觉塔复用（同帧两调用重复编码 2×1.9s，**可做：embedding 缓存**）
- CPU 版对照基线：web_vllm_cpp/output/aarch64/bench_models 已编好没跑过（需要 GGUF 上板 5G，盘不够）→ 暂缓

### 2026-10-16（YOLO → RKNN NPU：Detector 后端切换，转换/接口三坑）
- `cpp/tools/convert_yolo_rknn.py` + `trim_yolo_onnx.py`；`detector.cpp` 双后端 `#ifdef WEBVLLM_RKNN`（ORT 路径原样保留）
- ⚠️ 坑7 **trim 反而不能转**：6 输出 ONNX 直接转换正常（toolkit 融合正常），裁成单输出后所有 optimization_level
  都在 `_p_seperate_conv_rs_split` IndexError → 放弃 trim，板侧 `rknn_outputs_get` 只取 index 0（其余 5 个是中间张量残留）
- ⚠️ 坑8 **输入布局**：产物输入是 NHWC；float NCHW 喂入报 "unsupported src layout for normalize" 且连锁
  outputs_get param invalid。改 **uint8 NHWC 直喂**，/255 用转换时 mean_values=0/std_values=255 折进模型
- ⚠️ 坑9 **onnxsim 装不上**：pip 把 rich 装进 ~/.local（user-site 串包老坑），env 内 `import rich` 失败；
  最终不需要 onnxsim（见坑7），但 `pip install --prefix <env>` 也无效（装到 lib/site-packages 错位）
- ⚠️ 坑10 **scp_pi.py 忘带 push 子命令 = 静默变 pull**，把板上旧文件反下载回本地且显示成功；推送一律走 `push_cpp.sh`
- 状态：新产物（7MB fp16，mean/std 内置）+ 新二进制已重编，待上板验证 YOLO 输出与 ORT golden 对齐后并入服务

### 2026-10-16（YOLO RKNN golden 对齐达成：±1px / conf ±0.004）
- ⚠️ 坑11 **outputs_get 必须"全量请求 + want_float=1"**：只请 1 个（哪怕 index 0）报
  `param output index(127) is out of range(6)`（127=库内部对中间张量输出的原生 id，与传入 index 无关）；
  此前 "want_float=1 稳定 PARAM_INVALID" 是 normalize 连锁失败时的误判
- 板上 yolo_probe vs `web_vllm_cpp/golden/yolo_golden.json`：RKNN 检出 [279,2,318,113]@0.6709 vs
  golden [279,1,318,113]@0.6750 —— fp16 NPU 正常漂移，判定通过
- 新增 `yolo_probe` 工具（CMake 第三目标）打印检测框；`push_cpp.sh` 固化推送+md5 核对

### 2026-10-16（告警链路端到端验证通过 ✅：YOLO-RKNN 检测 → RKLLM describe → BGE 命中 → 告警落盘）
- 方法：`e2e_add_rule.sh` 注入必过规则（正例"一个人在画面中"、threshold 0.01，先备份 tasks.json）
  → 真实摄像头检出 → 告警落盘 `data/alerts_images/<日期>/<设备>/E2E冒烟/{original,roi}/` + alerts_history
- **VLM describe 实锤**：告警记录含 `base_description`（英文基础描述）+ `description`
  （微调中文短描述风格"该行人背对镜头，穿着白色T恤和黑色裤子…"）——RKLLM 后端 describe 在真实链路工作
- BGE 向量命中 positive_score 0.4859；规则命中路径与 CPU 版行为一致
- 复原用 `e2e_rm_rule.sh` / `e2e_finalize.sh`（bench + 清测试告警 + 重启）
- ⚠️ 服务 stdout 是块缓冲，检测行在 web.log 里会延迟出现；验证一律看 data/ 落盘文件与 REST，不看 web.log
- ⚠️ /api/alerts 与落盘文件不同步（返回空但 alerts_history 有记录）——REST 语义待查（低优先）
- NPU YOLO 基线（bench --skip vlm，服务停机测）：**加载 104ms / 单次 114.7ms（8.72fps）**，
  对比 ORT-CPU 基线 188ms/146.5ms(6.8fps) → 推理提速 22%、加载快一倍；BGE 仍 ORT-CPU（单条 8.0ms/batch5 53.2ms）
- ⚠️ 坑12：ssh 内联命令里的 `pkill -f xxx` 会匹配 ssh 自身命令行 → 自杀断连（exit 255）；带 pkill 的复合命令一律写脚本文件推上跑
  （`e2e_finalize.sh`/`bench_rk_board.sh` 已固化：停服务→bench→自动重启）
- 板上现场已复原：tasks.json 还原（测试规则 rule_e2e_test 与 23 条测试告警已清）、服务运行中（5003）

### 2026-10-19（CPU vs NPU 同机对照表，RK3588 8核 ORT-CPU vs NPU，方法一致）
| 模型 | 后端 | 加载 | 单次推理 | 相对 CPU |
|---|---|---|---|---|
| YOLOv8n | ORT-CPU | 224ms | 145.1ms (6.89fps) | 基准 |
| YOLOv8n | **RKNN-NPU (fp16)** | 104ms | **114.7ms (8.72fps)** | 推理 ×1.27，加载 ×2.2 |
| BGE-small-zh | ORT-CPU | 202ms | 单条 7.2ms / batch5 46ms | 无 NPU 版（待转） |
| Qwen3-VL-2B | **RKLLM-NPU (w8a8)** | 8062ms | 7853ms（120 token 生成，含视觉 1.9s） | 板上 CPU 不可测（GGUF 装不下盘）；x86 生产 CPU combined 十秒级/帧 |
- 脚本：`bench_rk_board.sh`（NPU）/ `bench_cpu_board.sh`（CPU，需板 `~/wvc/lib` = web_vllm_cpp aarch64 llama libs）
- 解读：YOLO 收益有限（2.9M 参数不撞算力瓶颈，且仍是 fp16——转 int8 常规还能再 ~2×）；
  BGE 单条 7ms 非瓶颈，NPU 化优先级低；链路真正瓶颈仍是 VLM 生成（combined 单帧 ≈ 0.1 + 2×7.9 ≈ 16s）

### 2026-10-19（VLM CPU 基线补测完成——盘其实够，之前"装不下"结论作废）
- GGUF f16（3.3G 模型 + 0.78G mmproj）推上板（剩余 5.1G，占至 96%，测完已删）。
  脚本 `bench_vlm_cpu_board.sh`（CPU）/ `bench_vlm_rk_board.sh`（NPU 同参补测），均 flock 单实例 + 跑完自动重启服务。
- **VLM 对照（同板、服务停机、同 256×256 合成图、max_tokens=120）**：
  | 后端 | 加载 | 单帧生成 | 吞吐 |
  |---|---|---|---|
  | llama.cpp f16 CPU (8核) | 3.8s | **31660ms** (~3.8 tok/s) | 17.5 字/s |
  | RKLLM w8a8 NPU (3核) | 12.0s | **8157ms** (~14.7 tok/s) | 46.7 字/s |
  → NPU 端到端 **×3.9**（剥掉视觉+prefill 固定开销后 decode 约 ×3.8），且这是 w8a8 换 f16 的综合结果；
    CPU 加载反而快 3×（mmap 即返回 vs rkllm 拷进 NPU 内存），冷启动体验 CPU 占优。
- ⚠️ **重大测量陷阱（已修）**：`bench_*_board.sh` 旧 pkill 模式 `output_cpp/web_vllm_rknn` 匹配不到
  实际进程名 `./web_vllm_rknn`（exec 相对路径启动），服务根本没停过——首发 CPU bench 被服务抢核，
  单轮虚高到 11 分钟/轮（真实 32s）。三个 bench 脚本已全部改 `pkill -x web_vllm_rknn`。
  此前 YOLO/BGE 的 NPU 基线若经该脚本测得，同样存疑，必要时重跑（bench_rk 全程仅 1 分钟）。
- 板上现场：GGUF 已删（盘回 5.1G free），服务 5003 运行中。

### 2026-10-19（模型盘点 & RKLLM 多模态支持调研）
- **两个 vision .rknn 实测等价**：`qwen3-vl_vision`（基座导出）与 `qwen3vl-ft_vision`（微调合并版导出）
  的 HF 源权重全量 diff——315 个 `model.visual.*` 张量 **0 差异**（LoRA 只加语言层 adapter，视觉塔冻结）。
  md5 不同仅是 rknn 打包元数据。两对模型的真实差异全在语言塔（基座长文分点 vs 微调短描述风格）。
  空间紧张时可只保留 ft 版省 816M。
- **qwen3vl-ft 构成**：视觉塔 24 层 ViT hidden1024 / 407M + 语言塔 28 层 Qwen3 hidden2048 / 1720.6M
  （tie embeddings 无独立 lm_head），bf16 原始 **4.05GB**。部署即拆两产物：vision→rknn-toolkit2（.rknn 816M）、
  语言塔→rkllm-toolkit（w8a8 .rkllm 2.3G）；GGUF f16 路线（3.3G+782M）只是换封装未量化。
- **板端在线服务实际加载 4 模型**：ft vision rknn + ft rkllm（NPU 3核）+ yolov8n rknn（NPU）+
  bge-small-zh.onnx（唯一 ORT-CPU，7ms/条非瓶颈）；基座对在 ~/output_rk3588/models/ 仅 run_base.sh A/B 用。
- **RKLLM（本地 v1.3.1）多模态支持**（README + SDK 1.3.1 PDF + export_vision.py choices）：
  - 全流程官方 demo：Qwen2-VL-2B/7B、Qwen2.5-VL-3B、**Qwen3-VL**、Qwen3.5、InternVL3-1B（3.5 v1.2.3）、
    DeepSeek-OCR、Gemma-4-E2B（唯一带音频）；脚本支持名：minicpm-v-2_6 / qwen2_5-vl-3b / qwen3-vl /
    qwen3.5 / smolvlm / internvl3-1b / deepseekocr。
  - 结构：多模态=rknn-toolkit2 转视觉塔 + rkllm-toolkit 转语言塔，两条链自由拼；
    量化选项 w4a16/w8a8（各含分组）、GPTQ 直转；平台 RK3588/3576/3562/RV1126B。
  - 最小的几个：SmolVLM 256M/500M < LFM2.5-VL-450M（无视觉导出脚本，接入最难）< Qwen3.5-0.8B
    < InternVL3-1B(~0.9B) < Gemma4-E2B(有效2B)。**Qwen3.5 是多模态**（README demo 有
    Qwen3.5-0.8B_vision_rknn + _w8a8_rkllm 成对；export 类 `qwen3_5_vl_vision` 取 vlm.visual），
    且 patch16/merge2/temporal2 与 Qwen3-VL 同款 → 换型管线可复用，但 0.8B 含视觉塔，
    语言塔只剩 ~0.4-0.5B，微调前务必先转未微调版上板用告警图验描述质量（预估生成 8.2s→3-4s，
    combined 帧 16s→8s 内）。

### 2026-10-19（Qwen3.5-0.8B 实测：转换全通，板端疑似不可用——线性注意力疑似不吃 RK3588 NPU）
- 模型：`Qwen/Qwen3.5-0.8B`（hf-mirror 下载 1.75G → /data/yuesang/LLM/Qwen3.5-0.8B）。
  确认为多模态（Qwen3_5ForConditionalGeneration）：视觉塔 12 层 ViT-768（patch16/merge2，out 1024）
  + 语言塔 24 层 hidden1024 **混合线性注意力**（每 4 层才 1 层 full attention，gated deltanet 风格）。
- x86 转换全部一次通过（rkllm env = transformers 5.8 有 qwen3_5；rknn env 的 4.57 没有，ONNX 导出必须在 rkllm env 跑）：
  - 视觉：export_vision.py --model_name=qwen3.5 → ONNX(403M, [196,1024]) → rknn **210M**（对比 ft vision 816M）
  - 校准：work_q35/ 复用 ft 的 20 条行人数据集（勿覆盖 data/ 原产物；inputs.json 软链必须用绝对路径，相对路径会悬空）
  - 语言塔：export_rkllm.py（CWD=/tmp scratchpad）→ w8a8 **1.3G**（vocab 248k 导致 embeddings 偏大）
  - 产物入库 models/rk3588/（qwen3.5_vision_rk3588.rknn + qwen3.5-0.8b_w8a8_rk3588.rkllm），已推板 ~/output_rk3588/models/
- ⚠️ **deploy demo 是交互 REPL**（main.cpp `while(true)`+getline，喂 `<image>提示词` 才走视觉分支，`exit` 才退出）：
  非交互直跑 = 空 prompt 死循环（"Hello!…" 是纯聊天回复，图像从未进模型）。首轮 32 分钟"生成"即此假象。
  正确姿势：`printf '<image>提示词\nexit\n' | ./demo ...`（bench_q35_vs_ft_board.sh 已固化，带超时保护）。
- 负面信号（未复测证实）：即便空聊天单轮（~15 token）qwen3.5 也 ~5 分钟（<0.1 tok/s）；
  NPU load 100%、CPU 也 60-155%，疑似线性注意力层 decode 未获 RK3588 NPU 优化
  （官方 changelog decode 优化只提 RK3576）。**初步结论：Qwen3.5 线不适合 RK3588，别投入微调**；
  板上单图正式对比（喂对 prompt、600s 超时）待跑。
- ⚠️ 意外状态待查：测量尾声发现板上 `~/output_rk3588/` 目录整体消失（服务进程仍在内存运行；
  原因未查，可能是 overlay/固件自动恢复或误删——下次操作板子前先核实，含刚推的 qwen3.5 模型文件）。

### 2026-10-19（npu_models/ 软链合集目录）
- 新建 npu_models/，集中 7 个 NPU 产物软链（在线 3 件套 + 基座对 + Qwen3.5 对），`du -L` 7.5G，方便一次性传输
- ⚠️ 传输必须解引用：scp 默认可用；rsync 加 -L、tar 加 -h（软链直传=断链，见 2026-10-15 拍平坑）

### 2026-10-19（清理：output_rk3588/models/ 去重，省 6.1G）
- 删除 output_rk3588/models/ 下 4 个副本（2×rkllm + 2×rknn），md5 逐一比对与 models/rk3588/ 完全一致后才删；目录留空占位
- 重推板子前需先回填：`cp models/rk3588/{qwen3-vl-2b-instruct_w8a8,qwen3vl-ft-merged_w8a8}*.rkllm models/rk3588/qwen3-vl*_vision_rk3588.rknn output_rk3588/models/`（勿用软链，防 rsync 拍平，见 10-15 坑）
- 其余大文件本次评估后保留：ONNX 中间产物 3.7G、rootfs.raw.img 3.2G、jammy-minimal.tgz 1.1G（用户选择不动）

### 2026-10-19（根目录脚本归档进 scripts/，按功能分组）
- 新布局：`scripts/bench/`（5 个 bench + stop_all）、`scripts/deploy/`（push_cpp.sh + scp_pi.py + ssh_pi.py）、`scripts/e2e/`（add/rm/finalize）、`scripts/firmware/`（board_backup/board_flash）
- 仅 push_cpp.sh 有路径假设，已修：cd 到 `$(dirname $(dirname $SCRIPT_DIR))`（项目根），py 工具走同目录 $PY；其余脚本要么在板端跑（cd ~/output_cpp，与本地无关）要么无路径依赖
- 板上副本（~/bench_*.sh、~/board_flash.sh 等）不受影响；推送/执行方式不变，只是本地源文件挪了位置

### 2026-10-19（output_cpp 自包含化：单文件夹部署，甩掉 ~/output_rk3588 依赖）
- 背景：板上 ~/output_rk3588/ 已被手动删除；平台不再依赖它
- output_cpp/models/ 加 2 个软链 → ../../npu_models/（本地真源仍是 npu_models，不占双倍空间）；
  run_web_rk.sh / run_bench_rk.sh 的 VLM_MODEL_NAME / VLM_MMPROJ_PATH 默认改指 $HERE/models/
- 新脚本 scripts/deploy/push_cpp_full.sh：`tar cf - -h output_cpp | ssh 'tar xf -'` 一次推整包 3.39G，
  `-h` 把软链解引用成真实模型落板（勿用 scp 推目录：无 -r 且软链直传=断链）
- 板上部署从此两步：`bash scripts/deploy/push_cpp_full.sh`（密码 pi）→ 板上 `cd ~/output_cpp && bash run_web_rk.sh`
- 旧的 push_cpp.sh（只推 3 个二进制）保留作小更新快速通道

### 2026-10-19（告警按日期查看：/api/alerts 支持 ?date=，跨天不再"隐身"）
- 起因：告警按天分文件（alerts_history/alerts_YYYY-MM-DD.json），旧 UI 只读当天 → 历史告警看似"没加载"
- server.cpp：/api/alerts、/api/alerts/clear、DELETE /api/alerts/<id> 全部支持 ?date=YYYY-MM-DD（缺省今天）；
  新增 GET /api/alerts/dates 返回有存档的日期列表（今天永远排第一）；json_store 加 alerts_file_for(date) +
  valid_alert_date 严格校验（防目录穿越）；坑：交叉 gcc-11 默认 std 无 string::ends_with，改 compare()
- 前端真源在 web_vllm_cpp/templates/alerts.html（构建时拷进 output_cpp/static/，直接改 static 会被覆盖！）：
  筛选栏加日期下拉框，列表/删除/清空都跟随所选日期
- 已重新交叉编译并组装 output_cpp/（SKIP_CLEAN=1 增量）；推送用 scripts/deploy/push_cpp_full.sh
- ⚠️ web_vllm_cpp（Python 版）同样有此按天限制，未改

### 2026-10-26（Qwen3.5-0.8B 结论翻案：RK3588 上完全可用，速度反超 ft-2B）
- 板上部署已整体迁至 **/home/web_vllm_rknn/**（新用户目录、扁平命名；二进制 md5 aa98bd3a 与 output_cpp 主版本一致；
  旧 ~/output_cpp、~/output_rk3588 均消失；板重启后服务不自启，bench 脚本负责拉起）。qwen3.5 双模型与 demo 随旧目录丢失，已重推（md5 校验一致，demo 来自
  rknn-llm/examples/multimodal_model_demo/deploy/build_aarch64/demo）。盘剩 2.7G，紧张。
- **10-19 的"线性注意力不吃 RK3588 NPU、单轮 5 分钟"结论作废**——那是 REPL 空 prompt 死循环假象（喂对 prompt 后从未复现）。
- 同图（真实告警图）同 prompt（线上中文描述 prompt）正式对比（bench_q35_vs_ft_board.sh 已修：适配新路径；robot: 回复与无换行
  "user: " 提示同行导致 sed '^robot:' 抓空，改 grep -ao 'robot:.*'；服务 pkill 后测，flock 防重）：
  | 模型 | 单轮总耗时（含加载） | 描述质量 |
  |---|---|---|
  | qwen3vl-ft-2B (w8a8) | 21.5s | 提环境（大厅/现代感，违反 prompt 禁令）、背景行人干扰 |
  | Qwen3.5-0.8B (w8a8 基座) | **13.5s**（加载 4.5+1.0s、视觉 0.8s，decode 仅 ~7s） | 全程聚焦行人、属性细（手持物/发长/朝向），未提环境 |
  → 基座未微调即反超线上微调模型的速度与遵Prompt度。**方向：值得投入 qwen3.5-0.8B 微调**（复用 ft 数据集走 LoRA→合并→rkllm 导出，
  语言塔仅 ~0.5B，转换更快）。
- 待办：①多告警图批量质量对比（含 BGE 命中率对比）②qwen3.5 视觉塔 448×448 vs 现线 448 一致性核对 ③若切换，线上 max_tokens/阈值需重调
  ④板盘 2.7G，装不下新旧两套并行 A/B 全量，需先清 ONNX/中间产物或换盘

### 2026-10-26（多图批量对比完成 + bench 两个脚本 bug 修复）
- 6 张真实告警图（每规则一张）× 双模型对比（板上 /tmp/multi_out.log）：
  | 维度 | qwen3vl-ft-2B | Qwen3.5-0.8B 基座 |
  |---|---|---|
  | 平均单轮（含加载） | 17.6s | **13.0s（×1.35）** |
  | 幻觉 | 时间戳幻觉「10-23-2026 12:33:30」、多人编故事（三套西装细节）、必带环境句 | 看不清时诚实弃权「没有清晰可见行人」（但会带一段环境描述）|
  | 遵 Prompt | 环境句屡禁不止 | 清晰帧下零环境句 |
  | 属性粒度 | 粗 | 细（手表/手机/朝向/步向）|
  → 微调 qwen3.5-0.8B 的主要收益点：压弃权长文与环境段、修「他/她」混用、对齐线上短句风格。数据可复用现有 ft 行人数据集。
- 脚本修复：①bench 末尾拉起服务时子进程继承 flock 的 fd9 → 锁被服务永久持有、下次 bench 全被挡
  「another bench running」（三个用 flock 的 bench 脚本均已加 9>&-）；②ssh_pi 包装下 `pkill -f multi_q35` 会自匹配杀掉自己的会话（255），
  需 `pkill -f "[m]ulti_q35"` 括号技巧；③板上长任务必须 setsid 落 launcher 脚本 + 输出重定向落盘，直接 `setsid nohup … &` 在通道断开时仍会被杀；
  ④告警图含真实行人 PII，禁止 scp 回本地，质量比对只用板端文本输出。

### 2026-10-26（NPU 推理速度对比正式版：Qwen3.5-0.8B vs Qwen3-VL-2B-ft）
- 脚本 scripts/bench/bench_speed_q35_vs_ft2b_board.sh（同图同 prompt，mt=32/384 ×2 遍，双实例先停后自动拉起）。
  坑：demo 只在 /home/web_vllm_rknn/demo，脚本必须用绝对路径（首发 8 轮全 5ms 空跑）。
- 分阶段（page cache 热的第二轮；生成时长 = wall - load - vision）：
  | 阶段 | ft-2B | q35-0.8B | 倍数 |
  |---|---|---|---|
  | rkllm 加载 | 2.1–3.9s | 2.4–4.3s | ≈持平（都吃页缓存）|
  | 视觉编码 448² | ~1.95s | ~0.83s | **×2.3** |
  | prefill+生成(32tok) | 7.8s | 5.1s | ×1.5 |
  | 完整描述(自然 EOS，~66–75 tok) 生成 | 8.9–10.2s | 5.1→6.9s（+34tok 用 1.8s）| |
  | decode 速率 | ~18 tok/s（首遍噪声大 18–38）| ~19 tok/s | ≈持平 |
  | **单帧端到端（新进程含加载）** | **14.9s** | **12.0s** | **×1.24** |
- 结论：decode 速率两者打平（w8a8 下线性注意力没占到便宜也没吃亏）；q35 的优势全在**固定开销**——
  视觉塔小（×2.3）+ prefill 便宜；在线常驻服务（模型已加载）单帧收益≈×1.3–1.5（视觉+prefill），
  demo/冷启动场景 ×1.24。加载时间两模型几乎一样（都受页缓存冷热支配）。

### 2026-10-26（速度矩阵 v2 —— 修正上文粗估：decode 并非打平，常驻每帧两者几乎相同）
- 方法升级：mt∈{16,32,48,64}（均低于两模型自然 EOS 长度，token 数精确=mt，绕开字数换算）×2 遍
  → wall(mt) 最小二乘拟合；另在同一 demo 进程连喂 k=1/3 次同 prompt → 常驻边际帧成本。
  脚本 scripts/bench/bench_speed_matrix_board.sh，日志板上 /tmp/speed_matrix.log。
- 结果：
  | 指标 | ft-2B | q35-0.8B | 结论 |
  |---|---|---|---|
  | decode（拟合斜率） | **34.5 ms/tok ≈ 29.0 tok/s** | **53.1 ms/tok ≈ 18.8 tok/s** | **ft ×1.55**（上午"打平"是字数换算假象：q35 中文 1.36 字/tok vs ft 1.53）|
  | 字/秒（有效产出） | 45 | 26 | ft 快，但 q35 话更少 |
  | 冷启动总耗时（拟合截距） | 12.6s | 8.0s | q35 −37%（加载+视觉+prefill）|
  | 视觉编码 | ~1.95s | ~0.85s | q35 ×2.3 |
  | **常驻每帧（在线服务口径，mt=64）** | **5.09s** | **5.65s** | ft 略快 |
  | 典型帧（折算各自 EOS 长度 74/59 tok） | ~5.43s | ~5.38s | **几乎相同** |
- 修正结论：**在线常驻场景两模型单帧成本相同**；q35 的真实优势 = 冷启动/换型 −37%、视觉塔 ×2.3、
  模型文件 1.3G vs 2.3G（省盘）、以及**文本质量**（零环境句/时间戳幻觉）。"resident ×1.3-1.5"作废。
  微调 q35 若同时压长度+压幻觉，帧时间与质量可双收。
