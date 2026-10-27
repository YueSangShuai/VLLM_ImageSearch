# 构建系统变更记录（web_vllm_cpp）

日期：2026-10-13 ~ 2026-10-14

---

## 一、当前构建体系（最新状态，日常使用只看这节）

### 1.1 目录结构

```
web_vllm_cpp/
├── build_scripts/                  # 所有构建/运行脚本
│   ├── common/
│   │   └── build_onnxruntime.sh    # ORT v1.18.1 源码编译（原生/交叉通用）
│   ├── cpu/                        # x86 CPU 编译
│   │   ├── build_thirdparty.sh     #   三方库：ORT + llama.cpp
│   │   ├── build_app.sh            #   程序 → output/cpu/ + ldd 自检
│   │   └── run_web.sh              #   启动脚本源（拷成 output/cpu/run_web.sh）
│   ├── aarch64/                    # NanoPi R6S 交叉编译（在 x86 主机上编）
│   │   ├── setup_cross_toolchain.sh#   一键装交叉工具链（免 root）
│   │   ├── aarch64-linux-gnu.toolchain.cmake
│   │   ├── qemu-emulator.sh        #   try_run 探测用的 qemu 包装
│   │   ├── build_thirdparty.sh     #   三方库 → *-aarch64 目录
│   │   ├── build_app.sh            #   程序 → output/aarch64/ + qemu 依赖自检
│   │   └── run_web.sh              #   启动脚本源（拷成 output/aarch64/run_web.sh）
│   └── offline/                    # 目标机"原生编译"的离线物料方案
│       ├── fetch_offline.sh        #   有网机器：下载全部源码物料
│       └── apply_offline.sh        #   目标机：接入离线物料（一次性）
├── output/                         # ★ 所有编译产物统一在此的子目录下
│   ├── cpu/                        #   x86 CPU 产物（原 deploy-cpu）
│   └── aarch64/                    #   NanoPi 产物
├── build/                          # 编译中间目录（llama-cpu/app-cpu/llama-aarch64/…）
└── third_party/                    # 三方库源码 + 编译安装目录（见 1.3）
```

### 1.2 使用方式

| 目标 | 步骤 | 产物 |
|---|---|---|
| x86 CPU | `build_scripts/cpu/build_thirdparty.sh` → `build_app.sh` | `output/cpu/`，启动 `output/cpu/run_web.sh` |
| NanoPi R6S（交叉） | `setup_cross_toolchain.sh`（一次性）→ `aarch64/build_thirdparty.sh` → `build_app.sh` | `output/aarch64/`，拷到 NanoPi 同布局运行其 `run_web.sh` |
| NanoPi R6S（目标机原生编） | 见 §3.6 离线物料方案 | 目标机上的 `output/cpu/` |

- 三方库与程序分脚本，单独重跑 `build_thirdparty.sh` 只重编依赖
- 默认**全量重编**；`SKIP_CLEAN=1` 走增量
- 产物目录 `output/*/run_web.sh` 假定位于 `仓库根/output/<arch>/`，data/、models/ 从仓库根解析

### 1.3 三方库清单（全源码编译，零预编译二进制）

| 依赖 | 版本/来源 | 编译方式 |
|---|---|---|
| llama.cpp | 仓库自带 `llama_buildsrc/` | build_thirdparty.sh 编译（x86 与 aarch64 同源）|
| ONNX Runtime | v1.18.1，源码 clone（github，gh-proxy 兜底）| `common/build_onnxruntime.sh`，装到 `third_party/onnxruntime[-aarch64]/` |
| ORT 内部依赖 ×34 | deps.txt 锁定（abseil/protobuf/re2/date/tensorboard…）| ORT FetchContent 源码编译 |
| Eigen | gitlab commit `e7248b2…` 源码 zip（缓存在 `third_party/distfiles/`）| 经 `--use_preinstalled_eigen` 喂给 ORT（见坑 4.3）|
| httplib/nlohmann-json/stb | header-only | 无需编译，third_party/ 即源码 |
| 交叉工具链（编译机侧） | Ubuntu jammy 官方交叉 deb（gcc 11.4 + glibc 2.35）| 免 root 解包到 `third_party/aarch64-cross/`（非编译对象，是编译工具）|

历史 CUDA 运行库例外已随 GPU 版删除而不存在（§3.3）。

---

## 二、环境事实

- 编译机：x86_64 Ubuntu（72 核），gcc 11.4，系统 cmake 3.30；无 sudo 交互（用免 root 方案）
- 目标机：NanoPi R6S，aarch64，Ubuntu 22.04，**GCC 11.3 / binutils 2.38 / glibc 2.35 / python 3.10**，目标架构=本机（可原生编译）
- 网络：编译期代理 `PROXY=http://10.8.18.5:7890`（构建脚本默认导出 http(s)_proxy 四项，`PROXY=` 覆盖/置空）；直连 gitlab.com 被 Cloudflare 403 拦截；github 可直连（脚本仍保留 gh-proxy 兜底）；pip 已配阿里云镜像
- 版本锁定：ORT v1.18.1（`ORT_VERSION` 可换但需验证模型兼容）；Eigen commit `e7248b26a1ed53fa030c5c459f7ea095dfd276ac`

---

## 三、变更记录（按时间顺序）

### 3.1 三方库强制全量重编（2026-10-13）

背景：cmake 增量会复用旧缓存/`.o`，改源码或编译参数后产物可能与源码不一致。
现在**所有构建脚本默认 configure 前清空三方库构建目录**（llama 的 cmake 缓存与
全部旧目标文件一并删除），统一 `SKIP_CLEAN=1` 回退增量。

### 3.2 删除全部 docker 构建路径（2026-10-13）

本机 docker.io 拉不动，docker 路线是死代码。删除：`build.sh`、`build_image.sh`、
`Dockerfile`、`Dockerfile.multi`、`docker-compose.yml`、`docker-compose.gpu.yml`、
`scripts/make_devel_base.sh`；清理脚本/CMakeLists 中的 docker 引用。
未动：PLAN.md/PROGRESS.md 历史记述、llama_buildsrc/ 内上游自带文件。

### 3.3 删除 GPU 版构建路线（2026-10-13）

`build_scripts/host/` 整目录删除，项目只剩 CPU 构建线（与生产一致）。
`scripts/rest_ab_test.sh` C++ 侧改指 CPU 产物目录并显式 `VLM_DEVICE=cpu
VLM_GGUF_N_GPU_LAYERS=0`（Python 侧保持 GPU）。自此"deploy 拷贝 NVIDIA 闭源
CUDA 运行库"这一预编译例外不复存在——全项目零预编译二进制。

### 3.4 ONNX Runtime 改为源码编译（2026-10-13，核心改动）

背景：`third_party/onnxruntime/` 原是官方预编译 `.so`，是全项目最后一个预编译库。
新增 ORT 源码编译脚本（现位于 `build_scripts/common/build_onnxruntime.sh`）：

- clone v1.18.1（浅克隆+tag 校验，版本不符拒编）；构建依赖装独立 venv 不污染 conda
- `build.py --config Release --build_shared_lib --skip_tests`，CPU EP、不裁剪算子
  （保持与官方包等价的算子覆盖，yolo/bge 模型兼容）
- 产物装回 `third_party/onnxruntime/{include,lib}`，**与官方包布局一致**：
  `lib/` 拷 `build/Release/libonnxruntime.so*`；`include/` 用源码树
  `core/session/*.h` 拍平到顶层 + `cpu_provider_factory.h`/`provider_options.h`
  （工程代码 `#include <onnxruntime_cxx_api.h>` 零改动）
- 原预编译包被彻底删除（include/lib 替换、LICENSE/README 等残留清除）

**踩坑（均已修复并写入脚本）**：
1. v1.18.1 的 build.py 没有 `--install`、`--skip_docs_generation`、
   `--allow_reuse_build_dirs`（1.19+ 才有），传了直接 unrecognized arguments；
   且它从不清理 build 目录（全量重编靠脚本显式 rm）
2. 头文件 install 布局在 `include/onnxruntime/core/session/`，须拍平（见上）
3. Eigen 从 gitlab 下载：直连被 Cloudflare 403 → 走代理；且 gitlab 重新打包导致
   zip 字节与 deps.txt 记录的 SHA1 不一致 → 走 `--use_preinstalled_eigen`
   直接喂同 commit 源码（跳过 FetchContent 哈希校验，内容仍是该 commit 完整源码）
4. 编译期下载依赖走代理（§2 的 PROXY 机制）

### 3.5 构建脚本目录化重构 + 产物统一（2026-10-13）

- 旧 `build_host.sh`/`build_cpu.sh` 拆分迁移为 `build_scripts/{cpu,aarch64}/` 下
  "三方库编译"与"程序编译"两个独立脚本；运行脚本源移入 `build_scripts/*/run_web.sh`
- **产物统一**：`deploy-cpu/` → `output/cpu/`；交叉产物 → `output/aarch64/`。
  `run_web.sh` 的 data/models 锚定改为上两级（仓库根），进程匹配改为
  `web_vllm_cpp/output/<arch>/web_vllm_cpp`
- 旧名对照：`build_cpu.sh`→`build_scripts/cpu/build_*`；`build_host.sh`→已删（GPU 移除）；
  `run_web_cpu.sh`→`build_scripts/cpu/run_web.sh`；`scripts/build_onnxruntime.sh`→
  `build_scripts/common/build_onnxruntime.sh`；`deploy-cpu/`→`output/cpu/`
- `CMakeLists.txt`、`scripts/rest_ab_test.sh`、离线脚本等所有引用同步更新

### 3.6 aarch64 离线物料方案——目标机原生编译路线（2026-10-13）

NanoPi 无外网时，在有网机器 `fetch_offline.sh` 下载全部"源码级"物料（约 300MB，
清单 `third_party/offline/MANIFEST.txt`），rsync 整个仓库过去后目标机原生编译：

| 物料 | 内容 |
|---|---|
| `offline/deps/`（34 zip，254M） | ORT deps.txt 全部源码包，**34/34 SHA1 校验一致** |
| `offline/wheels-aarch64/`（12 个） | ORT 构建 Python 依赖 cp310/aarch64 wheel + **cmake 3.31 + ninja** |
| `distfiles/eigen-*.zip` | Eigen 指定 commit 源码 |
| ORT/llama 源码 | 随仓库 |

流程：`fetch_offline.sh`（有网机）→ rsync → 目标机 `apply_offline.sh`（改写
deps.txt 为本地绝对路径 + 写 `.applied` 标记）→ 正常跑 `cpu/` 两脚本，全程免联网。
适配点：Ubuntu 22.04 apt cmake 3.22 < ORT 要求的 3.26 → 自动用 venv 内 pip cmake 3.31
（`--cmake_path`）；**不能装 cmake 4.x**（会硬拒 ORT 老依赖的 `cmake_minimum<3.5`）；
venv pip 自动 `--no-index --find-links`。

### 3.7 aarch64 交叉编译路线——x86 主机直编（2026-10-14，已完成）

免 sudo：`setup_cross_toolchain.sh` 从 jammy 官方源下载交叉编译 deb 包
（gcc-11-aarch64 / binutils / libc6-dev+libc6+libgcc-s1+libgomp1-arm64-cross，
**glibc 2.35 与目标机一致**）`dpkg-deb -x` 解包到 `third_party/aarch64-cross/`。

要点与坑（全部固化在脚本内）：
1. gcc 需 `-B <gcc-cross>/11` + `--sysroot=<root>`；binutils 需
   `LD_LIBRARY_PATH=<root>/usr/lib/x86_64-linux-gnu`（libbfd/libopcodes-*-arm64）→
   统一封装在 `third_party/aarch64-cross/wrap/`
2. 链接期缺 `libc.so.6`/`ld-linux`/`libgcc_s.so.1` → 需 libc6-arm64-cross、
   libgcc-s1-arm64-cross；并在 `<root>/lib/` 建符号链（qemu 动态加载器自检用）
3. **libgomp 坑**：CMake FindOpenMP 把 OpenMP 解析成 gcc-cross 内部静态
   `libgomp.a`，静态库链进 .so 必炸（R_AARCH64_ADR_PREL_PG_HI21）→
   删/停用 gcc-cross 的 libgomp.a+spec，llama 显式 `-DGGML_OPENMP=OFF`
   （ggml 自带线程池，功能不减）
4. ORT 交叉：`ORT_ARCH_TAG=aarch64 + ORT_CROSS_TOOLCHAIN + qemu-emulator.sh`
   （try_run 探测）；protoc 用 deps.txt 的 protoc_linux_x64 预编译 codegen 工具
   （ORT 官方交叉方案，属编译工具而非目标三方库）；产物装
   `third_party/onnxruntime-aarch64/`；依赖下载复用 3.6 的本地物料，全程免外网

### 3.8 验证记录

- **2026-10-13 x86**：ORT 源码编译成功（libonnxruntime.so.1.18.1，23MB，ldd 零
  缺失）；`build_cpu.sh`（现 build_app.sh）端到端通过；三方库来源逐项核查全源码
- **2026-10-13 离线物料**：deps 34/34 SHA1 一致；apply_offline 本机演练
  34 项本地化 0 哈希异常；aarch64 wheel 完整（含 cmake 3.31 替换掉误拉的 4.x）
- **2026-10-14 aarch64 交叉**：ORT aarch64 21MB 编译成功；`output/aarch64/` 全
  产物 ELF=ARM aarch64 PIE；主程序与 libonnxruntime/libggml-cpu 最高 GLIBC 符号
  **2.34 ≤ 目标机 2.35**；qemu ld.so --list 依赖自检零缺失

### 3.9 模型耗时基准 bench_models（2026-10-14）

新增 src/bench_models.cpp：测 YOLO/BGE/Qwen-VL 三模型在本平台的加载+推理耗时，
部署目录跑 `run_bench.sh`（新脚本，env 同 run_web.sh）。2026-10-14 aarch64 交叉编译通过，
产物已入 output/aarch64/；x86 版待下次 build_app.sh 生成。

---

## 四、被删除/迁移文件总表

| 旧路径 | 处置 |
|---|---|
| `build.sh` / `build_image.sh` / `Dockerfile` / `Dockerfile.multi` / `docker-compose*.yml` / `scripts/make_devel_base.sh` | 删除（docker 路线废弃）|
| `build_host.sh`（含 GPU 路线） | 删除（GPU 版整体移除）|
| `build_cpu.sh` | 拆分为 `build_scripts/cpu/build_thirdparty.sh` + `build_app.sh` |
| `run_web.sh` / `run_web_cpu.sh`（仓库根） | 迁移为 `build_scripts/{cpu,aarch64}/run_web.sh`（deploy 内拷贝机制不变）|
| `scripts/build_onnxruntime.sh` | 迁移为 `build_scripts/common/build_onnxruntime.sh` |
| `third_party/onnxruntime/` 官方预编译包 | 删除，由源码编译产物原位替换 |
| `deploy-cpu/` | 重命名为 `output/cpu/` |

## 五、一句话速查

```bash
# x86 CPU 编译
bash build_scripts/cpu/build_thirdparty.sh && bash build_scripts/cpu/build_app.sh
# NanoPi 交叉编译（x86 上）
bash build_scripts/aarch64/setup_cross_toolchain.sh          # 仅首次
bash build_scripts/aarch64/build_thirdparty.sh && bash build_scripts/aarch64/build_app.sh
# NanoPi 原生编译（无外网目标机）
bash build_scripts/offline/fetch_offline.sh   # 有网机执行
# → rsync 仓库 → 目标机: bash build_scripts/offline/apply_offline.sh && cpu/ 两脚本
# 增量模式：命令前加 SKIP_CLEAN=1；代理：PROXY=xxx / PROXY=（不用）前缀
```
