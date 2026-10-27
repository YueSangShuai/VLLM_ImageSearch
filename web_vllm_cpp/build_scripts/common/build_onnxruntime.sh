#!/bin/bash
# 从源码编译 ONNX Runtime（CPU EP），替代 third_party/onnxruntime/ 里的官方预编译 .so。
#
# - 版本锁定 1.18.1：与工程 C++ 代码所用 C API 及原预编译包一致，勿随意升级
# - 布局不变：产物仍是 third_party/onnxruntime/{include,lib}，上层构建无感切换
# - 由 build_scripts/{cpu,host}/build_thirdparty.sh 调用，也可单独运行
# - 默认全量重编（清空 ORT build 目录）；SKIP_CLEAN=1 复用增量（cmake --build 语义）
# - 首次需从 github clone（约 1GB，走 gh-proxy 镜像重试）；已存在则直接用
# - 编译依赖装在独立 venv（third_party/onnxruntime-src/.venv），不污染 conda
# - 全量编译 72 核约 15~30 分钟
# - 三方依赖全部源码编译（ORT 自带 FetchContent），仅 Eigen 需特殊处理，见第 2.5 步
#
# 网络：本机直连 gitlab.com 被 Cloudflare 拦（403），依赖走 HTTP 代理；
#       换机器用 PROXY= 覆盖，PROXY="" 则完全不用代理。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # 仓库根（本脚本在 build_scripts/common/ 下）
ORT_VERSION="${ORT_VERSION:-1.18.1}"
SRC="$HERE/third_party/onnxruntime-src"
# 交叉编译：ORT_ARCH_TAG=aarch64 时构建目录/安装目录加后缀，产物互不覆盖。
# 需同时提供 ORT_CROSS_TOOLCHAIN(cmake 工具链文件) 和 ORT_CROSS_EMULATOR(可执行 qemu 包装，可空)
ORT_ARCH_TAG="${ORT_ARCH_TAG:-}"
SFX="${ORT_ARCH_TAG:+-$ORT_ARCH_TAG}"
INSTALL="$HERE/third_party/onnxruntime$SFX"
VENV="$SRC/.venv"
PYTHON="${ORT_PYTHON:-/home/yuesang/miniconda3/bin/python3}"   # 需 >=3.10（ORT 1.18 硬性要求）
PROXY="${PROXY-http://10.8.18.5:7890}"
[ -n "$PROXY" ] && export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"

# Eigen：ORT deps.txt 锁定的 commit（deps.txt 有 URL+SHA1）
EIGEN_COMMIT="e7248b26a1ed53fa030c5c459f7ea095dfd276ac"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), sys.version' \
    || { echo "需要 Python>=3.10，当前 $($PYTHON --version 2>&1)（可用 ORT_PYTHON 指定）"; exit 1; }
command -v cmake >/dev/null || { echo "需要 cmake"; exit 1; }

# ---------- 1. 取源码（锁 tag） ----------
if [ ! -d "$SRC/.git" ]; then
    echo "==> clone onnxruntime v$ORT_VERSION（github 不通自动走 gh-proxy 镜像）"
    clone_ok=0
    for url in "https://github.com/microsoft/onnxruntime.git" \
               "https://gh-proxy.com/https://github.com/microsoft/onnxruntime.git"; do
        if git clone --depth 1 --branch "v$ORT_VERSION" "$url" "$SRC"; then clone_ok=1; break; fi
        rm -rf "$SRC"
    done
    [ "$clone_ok" = "1" ] || { echo "clone 失败（github/gh-proxy 均不通）"; exit 1; }
fi
got=$(git -C "$SRC" describe --tags 2>/dev/null || echo "")
[ "$got" = "v$ORT_VERSION" ] || { echo "源码版本是 '$got'，要求 v$ORT_VERSION（third_party 头文件按此版本编译）"; exit 1; }

# ---------- 2. 编译依赖装进独立 venv ----------
# 离线模式（目标机跑过 apply_offline.sh，且存在本机架构的 wheels-<arch>/）：
# pip --no-index 从本地 wheel 装，包含 cmake/ninja（ORT 1.18 需 CMake>=3.26，
# Ubuntu 22.04 apt 只有 3.22；venv 里的 cmake 会在第 3 步通过 --cmake_path 优先使用）
OFFLINE_DIR="$HERE/third_party/offline"
WHEELS="$OFFLINE_DIR/wheels-$(uname -m)"
if [ -e "$OFFLINE_DIR/.applied" ] && [ -d "$WHEELS" ]; then
    OFFLINE=1
else
    OFFLINE=0
fi
if [ ! -e "$VENV/.done" ]; then
    echo "==> 创建 venv 并安装 ORT 构建依赖（requirements.txt；离线=$OFFLINE）"
    "$PYTHON" -m venv "$VENV"
    if [ "$OFFLINE" = "1" ]; then
        "$VENV/bin/pip" install --quiet --no-index --find-links "$WHEELS" \
            -r "$SRC/requirements.txt" wheel setuptools cmake ninja
    else
        "$VENV/bin/pip" install --quiet --upgrade pip
        "$VENV/bin/pip" install --quiet -r "$SRC/requirements.txt" wheel
    fi
    touch "$VENV/.done"
fi

# ---------- 2.5 Eigen：拿 deps.txt 锁定的 commit 源码，走 preinstalled 路线 ----------
# 为什么特殊处理：ORT deps.txt 对 gitlab 的 zip 记了 SHA1，但 gitlab 重新打包后字节变了，
# FetchContent 哈希校验必挂；且 gitlab 直连被 Cloudflare 403。这里用代理下载【同一 commit】
# 的源码 zip 解开，走 --use_preinstalled_eigen 直接喂给 cmake（preinstalled 分支跳过
# FetchContent，不校验 zip 哈希）——内容仍是该 commit 的完整源码，不是任何预编译物。
EIGEN_ZIP="$HERE/third_party/distfiles/eigen-$EIGEN_COMMIT.zip"
EIGEN_SRC="$HERE/third_party/eigen-src/eigen-$EIGEN_COMMIT"
if [ ! -d "$EIGEN_SRC" ]; then
    if [ ! -f "$EIGEN_ZIP" ]; then
        echo "==> 下载 Eigen commit $EIGEN_COMMIT（gitlab，走代理 ${PROXY:-无}）"
        curl -fL --retry 2 -o "$EIGEN_ZIP" \
            "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_COMMIT/eigen-$EIGEN_COMMIT.zip"
    fi
    unzip -q -d "$HERE/third_party/eigen-src" "$EIGEN_ZIP"
    [ -d "$EIGEN_SRC" ] || { echo "解压后未找到 $EIGEN_SRC"; exit 1; }
fi
EIGEN_FLAGS=(--use_preinstalled_eigen --eigen_path "$EIGEN_SRC")

# ---------- 3. 源码编译（CPU EP + 共享库） ----------
# 不加 --disable_* 裁剪特性集，保持与官方预编译包等价的算子覆盖（yolo/bge 模型兼容）
# 注意：1.18.1 的 build.py 没有 clean/reuse 相关参数，它总是保留 build 目录增量构建；
# 所以全量重编靠下面显式 rm，SKIP_CLEAN 分支什么都不用传。
BUILD_DIR="$SRC/build$SFX"
if [ "${SKIP_CLEAN:-0}" = "1" ]; then
    echo "==> SKIP_CLEAN=1：保留 $BUILD_DIR 增量编译"
else
    echo "==> 全量重编：清空 $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# 注意：1.18.1 的 build.py 没有 --install/--skip_docs_generation 参数（1.19+ 才有），
# 产物在 build/Release/ 下，头文件要按官方包布局手工组装（见第 4 步）
# venv 里有 pip 版 cmake（>=3.26）时优先用它，绕开系统 cmake 版本过旧的问题
CMAKE_FLAGS=()
[ -x "$VENV/bin/cmake" ] && CMAKE_FLAGS=(--cmake_path "$VENV/bin/cmake")

CROSS_FLAGS=()
if [ -n "$ORT_ARCH_TAG" ]; then
    [ -f "${ORT_CROSS_TOOLCHAIN:-}" ] || { echo "交叉模式需设置 ORT_CROSS_TOOLCHAIN=<工具链文件>"; exit 1; }
    CROSS_FLAGS=(--arm64 --cmake_extra_defines CMAKE_TOOLCHAIN_FILE="$ORT_CROSS_TOOLCHAIN")
    [ -x "${ORT_CROSS_EMULATOR:-}" ] && CROSS_FLAGS+=(--cmake_extra_defines CMAKE_CROSSCOMPILING_EMULATOR="$ORT_CROSS_EMULATOR")
fi

"$VENV/bin/python" "$SRC/tools/ci_build/build.py" \
    --build_dir "$BUILD_DIR" \
    --config Release \
    --build_shared_lib \
    --parallel "$(nproc)" \
    --skip_tests \
    "${CMAKE_FLAGS[@]}" \
    "${CROSS_FLAGS[@]}" \
    "${EIGEN_FLAGS[@]}"

# ---------- 4. 组装安装目录（include/ + lib/，与官方预编译包布局一致） ----------
STAGE="$BUILD_DIR/Release"
[ -e "$STAGE/libonnxruntime.so.$ORT_VERSION" ] \
    || { echo "编译产物缺 libonnxruntime.so.$ORT_VERSION：$STAGE"; exit 1; }
if [ -e "$INSTALL/VERSION_NUMBER" ] && [ "$(cat "$INSTALL/VERSION_NUMBER")" != "$ORT_VERSION" ]; then
    echo "原 third_party/onnxruntime 版本 $(cat "$INSTALL/VERSION_NUMBER") != $ORT_VERSION，备份到 .prebuilt.bak"
    mv "$INSTALL" "$INSTALL.prebuilt.bak"
fi
mkdir -p "$INSTALL"
rm -rf "$INSTALL/include" "$INSTALL/lib" "$INSTALL/GIT_COMMIT_ID"
mkdir -p "$INSTALL/include" "$INSTALL/lib"
# 官方包 include/ 顶层就是拍平的会话 API 头；工程代码 #include <onnxruntime_cxx_api.h> 依赖此布局
cp -a "$SRC/include/onnxruntime/core/session/"*.h "$INSTALL/include/"
cp -a "$SRC/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h" "$INSTALL/include/"
cp -a "$SRC/include/onnxruntime/core/framework/provider_options.h" "$INSTALL/include/"
# 共享库（含 .so → .so.1.18.1 符号链）
cp -a "$STAGE"/libonnxruntime.so* "$INSTALL/lib/"
echo "$ORT_VERSION" > "$INSTALL/VERSION_NUMBER"
git -C "$SRC" rev-parse HEAD > "$INSTALL/GIT_COMMIT_ID"
# 清掉官方预编译包遗留的文档文件（现在目录里只剩源码编译产物）
rm -f "$INSTALL/LICENSE" "$INSTALL/Privacy.md" "$INSTALL/README.md" "$INSTALL/ThirdPartyNotices.txt"

echo "==> 完成：$INSTALL（源码编译，CPU EP，v$ORT_VERSION）"
ls -la "$INSTALL/lib" | head
