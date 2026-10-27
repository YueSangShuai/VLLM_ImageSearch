#!/bin/bash
# 【在有网机器上运行】把 aarch64 目标机（如 NanoPi R6S, Ubuntu 22.04, GCC 11.3）
# 离线编译所需的全部"源码级物料"下载到 third_party/offline/：
#   deps/     ORT cmake/deps.txt 里全部三方依赖源码 zip（abseil/protobuf/re2…）
#   wheels/   ORT 构建用 Python 依赖的 aarch64 wheel（numpy/protobuf/sympy…）+ cmake/ninja
#   （Eigen 已缓存在 third_party/distfiles/；ORT/llama 源码本就在仓库里）
#
# 目标机上：跑 build_scripts/offline/apply_offline.sh 一次（改写 deps.txt 指向本地
# 文件、建离线 venv），之后正常跑 build_scripts/cpu/ 下的脚本即可，全程不需要网络。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$HERE/third_party/onnxruntime-src"
OUT="$HERE/third_party/offline"
PROXY="${PROXY-http://10.8.18.5:7890}"
PY="${ORT_PYTHON:-/home/yuesang/miniconda3/bin/python3}"
TARGET_PY="3.10"          # 目标机 python 版本（Ubuntu 22.04 = 3.10）

[ -d "$SRC" ] || { echo "缺 ORT 源码 $SRC —— 先在有网机器跑一次 build_scripts/common/build_onnxruntime.sh（或手动 clone v1.18.1）"; exit 1; }
mkdir -p "$OUT/deps" "$OUT/wheels-aarch64"

# ---------- 1. deps.txt 全部源码 zip ----------
# aarch64 linux 目标用不到的条目（windows/32bit/macos protoc）跳过以省空间；
# apply_offline.sh 只改写"本地已存在文件"的条目，其余保持远程 URL（不会被用到）
echo "==> 下载 deps.txt 依赖源码 zip（代理 ${PROXY:-无}）"
[ -n "$PROXY" ] && export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
while IFS=';' read -r name url sha; do
    case "$name" in \#*|"") continue;; esac
    case "$name" in protoc_win*|protoc_mac*|protoc_linux_x86|eigen) continue;; esac
    dest="$OUT/deps/$name.zip"
    if [ -s "$dest" ] && [ "$(sha1sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
        echo "    已存在且哈希匹配，跳过：$name"
        continue
    fi
    echo "    下载 $name"
    curl -fL --retry 3 -o "$dest" "$url" || { echo "!! $name 下载失败：$url"; rm -f "$dest"; continue; }
    got=$(sha1sum "$dest" | cut -d' ' -f1)
    [ "$got" = "$sha" ] || echo "    警告：$name SHA1 不匹配（got=$got expect=$sha）——目标机上会 FetchContent 校验失败，需排查"
done < "$SRC/cmake/deps.txt"

# ---------- 2. Eigen（指定 commit 源码 zip，已在 distfiles 则跳过） ----------
EIGEN_COMMIT="e7248b26a1ed53fa030c5c459f7ea095dfd276ac"
EIGEN_ZIP="$HERE/third_party/distfiles/eigen-$EIGEN_COMMIT.zip"
if [ ! -s "$EIGEN_ZIP" ]; then
    echo "==> 下载 Eigen commit zip"
    curl -fL --retry 2 -o "$EIGEN_ZIP" \
        "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_COMMIT/eigen-$EIGEN_COMMIT.zip"
fi

# ---------- 3. 目标机 Python 构建依赖（aarch64 wheel） ----------
# 包含 cmake/ninja：Ubuntu 22.04 apt cmake=3.22 偏旧，ORT 1.18 需要更新版本
echo "==> 下载 aarch64 wheel（python $TARGET_PY）"
"$PY" -m pip download --dest "$OUT/wheels-aarch64" \
    --only-binary=:all: \
    --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 \
    --python-version "$TARGET_PY" \
    -r "$SRC/requirements.txt" "cmake<4" ninja wheel setuptools 2>&1 | tail -3

# ---------- 4. 清单 ----------
{
    echo "# 离线物料清单（fetch_offline.sh 生成）"
    echo "目标机：aarch64 / Ubuntu 22.04 / python $TARGET_PY / GCC 11.3"
    echo
    du -sh "$OUT/deps" "$OUT/wheels-aarch64" "$HERE/third_party/distfiles"
    echo; echo "deps/ ($(ls "$OUT/deps" | wc -l) 个):"; ls "$OUT/deps"
    echo; echo "wheels-aarch64/ ($(ls "$OUT/wheels-aarch64" | wc -l) 个):"; ls "$OUT/wheels-aarch64"
} > "$OUT/MANIFEST.txt"

echo
echo "==> 完成。拷到目标机（保持相对路径不变）："
echo "    rsync -a --exclude build/ --exclude output/models . 目标机:~/web_vllm_cpp/"
echo "目标机执行：bash build_scripts/offline/apply_offline.sh && bash build_scripts/cpu/build_thirdparty.sh"
