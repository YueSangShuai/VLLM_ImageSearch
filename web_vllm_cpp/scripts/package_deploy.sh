#!/bin/bash
# 把 deploy/ 补全为可跨机运行的自包含包，并打 tar.gz。
# 目标机要求：Ubuntu 22.04 同档 glibc(≥2.35)、NVIDIA 驱动≥535、GPU sm_80、ffmpeg。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$HERE/deploy"
LIB="$DEPLOY/lib"
CUDA=${CUDA_HOME:-/usr/local/cuda-12.2}
[ -d "$DEPLOY" ] || { echo "先跑 build_host.sh"; exit 1; }

echo "==> 1/4 补 CUDA 运行库（libcublas/libcublasLt/libnvrtc 等，不依赖本机 ldconfig 命中情况）"
for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libnvrtc.so.12; do
    src=$(ls "$CUDA"/lib64/${lib}* 2>/dev/null | grep -v "_static" | head -1 || true)
    [ -n "$src" ] || src=$( { ldconfig -p || true; } | awk -v l="$lib" '$1==l{print $NF; exit}')
    [ -n "$src" ] && cp -a -f "$src" "$LIB/" && echo "    + $lib"
done
# 补齐断链的 soname 符号链（cp 单文件后常缺）
cd "$LIB"
for f in libcublas.so.12.* libcublasLt.so.12.* libnvrtc.so.12.* libcudart.so.12.*; do
    [ -e "$f" ] || continue
    base=${f%%.*}.so.${f#*.so.}; base=${base%%.*}
    [ -e "$base" ] || ln -sf "$f" "$base"
done
cd "$HERE"

echo "==> 2/4 带 libstdc++/libgcc_s（目标机 gcc<11 时避免 GLIBCXX 缺失；rpath \$ORIGIN/../lib 优先命中自带版）"
for f in libstdc++.so.6 libgcc_s.so.1; do
    real=$( { ldconfig -p || true; } | awk -v l="$f" '$1==l{print $NF; exit}')
    [ -n "$real" ] && cp -a -f "$(realpath "$real")" "$LIB/$f" && echo "    + $f"
done

echo "==> 3/4 全依赖闭包自检（目标机视角：只用 deploy/lib，屏蔽系统 CUDA/llama）"
missing=0
for so in "$LIB"/libggml-cuda.so.0 "$LIB"/libmtmd.so.0 "$DEPLOY/web_vllm_cpp"; do
    LD_LIBRARY_PATH="$LIB" ldd "$so" 2>&1 | awk '/not found/{print "    缺: " $1; f=1} END{exit f}' || missing=1
done
# libcuda.so.1 属驱动，允许 not found（运行时由目标机驱动提供）
LD_LIBRARY_PATH="$LIB" ldd "$DEPLOY/web_vllm_cpp" 2>&1 | grep "not found" | grep -v libcuda || true
[ $missing -eq 0 ] && echo "    闭包完整（libcuda 由目标机驱动提供）"

echo "==> 4/4 打包"
TAR="$HERE/web_vllm_cpp_deploy_$(date +%Y%m%d).tar.gz"
tar -czf "$TAR" -C "$HERE" deploy
echo "==> $TAR ($(du -h "$TAR" | cut -f1))"
cat <<'EOF'

目标机部署步骤：
  tar xzf web_vllm_cpp_deploy_*.tar.gz && cd deploy
  # 模型文件不在包内（~4GB），拷贝或改环境变量指路径：
  #   VLM_MODEL_NAME / VLM_MMPROJ_PATH（GGUF）、EMBEDDING_ONNX_PATH、
  #   EMBEDDING_VOCAB_PATH、DETECTOR_ONNX_PATH
  # 数据目录：WEB_VLLM_DATA_DIR（默认 <deploy 上级>/data，即本仓库自带 data/）
  # 确认 ffmpeg 已安装；勿与 Python 版同时运行（单实例）
  ./run_web.sh
前置：NVIDIA 驱动 ≥535、GPU 为 sm_80（A800/A100）、glibc ≥2.35（Ubuntu 22.04）。
其他 GPU 架构/更老发行版：在编译机改 CMAKE_CUDA_ARCHITECTURES 后重编再打包。
EOF
