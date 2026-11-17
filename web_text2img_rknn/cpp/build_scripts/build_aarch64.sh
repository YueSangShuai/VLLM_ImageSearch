#!/bin/bash
# 【aarch64 交叉编译】web_text2img_rknn（RK3588 NPU 版）：
# 复用 web_vllm_rknn 的 rknn-llm 头/库 + web_vllm_cpp 的交叉工具链/third_party。
# 依赖：无 ORT / 无 sentencepiece（双塔全在 NPU，分词手写复刻）。
# 产物 output/：可整体 scp 到板上 ~/web_text2img_rknn/。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"             # cpp/
ROOT="$(cd "$HERE/.." && pwd)"                        # web_text2img_rknn/
FG="$(cd "$ROOT/.." && pwd)"                          # FG-CLIP/
VLLM="$FG/web_vllm_cpp"
RKNN_REPO="$FG/web_vllm_rknn"
RKNN_INC="$RKNN_REPO/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/include"
RKNN_LIB="$RKNN_REPO/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64"
TCFILE="$VLLM/build_scripts/aarch64/aarch64-linux-gnu.toolchain.cmake"
BUILD="$HERE/build/aarch64"
DEPLOY="$ROOT/output"

for p in "$TCFILE" "$RKNN_LIB/librknnrt.so"; do
    [ -e "$p" ] || { echo "缺依赖：$p"; exit 1; }
done

if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$BUILD"
    rm -rf "$BUILD"
fi

echo "==> 1/2 交叉编译 web_text2img + golden_check（aarch64）"
cmake -S "$HERE" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TCFILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRKNN_INC="$RKNN_INC" \
    -DRKNN_LIB_DIR="$RKNN_LIB" \
    -DTHIRD_PARTY_DIR="$VLLM/third_party"
cmake --build "$BUILD" -j"$(nproc)"

echo "==> 2/2 组装 output/"
mkdir -p "$DEPLOY/lib" "$DEPLOY/static" "$DEPLOY/models" "$DEPLOY/data/gallery" "$DEPLOY/gallery"
cp -f "$BUILD/web_text2img" "$DEPLOY/web_text2img"
cp -f "$BUILD/golden_check" "$DEPLOY/golden_check"
cp -a -f "$RKNN_LIB/librknnrt.so"* "$DEPLOY/lib/"
cp -a "$HERE"/static/*.html "$DEPLOY/static/"
cp -a "$ROOT"/rknn/mars_image_fp16.rknn "$ROOT"/rknn/mars_text_fp16.rknn "$DEPLOY/models/"
cp -a "$ROOT"/models/vocab.txt "$DEPLOY/models/"
# golden 对拍数据（板端 golden_check 用）
if [ -d "$ROOT"/scripts/golden/raw ]; then
    mkdir -p "$DEPLOY/scripts/golden"
    cp -a "$ROOT"/scripts/golden/raw "$DEPLOY/scripts/golden/"
    cp -f "$ROOT"/output/run_golden_rk.sh "$DEPLOY/" 2>/dev/null || true
fi
# 运行脚本：优先保留板上/本地已有的定制版本
[ -e "$DEPLOY"/run_web_rk.sh ] || cp "$HERE"/run_web_rk.sh.template "$DEPLOY"/run_web_rk.sh
chmod +x "$DEPLOY"/run_*.sh "$DEPLOY"/web_text2img "$DEPLOY"/golden_check 2>/dev/null || true

file "$DEPLOY/web_text2img" | grep -q "ARM aarch64" || { echo "产物架构不对！"; exit 1; }
echo "==> 完成：$DEPLOY"
