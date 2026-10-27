#!/bin/bash
# 【在目标离线机上运行一次】把 fetch_offline.sh 下载的物料接进构建链：
#   1. 改写 onnxruntime-src/cmake/deps.txt：把本地已有 zip 的条目 URL 改成本地绝对路径
#      （SHA1 校验照常执行——文件与远程字节一致，校验必过）；原文件备份为 deps.txt.orig
#   2. 写 third_party/offline/.applied 标记 → build_onnxruntime.sh 自动切换：
#      venv 用 --no-index 本地 wheel 安装、优先用 venv 里的 cmake(>=3.26)
#   3. 自检：ORT 源码 tag、Eigen distfile、当前架构 wheel 目录
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$HERE/third_party/onnxruntime-src"
OUT="$HERE/third_party/offline"

[ -d "$OUT/deps" ] || { echo "缺 $OUT/deps —— 先在有网机器跑 build_scripts/offline/fetch_offline.sh 并把 third_party/ 拷过来"; exit 1; }
got=$(git -C "$SRC" describe --tags 2>/dev/null || echo "")
[ "$got" = "v1.18.1" ] || { echo "ORT 源码不在（应为 v1.18.1，当前 '$got'）"; exit 1; }
ls "$HERE"/third_party/distfiles/eigen-*.zip >/dev/null || { echo "缺 eigen distfile"; exit 1; }

echo "==> 改写 deps.txt 指向本地文件"
DEPS_TXT="$SRC/cmake/deps.txt" OUT_DIR="$OUT/deps" python3 - <<'PY'
import os, hashlib
deps = os.environ["DEPS_TXT"]; out = os.environ["OUT_DIR"]
orig = deps + ".orig"
if not os.path.exists(orig):
    os.replace(deps, orig)
src = open(orig).read().splitlines()
patched, missing, badhash = 0, [], 0
with open(deps, "w") as f:
    for line in src:
        if line.startswith("#") or not line.strip():
            f.write(line + "\n"); continue
        name, url, sha = (line.split(";") + ["", ""])[:3]
        local = os.path.join(out, name + ".zip")
        if os.path.isfile(local):
            h = hashlib.sha1(open(local, "rb").read()).hexdigest()
            if h != sha:
                print(f"  !! {name}: 本地文件 SHA1 不匹配 ({h} != {sha})，保留远程 URL")
                badhash += 1
            else:
                f.write(f"{name};{local};{sha}\n"); patched += 1
        else:
            # eigen 走 preinstalled 路线；其余是 windows/32bit/macos 专用，
            # aarch64 linux 构建不会 FetchContent，无需本地文件
            unused = {"eigen", "protoc_win64", "protoc_win32", "protoc_linux_x86",
                      "protoc_mac_universal", "directx_headers"}
            if url.startswith("http") and name not in unused:
                missing.append(name)   # aarch64 linux 会用到却没下载 → 报警
            f.write(line + "\n")
print(f"  本地化 {patched} 项，哈希异常 {badhash} 项")
if missing:
    print(f"  ⚠ 以下依赖将被 aarch64 构建用到但没有本地文件（构建时会尝试联网）: {missing}")
PY

touch "$OUT/.applied"
echo "==> 自检"
echo "    架构: $(uname -m)  wheel 目录: $(ls -d "$OUT"/wheels-* 2>/dev/null | tr '\n' ' ')"
[ -d "$OUT/wheels-$(uname -m)" ] || echo "    ⚠ 没有本架构（$(uname -m)）的 wheel 目录，venv 步骤将尝试联网装 pip 依赖"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)'; then
    echo "    ERROR: 需要 python>=3.10，当前 $(python3 --version 2>&1)"; exit 1
fi
echo "    python OK: $(python3 --version)"
command -v g++ >/dev/null && echo "    g++ OK: $(g++ --version | head -1)"
echo
echo "==> 完成。开始编译："
echo "    bash build_scripts/cpu/build_thirdparty.sh && bash build_scripts/cpu/build_app.sh"
