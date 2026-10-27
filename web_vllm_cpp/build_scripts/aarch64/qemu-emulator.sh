#!/bin/bash
# CMAKE_CROSSCOMPILING_EMULATOR：用 qemu-user 运行刚交叉编出的 aarch64 测试程序
# （ORT 依赖里 abseil 等用 try_run 探测特性时需要）
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../third_party/aarch64-cross/root" && pwd)"
exec /usr/bin/qemu-aarch64-static -L "$ROOT" "$@"
