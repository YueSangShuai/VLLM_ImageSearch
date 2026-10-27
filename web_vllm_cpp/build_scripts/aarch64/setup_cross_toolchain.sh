#!/bin/bash
# 免 root 安装 aarch64 交叉编译工具链到 third_party/aarch64-cross/：
#   从 Ubuntu jammy 官方源下载交叉编译 deb 包（gcc 11.4 + glibc 2.35，
#   与 NanoPi R6S Ubuntu 22.04 目标环境完全一致）并 dpkg-deb -x 解包。
# 前提：x86_64 Ubuntu 主机，apt 可用；/usr/bin/qemu-aarch64-static（仅自检用）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TC="$HERE/third_party/aarch64-cross"
R="$TC/root"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

DEBS=(
    gcc-11-aarch64-linux-gnu g++-11-aarch64-linux-gnu cpp-11-aarch64-linux-gnu
    binutils-aarch64-linux-gnu
    libgcc-11-dev-arm64-cross libstdc++-11-dev-arm64-cross
    libc6-dev-arm64-cross libc6-arm64-cross libgcc-s1-arm64-cross
    libgomp1-arm64-cross
    linux-libc-dev-arm64-cross
)

mkdir -p "$R" "$TC/wrap"
echo "==> 下载交叉编译 deb（jammy 官方源）"
(cd "$TMP" && apt-get download "${DEBS[@]}")

echo "==> 解包到 $R"
for d in "$TMP"/*.deb; do dpkg-deb -x "$d" "$R/"; done

echo "==> 生成 wrap/ 包装脚本（注入 -B/--sysroot/LD_LIBRARY_PATH）"
HOSTLIB="$R/usr/lib/x86_64-linux-gnu"
GCCVER=11
for w in gcc g++; do
    cat > "$TC/wrap/aarch64-linux-gnu-$w" <<EOF
#!/bin/bash
export LD_LIBRARY_PATH="$HOSTLIB:\${LD_LIBRARY_PATH:-}"
exec "$R/usr/bin/aarch64-linux-gnu-$w-$GCCVER" \\
     -B"$R/usr/lib/gcc-cross/aarch64-linux-gnu/$GCCVER" --sysroot="$R" "\$@"
EOF
done
for b in ar as cpp ld ld.bfd objcopy objdump readelf strip ranlib nm size strings; do
    real="$R/usr/bin/aarch64-linux-gnu-$b"
    [ -e "$real" ] || real="$R/usr/bin/aarch64-linux-gnu-$b-$GCCVER"
    [ -e "$real" ] || continue
    cat > "$TC/wrap/aarch64-linux-gnu-$b" <<EOF
#!/bin/bash
export LD_LIBRARY_PATH="$HOSTLIB:\${LD_LIBRARY_PATH:-}"
exec "$real" "\$@"
EOF
done
chmod +x "$TC"/wrap/*

# OpenMP 共享库的 dev 符号链（缺它时 gcc 会退到静态 libgomp.a，链进 .so 必炸）
[ -e "$R/usr/aarch64-linux-gnu/lib/libgomp.so.1.0.0" ] && \
    ln -sf libgomp.so.1.0.0 "$R/usr/aarch64-linux-gnu/lib/libgomp.so"

# qemu 自检需要 sysroot 里有 /lib/ld-linux...（真实设备上由 glibc 包自带）
mkdir -p "$R/lib"
for f in "$R"/usr/aarch64-linux-gnu/lib/*.so*; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    [ -e "$R/lib/$b" ] || ln -sf "../usr/aarch64-linux-gnu/lib/$b" "$R/lib/$b"
done

echo "==> 自检：编译并 qemu 运行一个 C++17+线程 测试程序"
cat > "$TMP/t.cpp" <<'CPP'
#include <iostream>
#include <thread>
extern "C" int add(int a,int b){return a+b;}
int main(){std::cout<<add(2,3)<<"\n";std::thread t([]{});t.join();return 0;}
CPP
"$TC/wrap/aarch64-linux-gnu-g++" -O2 -std=c++17 "$TMP/t.cpp" -o "$TMP/t" -static-libstdc++ -pthread
file "$TMP/t" | grep -q "ARM aarch64" || { echo "产物不是 aarch64 ELF"; exit 1; }
if command -v qemu-aarch64-static >/dev/null; then
    got="$(qemu-aarch64-static -L "$R" "$TMP/t")"
    [ "$got" = "5" ] || { echo "qemu 运行结果异常: $got"; exit 1; }
    maxg="$(readelf -V "$TMP/t" | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1)"
    echo "    运行 OK；最高 GLIBC 符号版本 $maxg（NanoPi=GLIBC_2.35，须 ≤ 之）"
else
    echo "    （无 qemu-aarch64-static，跳过运行自检）"
fi
echo "==> 完成：工具链在 $TC"
