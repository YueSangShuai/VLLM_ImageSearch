# aarch64 交叉编译工具链（NanoPi R6S / Ubuntu 22.04 / glibc 2.35 兼容）
# 编译器来自 third_party/aarch64-cross/wrap/（Ubuntu jammy 交叉 deb 免 root 解包，
# 由 build_scripts/aarch64/setup_cross_toolchain.sh 生成）
get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(XROOT "${REPO}/third_party/aarch64-cross/root")
set(XWRAP "${REPO}/third_party/aarch64-cross/wrap")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   "${XWRAP}/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${XWRAP}/aarch64-linux-gnu-g++")
set(CMAKE_SYSROOT "${XROOT}")

set(CMAKE_AR     "${XWRAP}/aarch64-linux-gnu-ar"     CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB "${XWRAP}/aarch64-linux-gnu-ranlib" CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP  "${XWRAP}/aarch64-linux-gnu-strip"  CACHE FILEPATH "" FORCE)
set(CMAKE_NM     "${XWRAP}/aarch64-linux-gnu-nm"     CACHE FILEPATH "" FORCE)
set(CMAKE_OBJCOPY "${XWRAP}/aarch64-linux-gnu-objcopy" CACHE FILEPATH "" FORCE)
set(CMAKE_OBJDUMP "${XWRAP}/aarch64-linux-gnu-objdump" CACHE FILEPATH "" FORCE)
set(CMAKE_READELF "${XWRAP}/aarch64-linux-gnu-readelf" CACHE FILEPATH "" FORCE)

set(CMAKE_FIND_ROOT_PATH "${XROOT}/usr/aarch64-linux-gnu" "${XROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
