cmake_minimum_required(VERSION 3.18)
project(web_vllm_rknn_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 依赖路径（由 build_scripts/ 传入；默认指向仓库内 rknn-llm 与 web_vllm_cpp third_party） ----
# RKNN_INC / RKNN_LIB_DIR  : rknn_api.h 与 librknnrt.so
# RKLLM_INC / RKLLM_LIB_DIR: rkllm.h 与 librkllmrt.so
# ORT_DIR                  : ONNX Runtime 根目录（include/ + lib/，aarch64）
# THIRD_PARTY_DIR          : stb/httplib/json 头（复用 web_vllm_cpp/third_party）
set(RKNN_INC "" CACHE PATH "")
set(RKNN_LIB_DIR "" CACHE PATH "")
set(RKLLM_INC "" CACHE PATH "")
set(RKLLM_LIB_DIR "" CACHE PATH "")
set(ORT_DIR "" CACHE PATH "")
set(THIRD_PARTY_DIR "" CACHE PATH "")
# YOLO 后端: onnx | rknn（BGE 恒为 ORT）
set(WEBVLLM_BACKEND "rknn" CACHE STRING "")

# 三模型耗时基准（与在线服务同一套 env）
add_executable(bench_models
    src/bench_models.cpp
    src/config.cpp
    src/util.cpp
    src/frame.cpp
    src/detector.cpp
    src/text_embedder.cpp
    src/rknn_image_enc.cc
    src/vlm_describer.cpp
    src/attr_head.cpp
)

add_executable(web_vllm_rknn
    src/main.cpp
    src/config.cpp
    src/util.cpp
    src/json_store.cpp
    src/frame.cpp
    src/detector.cpp
    src/text_embedder.cpp
    src/rknn_image_enc.cc
    src/vlm_describer.cpp
    src/attr_head.cpp
    src/matcher.cpp
    src/stream_processor.cpp
    src/monitor.cpp
    src/server.cpp
)

# 检测框打印（golden 对齐用）
add_executable(yolo_probe
    src/yolo_probe.cpp
    src/frame.cpp
    src/detector.cpp
)

find_package(Threads REQUIRED)

function(configure_app_target tgt)
if(WEBVLLM_BACKEND STREQUAL "rknn")
    target_compile_definitions(${tgt} PRIVATE WEBVLLM_RKNN)
endif()
target_include_directories(${tgt} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}
    ${RKNN_INC}
    ${RKLLM_INC}
    ${ORT_DIR}/include
)
target_link_directories(${tgt} PRIVATE
    ${RKNN_LIB_DIR}
    ${RKLLM_LIB_DIR}
    ${ORT_DIR}/lib
)
target_link_libraries(${tgt} PRIVATE
    rknnrt
    rkllmrt
    onnxruntime
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
# 运行时优先从二进制同级的 lib/ 目录加载 .so
set_target_properties(${tgt} PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
)
endfunction()

configure_app_target(web_vllm_rknn)
configure_app_target(bench_models)
configure_app_target(yolo_probe)
