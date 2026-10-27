cmake_minimum_required(VERSION 3.18)
project(web_vllm_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 依赖路径（由 build_scripts/cpu/build_app.sh 传入） ----
# LLAMA_LIB_DIR  : libllama/libmtmd/libggml* 所在目录
# LLAMA_INC_DIRS : 分号分隔的头文件目录（llama.h / ggml / mtmd）
# ORT_DIR        : ONNX Runtime 根目录（include/ + lib/）
set(LLAMA_LIB_DIR "" CACHE PATH "")
set(LLAMA_INC_DIRS "" CACHE STRING "")
set(ORT_DIR "" CACHE PATH "")

add_executable(web_vllm_cpp
    src/main.cpp
    src/config.cpp
    src/util.cpp
    src/json_store.cpp
    src/frame.cpp
    src/detector.cpp
    src/text_embedder.cpp
    src/vlm_describer.cpp
    src/matcher.cpp
    src/stream_processor.cpp
    src/monitor.cpp
    src/server.cpp
)

# 独立基准测试程序：测量三个模型（YOLO / BGE / Qwen-VL）在本平台的加载与推理耗时
add_executable(bench_models
    src/bench_models.cpp
    src/config.cpp
    src/util.cpp
    src/frame.cpp
    src/detector.cpp
    src/text_embedder.cpp
    src/vlm_describer.cpp
)

find_package(Threads REQUIRED)

# 两个可执行目标共用同一套头文件/链接配置
# llama-common：chat template / sampler / 模型加载（vlm_describer 直接复用 mtmd-cli 同款实现）
function(configure_app_target tgt)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${LLAMA_INC_DIRS}
        ${ORT_DIR}/include
    )
    target_link_directories(${tgt} PRIVATE
        ${LLAMA_LIB_DIR}
        ${ORT_DIR}/lib
    )
    target_link_libraries(${tgt} PRIVATE
        mtmd llama-common llama ggml ggml-base
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

configure_app_target(web_vllm_cpp)
configure_app_target(bench_models)
