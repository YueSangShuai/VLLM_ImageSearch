cmake_minimum_required(VERSION 3.18)
project(web_siglip_reid_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 依赖路径（由 build_scripts/ 传入；默认指向仓库内 rknn-llm 与 web_vllm_cpp third_party） ----
# RKNN_INC / RKNN_LIB_DIR : rknn_api.h 与 librknnrt.so
# ORT_DIR                 : ONNX Runtime 根目录（include/ + lib/，aarch64）
# THIRD_PARTY_DIR         : stb/httplib/json 头（复用 web_vllm_cpp/third_party）
# SP_DIR                  : sentencepiece 根目录（src/ 头 + build-aarch64/lib/*.a 静态库）
set(RKNN_INC "" CACHE PATH "")
set(RKNN_LIB_DIR "" CACHE PATH "")
set(ORT_DIR "" CACHE PATH "")
set(THIRD_PARTY_DIR "" CACHE PATH "")
set(SP_DIR "" CACHE PATH "")

file(GLOB_RECURSE SP_LIBS "${SP_DIR}/build-aarch64/*.a")
if(NOT SP_LIBS)
    message(FATAL_ERROR "缺 sentencepiece 静态库，先跑 build_scripts/build_sentencepiece_aarch64.sh（查 ${SP_DIR}/build-aarch64/lib/）")
endif()

# 公共源文件（两目标共用）
set(COMMON_SRC
    src/config.cpp
    src/util.cpp
    src/frame.cpp
    src/detector.cpp
    src/siglip_vision.cpp
    src/text_embedder.cpp
)

# 三模型耗时基准（与在线服务同一套 env）
add_executable(bench_models
    src/bench_models.cpp
    ${COMMON_SRC}
)

add_executable(web_siglip_reid
    src/main.cpp
    src/json_store.cpp
    src/matcher.cpp
    src/stream_processor.cpp
    src/monitor.cpp
    src/server.cpp
    ${COMMON_SRC}
)

find_package(Threads REQUIRED)

function(configure_app_target tgt)
target_include_directories(${tgt} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}
    ${RKNN_INC}
    ${ORT_DIR}/include
    ${SP_DIR}/src
)
target_link_directories(${tgt} PRIVATE
    ${RKNN_LIB_DIR}
    ${ORT_DIR}/lib
)
target_link_libraries(${tgt} PRIVATE
    rknnrt
    onnxruntime
    ${SP_LIBS}          # 自含（abseil 已捆绑），顺序：sentencepiece 在前
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
# 运行时优先从二进制同级的 lib/ 目录加载 .so
set_target_properties(${tgt} PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
)
endfunction()

configure_app_target(web_siglip_reid)
configure_app_target(bench_models)

# 板端 golden 对拍（scripts/golden/）：只用视觉塔+帧工具，不链 ORT/sentencepiece
add_executable(rknn_check
    ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/golden/rknn_check.cpp
    src/siglip_vision.cpp
    src/frame.cpp
    src/util.cpp
)
target_include_directories(rknn_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${THIRD_PARTY_DIR} ${RKNN_INC})
target_link_directories(rknn_check PRIVATE ${RKNN_LIB_DIR})
target_link_libraries(rknn_check PRIVATE rknnrt Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(rknn_check PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib" INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
