cmake_minimum_required(VERSION 3.18)
project(web_text2img_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 依赖路径由 build_scripts/build_aarch64.sh 传入（复用 web_vllm 系列的 rknn 头/库与 third_party）
set(RKNN_INC "" CACHE PATH "")
set(RKNN_LIB_DIR "" CACHE PATH "")
set(THIRD_PARTY_DIR "" CACHE PATH "")

find_package(Threads REQUIRED)
# stb 实现内联在 frame.cpp（与 web_vllm_* 同法），无独立源文件

add_executable(web_text2img
    src/main.cpp
    src/config.cpp
    src/util.cpp
    src/frame.cpp
    src/image_embedder.cpp
    src/text_embedder.cpp
    src/gallery.cpp
    src/search_service.cpp
    src/server.cpp
)

target_include_directories(web_text2img PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}
    ${RKNN_INC}
)
target_link_directories(web_text2img PRIVATE ${RKNN_LIB_DIR})
target_link_libraries(web_text2img PRIVATE rknnrt Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(web_text2img PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib" INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")

# 板端 golden 对拍：走生产同款 TextEmbedder/ImageEmbedder 代码路径
add_executable(golden_check
    ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/golden/golden_check.cpp
    src/image_embedder.cpp
    src/text_embedder.cpp
    src/util.cpp
    src/frame.cpp
)
target_include_directories(golden_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${THIRD_PARTY_DIR} ${RKNN_INC})
target_link_directories(golden_check PRIVATE ${RKNN_LIB_DIR})
target_link_libraries(golden_check PRIVATE rknnrt Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(golden_check PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib" INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")

# 排障：单图 embedding / resize 像素导出
add_executable(embed_dump
    ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/golden/embed_dump.cpp
    src/image_embedder.cpp
    src/util.cpp
    src/frame.cpp
)
target_include_directories(embed_dump PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${THIRD_PARTY_DIR} ${RKNN_INC})
target_link_directories(embed_dump PRIVATE ${RKNN_LIB_DIR})
target_link_libraries(embed_dump PRIVATE rknnrt Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(embed_dump PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib" INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
