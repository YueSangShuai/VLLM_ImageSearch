#pragma once
// Qwen3-VL 视觉塔 RKNN 封装（移植自 rknn-llm v1.3.1 multimodal_model_demo/deploy/src/image_enc.cc）。
// 输出布局：逐 image token 交错拼接 n_output 段 embedding（deepstack 融合由 rkllm 端完成）。
#include "rknn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    int model_image_token;
    int model_embed_size;
} rknn_app_context_t;

int init_imgenc(const char* model_path, rknn_app_context_t* app_ctx, const int core_num);
int release_imgenc(rknn_app_context_t* app_ctx);
// img_data: NHWC uint8 RGB（model_width*model_height*3）；out_result: float32 缓冲，
// 大小 = model_image_token * io_num.n_output * model_embed_size
int run_imgenc(rknn_app_context_t* app_ctx, void* img_data, float* out_result);

#ifdef __cplusplus
}
#endif
