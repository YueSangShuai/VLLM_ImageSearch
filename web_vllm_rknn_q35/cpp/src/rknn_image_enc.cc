#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_image_enc.h"

int init_imgenc(const char* model_path, rknn_app_context_t* app_ctx, const int core_num)
{
    int ret;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init fail! ret=%d path=%s\n", ret, model_path);
        return -1;
    }

    if (core_num == 2) {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
    } else if (core_num == 3) {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    } else {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);
    }
    if (ret < 0) {
        fprintf(stderr, "rknn_set_core_mask fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query IN_OUT_NUM fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "rknn_query INPUT_ATTR fail! ret=%d\n", ret);
            rknn_destroy(ctx);
            return -1;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            fprintf(stderr, "rknn_query OUTPUT_ATTR fail! ret=%d\n", ret);
            rknn_destroy(ctx);
            return -1;
        }
    }

    // image_token / embed_size：取 output0 dims 中第一个 >1 的维度对（[1,196,2048] 型布局）
    app_ctx->model_image_token = 0;
    app_ctx->model_embed_size = 0;
    for (int i = 0; i < 4; i++) {
        if (output_attrs[0].dims[i] > 1) {
            app_ctx->model_image_token = output_attrs[0].dims[i];
            app_ctx->model_embed_size = output_attrs[0].dims[i + 1];
            break;
        }
    }
    if (app_ctx->model_image_token <= 0 || app_ctx->model_embed_size <= 0) {
        fprintf(stderr, "unexpected vision output dims\n");
        rknn_destroy(ctx);
        return -1;
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }

    return 0;
}

int release_imgenc(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int run_imgenc(rknn_app_context_t* app_ctx, void* img_data, float* out_result)
{
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[app_ctx->io_num.n_output];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf   = img_data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        fprintf(stderr, "rknn_inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    for (int j = 0; j < app_ctx->io_num.n_output; j++) {
        outputs[j].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    if (app_ctx->io_num.n_output == 1) {
        memcpy(out_result, outputs[0].buf, outputs[0].size);
    } else {
        // 逐 token 交错拼接 n_output 段（deepstack + hidden），与 rkllm 期望布局一致
        for (int i = 0; i < app_ctx->model_image_token; i++) {
            for (int j = 0; j < app_ctx->io_num.n_output; j++) {
                memcpy(out_result + i * app_ctx->io_num.n_output * app_ctx->model_embed_size
                           + j * app_ctx->model_embed_size,
                       (float*)(outputs[j].buf) + i * app_ctx->model_embed_size,
                       sizeof(float) * app_ctx->model_embed_size);
            }
        }
    }

    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    return ret;
}
