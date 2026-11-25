// vision_bench: RKNN 视觉塔定形推理测速（gecko-InternViT / Qwen3-VL-ViT / LFM-SigLIP2）。
// 自动查询输入：pixel(NCHW fp32, 填 0)；grid_thw(int64, 按 pixel H/W 与 patch 自动填)。
// 用法：./vision_bench model.rknn [iters=10] [patch=16]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rknn_api.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.rknn [iters] [patch]\n", argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 10;
    int patch = argc > 3 ? atoi(argv[3]) : 16;

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); return 1; }
    fclose(f);
    rknn_context ctx = 0;
    if (rknn_init(&ctx, buf.data(), sz, 0, nullptr) != 0) { fprintf(stderr, "rknn_init failed\n"); return 2; }

    rknn_input_output_num ion;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &ion, sizeof(ion));
    printf("# in=%d out=%d\n", ion.n_input, ion.n_output);

    std::vector<rknn_input> inputs(ion.n_input);
    std::vector<std::vector<char>> buffers(ion.n_input);
    std::vector<rknn_tensor_attr> attrs(ion.n_input);
    int px_h = 0, px_w = 0;
    for (uint32_t i = 0; i < ion.n_input; ++i) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a));
        a.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
        attrs[i] = a;
        buffers[i].assign(a.size_with_stride ? a.size_with_stride : a.size, 0);
        printf("# in[%u] %s ndims=%u dims=[%u,%u,%u,%u] type=%d size=%u\n", i, a.name,
               a.n_dims, a.dims[0], a.dims[1], a.dims[2], a.dims[3], a.type, a.size_with_stride);
        if (a.n_dims == 4) { px_h = a.dims[2]; px_w = a.dims[3]; }
    }
    for (uint32_t i = 0; i < ion.n_input; ++i) {
        rknn_tensor_attr& a = attrs[i];
        memset(&inputs[i], 0, sizeof(rknn_input));
        inputs[i].index = i;
        inputs[i].buf = buffers[i].data();
        inputs[i].size = buffers[i].size();
        inputs[i].pass_through = 1;
        inputs[i].type = (a.type == RKNN_TENSOR_UNDEFINED) ? RKNN_TENSOR_FLOAT32 : (rknn_tensor_type)a.type;
        inputs[i].fmt = RKNN_TENSOR_NCHW;
        // grid_thw: [t,h,w] int64 —— 由 pixel 的 H/W 推
        if (a.n_dims == 2 && a.dims[1] == 3 && (strstr(a.name, "grid") || px_h)) {
            int64_t* g = (int64_t*)buffers[i].data();
            g[0] = 1; g[1] = px_h / patch; g[2] = px_w / patch;
            printf("# fill grid_thw: %ld %ld %ld\n", (long)g[0], (long)g[1], (long)g[2]);
        }
    }
    if (rknn_inputs_set(ctx, ion.n_input, inputs.data()) != 0) { fprintf(stderr, "inputs_set failed\n"); return 3; }

    double total = 0; int n_run = 0;
    for (int i = 0; i < iters; ++i) {
        auto s0 = std::chrono::steady_clock::now();
        int ret = rknn_run(ctx, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        if (ret != 0) { printf("rknn_run ret=%d\n", ret); return 4; }
        double ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
        printf("iter %d %.1f ms\n", i, ms);
        if (i > 0) { total += ms; n_run++; }
    }
    printf("# AVG %.1f ms over %d iters\n", total / (n_run ? n_run : 1), n_run);

    for (uint32_t i = 0; i < ion.n_output; ++i) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a));
        a.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        printf("# out[%u] %s dims=[%u,%u,%u,%u]\n", i, a.name, a.dims[0], a.dims[1], a.dims[2], a.dims[3]);
    }
    rknn_destroy(ctx);
    return 0;
}
