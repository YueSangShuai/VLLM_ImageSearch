#include "siglip_vision.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

SiglipVision::SiglipVision(const std::string &rknn_path) {
    int ret = rknn_init(&ctx_, (void *)rknn_path.c_str(), 0, 0, NULL);
    if (ret < 0) throw std::runtime_error("rknn_init SigLIP 视觉塔失败 ret=" + std::to_string(ret));
    const char *cores = getenv("RKNN_SIGLIP_CORE_NUM");
    int core = (cores && atoi(cores) > 0) ? atoi(cores) : 1;  // 默认单核，与 YOLO 分时共享
    ret = rknn_set_core_mask(ctx_, core == 3 ? RKNN_NPU_CORE_0_1_2
                                 : core == 2 ? RKNN_NPU_CORE_0_1
                                             : RKNN_NPU_CORE_0);
    if (ret < 0) throw std::runtime_error("rknn_set_core_mask SigLIP 失败");
    rknn_input_output_num io_num = {};
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    in_attr_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_));
    out_attr_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr_, sizeof(out_attr_));
    dim_ = (int)out_attr_.n_elems;
    fprintf(stderr, "SigLIP vision rknn in: dims=[%d,%d,%d,%d] out: n_elems=%d (n_out=%u)\n",
            in_attr_.dims[0], in_attr_.dims[1], in_attr_.dims[2], in_attr_.dims[3],
            (int)out_attr_.n_elems, (unsigned)io_num.n_output);
    if (io_num.n_output != 1)
        fprintf(stderr, "警告: 期望 1 个输出，实际 %u 个（只取 index 0）\n", (unsigned)io_num.n_output);
}

SiglipVision::~SiglipVision() {
    if (ctx_) rknn_destroy(ctx_);
}

std::vector<float> SiglipVision::embed(const ImageRGB &crop) {
    std::lock_guard<std::mutex> lk(mu_);
    // processor 口径（preprocessor_config.json size={224,224}）：非等比直缩 224x224，
    // 无中心裁剪；resample=bicubic，这里用双线性近似（与原框架 YOLO/VLM 一致，
    // golden 对拍量化过双线性/双三次对 embedding cosine 的影响）
    ImageRGB img = resize_bilinear(crop, 224, 224);

    rknn_input in = {};
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = (uint32_t)img.rgb.size();
    in.pass_through = 0;   // 走 input 归一化（mean/std 烘在模型里）
    in.buf = img.rgb.data();
    int ret = rknn_inputs_set(ctx_, 1, &in);
    if (ret < 0) throw std::runtime_error("SigLIP rknn_inputs_set 失败");
    ret = rknn_run(ctx_, nullptr);
    if (ret < 0) throw std::runtime_error("SigLIP rknn_run 失败");

    rknn_output out = {};
    out.index = 0;
    out.want_float = 1;
    ret = rknn_outputs_get(ctx_, 1, &out, NULL);
    if (ret < 0) throw std::runtime_error("SigLIP rknn_outputs_get 失败 ret=" + std::to_string(ret));

    size_t n = out_attr_.n_elems;
    std::vector<float> vec(n);
    memcpy(vec.data(), out.buf, n * sizeof(float));   // want_float=1 → 恒 float32
    rknn_outputs_release(ctx_, 1, &out);

    double norm = 0;
    for (float v : vec) norm += (double)v * v;
    norm = std::sqrt(norm);
    if (norm > 0) for (auto &v : vec) v = (float)(v / norm);
    dim_ = (int)n;
    return vec;
}
