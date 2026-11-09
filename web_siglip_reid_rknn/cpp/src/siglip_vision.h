#pragma once
// SigLIP2 视觉塔（RKNN, rk3588）：行人裁剪 → 224x224 → 768-d L2 归一化图像向量。
// 输入约定：uint8 RGB NHWC 0-255 直喂，(x-127.5)/127.5 归一化烘在模型 input norm 里
// （export/rknn_convert.py 的 mean/std_values），与 golden pixel_pm1([-1,1] NCHW) 等价。
// 与 Detector 同款 rknn C API 用法（want_float=1、全输出请求）。
#include "frame.h"
#include <rknn_api.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class SiglipVision {
public:
    explicit SiglipVision(const std::string &rknn_path);
    ~SiglipVision();

    // 返回 L2 归一化后的图像向量（dim 见 dim()，正常 768）
    std::vector<float> embed(const ImageRGB &crop);
    int dim() const { return dim_; }

private:
    rknn_context ctx_ = 0;
    rknn_tensor_attr in_attr_ = {};
    rknn_tensor_attr out_attr_ = {};
    int dim_ = 768;
    std::mutex mu_;   // 与 rkllm 版一致：NPU 会话级串行
};
