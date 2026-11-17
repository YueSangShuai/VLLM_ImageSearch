#pragma once
// MARS 图像塔（RKNN rk3588）：行人图 → 224x224 → 256-d L2 归一化向量。
// 输入约定与 siglip_vision 相同：uint8 RGB NHWC 0-255 直喂，(x-127.5)/127.5
// 归一化烘在模型 input norm 里（export/rknn_convert.py mean/std_values）。
// 图内已含 vision_proj + L2 归一化（onnx_export.py），此处再归一化一次兜底。
#include "frame.h"
#include <rknn_api.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class ImageEmbedder {
public:
    explicit ImageEmbedder(const std::string &rknn_path);
    ~ImageEmbedder();

    // 返回 L2 归一化后的 256-d 图像向量
    std::vector<float> embed(const ImageRGB &img);
    int dim() const { return dim_; }

private:
    rknn_context ctx_ = 0;
    rknn_tensor_attr in_attr_ = {};
    rknn_tensor_attr out_attr_ = {};
    int dim_ = 256;
    std::mutex mu_;   // NPU 会话级串行
};
