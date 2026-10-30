#pragma once
// YOLOv8n 行人检测：复刻 ultralytics 的 letterbox(auto) + 解码 + NMS。
// 接口对齐 Python PersonDetector.detect：输入帧（RGB），输出原帧坐标 bbox + conf。
// 后端编译期二选一：默认 ONNX Runtime（动态 letterbox 尺寸）；
// 定义 WEBVLLM_RKNN -> RKNN C API（固定 640x640 方形 letterbox，后处理几何不变）。
#include "frame.h"
#ifdef WEBVLLM_RKNN
#include "rknn_api.h"
#else
#include <onnxruntime_cxx_api.h>
#endif
#include <string>
#include <vector>
#include <memory>
#include <mutex>

struct Detection {
    int bbox[4];  // x1,y1,x2,y2（原帧坐标，int 截断，与 Python 一致）
    float confidence;
};

class Detector {
public:
    // rknn 后端时 onnx_path 传 .rknn 模型路径
    Detector(const std::string &onnx_path, float conf_th = 0.5f, float iou_th = 0.7f);
    ~Detector();
    std::vector<Detection> detect(const ImageRGB &frame);

private:
    struct Letterboxed {
        ImageRGB img;
        float ratio;   // 原图/缩放后 的逆（=1/r）
        float pad_l, pad_t;
    };
    Letterboxed letterbox(const ImageRGB &frame) const;

#ifdef WEBVLLM_RKNN
    rknn_context rknn_ctx_ = 0;
    rknn_input_output_num io_num_{};
    rknn_tensor_attr in_attr_{};
    rknn_tensor_attr out_attr_{};
#else
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
#endif
    float conf_th_, iou_th_;
    int max_side_ = 640, stride_ = 32;
    std::mutex mu_;
};
