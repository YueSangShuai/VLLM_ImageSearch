#pragma once
// YOLOv8n 行人检测：ONNX Runtime 推理，复刻 ultralytics 的 letterbox(auto) + 解码 + NMS。
// 接口对齐 Python PersonDetector.detect：输入帧（RGB），输出原帧坐标 bbox + conf。
#include "frame.h"
#include <onnxruntime_cxx_api.h>
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
    Detector(const std::string &onnx_path, float conf_th = 0.5f, float iou_th = 0.7f);
    std::vector<Detection> detect(const ImageRGB &frame);

private:
    struct Letterboxed {
        ImageRGB img;
        float ratio;   // 原图/缩放后 的逆（=1/r）
        float pad_l, pad_t;
    };
    Letterboxed letterbox(const ImageRGB &frame) const;

    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    float conf_th_, iou_th_;
    int max_side_ = 640, stride_ = 32;
    std::mutex mu_;
};
