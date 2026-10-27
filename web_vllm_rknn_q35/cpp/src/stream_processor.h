#pragma once
// 对应 processor.py StreamProcessor：单设备 RTSP 取流 -> YOLO 检测 -> 匹配 -> 告警落盘
#include "detector.h"
#include "json_store.h"
#include "matcher.h"

#include <atomic>
#include <functional>
#include <string>

// DeviceManager.get_rtsp_url：用户名/密码 URL 编码，按 stream_type 选主/子码流
std::string device_rtsp_url(const json & device);
// 与 app.py video_frame 一致：预览用未编码 URL（历史行为，保持一致）
std::string device_rtsp_url_raw(const json & device);

class StreamProcessor {
public:
    StreamProcessor(const json & device, Detector * detector, Matcher * matcher);

    void run();          // 设备线程主循环（对应 python start()）
    void stop();

    std::function<void(const json &)> on_alert;
    const json & deviceInfo() const { return device_; }

private:
    void processFrame(const ImageRGB & frame);
    void processDetection(const Detection & det, const ImageRGB & frame);

    json device_;
    Detector * detector_;
    Matcher * matcher_;
    std::atomic<bool> running_{false};
};
