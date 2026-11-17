#pragma once
// 对应 processor.py StreamProcessor：单设备 RTSP 取流 -> YOLO 检测 -> 匹配 -> 告警落盘
#include "detector.h"
#include "json_store.h"
#include "matcher.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <vector>

// DeviceManager.get_rtsp_url：用户名/密码 URL 编码，按 stream_type 选主/子码流
std::string device_rtsp_url(const json & device);
// 与 app.py video_frame 一致：预览用未编码 URL（历史行为，保持一致）
std::string device_rtsp_url_raw(const json & device);

class StreamProcessor {
public:
    StreamProcessor(const json & device, Detector * detector, Matcher * matcher,
                    bool debug_capture = false);

    void run();          // 设备线程主循环（对应 python start()）
    void stop();

    // 运行时热更新设备配置（ROI 矩形 roi_regions / 不规则区域 roi_polygons / 朝向过滤 /
    // 低质量阈值存于 devices.json 设备项，缺字段 = 全图 / 不限 / 64x32）。
    // HTTP/刷新线程写，设备线程按帧读，dev_mu_ 互斥。
    void updateDevice(const json & device);

    std::function<void(const json &)> on_alert;
    const json & deviceInfo() const { return device_; }

private:
    json deviceSnapshot() const;             // 加锁拷贝，供每帧配置解析
    void processFrame(const ImageRGB & frame);
    void processDetection(const Detection & det, const ImageRGB & frame);
    void captureDebug(const ImageRGB & person, const ImageRGB & frame,
                      int x1, int y1, int x2, int y2, bool low_quality,
                      const json & matches,
                      bool in_region = true,   // false 走区域外轻量路径（不调 VLM/匹配）
                      const json & orient_dropped = json());  // 朝向事后过滤丢弃的规则名

    json device_;
    mutable std::mutex dev_mu_;
    Detector * detector_;
    Matcher * matcher_;
    bool debug_capture_ = false;
    std::atomic<bool> running_{false};
};
