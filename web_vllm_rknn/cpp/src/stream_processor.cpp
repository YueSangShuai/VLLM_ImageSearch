#include "stream_processor.h"
#include "frame.h"
#include "util.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

static std::string getStr(const json & j, const char * key, const std::string & def = "") {
    if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
    return def;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string device_rtsp_url(const json & device) {
    std::string stream_type = getStr(device, "stream_type", "main");
    std::string path = stream_type == "main"
        ? getStr(device, "main_stream_path", "/ch01")
        : getStr(device, "sub_stream_path", "/ch01_sub");
    std::string port = getStr(device, "port", "554");
    if (port.empty()) port = "554";
    return build_rtsp_url(getStr(device, "username"), getStr(device, "password"),
                          getStr(device, "ip"), port, path);
}

std::string device_rtsp_url_raw(const json & device) {
    // 与 app.py video_frame 一致：不编码用户名/密码
    std::string stream_type = getStr(device, "stream_type", "main");
    std::string path = stream_type == "main"
        ? getStr(device, "main_stream_path", "/ch01")
        : getStr(device, "sub_stream_path", "/ch01_sub");
    std::string port = getStr(device, "port", "554");
    if (port.empty()) port = "554";
    return "rtsp://" + getStr(device, "username") + ":" + getStr(device, "password") +
           "@" + getStr(device, "ip") + ":" + port + path;
}

StreamProcessor::StreamProcessor(const json & device, Detector * detector, Matcher * matcher)
    : device_(device), detector_(detector), matcher_(matcher) {}

void StreamProcessor::run() {
    // StreamProcessor.start()：ffmpeg 单帧拉取 + 指数退避重连
    std::string rtsp_url = device_rtsp_url(device_);
    std::string device_name = getStr(device_, "name", "Unknown");
    printf("[%s] 开始连接流: %s\n", device_name.c_str(), rtsp_url.c_str());

    running_ = true;
    int reconnect_delay = 1;
    const int max_reconnect_delay = 30;

    while (running_) {
        bool ok = false;
        ImageRGB frame;
        std::vector<uint8_t> jpeg = grab_frame_jpeg(rtsp_url, 10);
        if (jpeg.empty()) {
            printf("[%s] 读取超时，等待重连...\n", device_name.c_str());
        } else if (!decode_jpeg(jpeg, frame)) {
            printf("[%s] 处理异常: Failed to decode frame\n", device_name.c_str());
        } else {
            reconnect_delay = 1;  // 成功后重置延迟
            ok = true;
        }
        if (ok && running_) processFrame(frame);

        if (running_) {
            // 分段 sleep，便于 stop() 尽快生效
            for (int i = 0; i < reconnect_delay * 10 && running_; i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            reconnect_delay = std::min(reconnect_delay * 2, max_reconnect_delay);
        }
    }
}

void StreamProcessor::stop() { running_ = false; }

// 从 frame 裁剪 [x1,y1,x2,y2)，越界自动收敛（对齐 numpy 切片语义）
static bool crop_region(const ImageRGB & src, int x1, int y1, int x2, int y2, ImageRGB & out) {
    x1 = std::max(0, std::min(x1, src.w));
    x2 = std::max(0, std::min(x2, src.w));
    y1 = std::max(0, std::min(y1, src.h));
    y2 = std::max(0, std::min(y2, src.h));
    int w = x2 - x1, h = y2 - y1;
    if (w <= 0 || h <= 0) return false;
    out.w = w; out.h = h;
    out.rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; y++) {
        const uint8_t * srow = src.rgb.data() + ((size_t)(y + y1) * src.w + x1) * 3;
        memcpy(out.rgb.data() + (size_t)y * w * 3, srow, (size_t)w * 3);
    }
    return true;
}

void StreamProcessor::processFrame(const ImageRGB & frame) {
    // _process_frame：错误隔离
    std::vector<Detection> detections = detector_->detect(frame);
    if (!detections.empty()) {
        printf("[%s] 检测到 %zu 个行人\n", getStr(device_, "name", "Unknown").c_str(),
               detections.size());
    }
    for (const auto & det : detections) {
        try {
            processDetection(det, frame);
        } catch (const std::exception & e) {
            printf("处理检测异常: %s\n", e.what());
        } catch (...) {
            printf("处理检测异常\n");
        }
    }
}

void StreamProcessor::processDetection(const Detection & det, const ImageRGB & frame) {
    // _process_detection：ROI 匹配 -> 告警落盘 -> 日志/回调
    int x1 = det.bbox[0], y1 = det.bbox[1], x2 = det.bbox[2], y2 = det.bbox[3];

    ImageRGB person;
    if (!crop_region(frame, x1, y1, x2, y2, person)) return;
    int w = person.w, h = person.h;
    bool is_low_quality = (w < 64 || h < 32);

    json matches = matcher_->match(person);
    if (!matches.is_array() || matches.empty()) return;

    int64_t alert_count = now_ms() % 1000000;

    std::string safe_device_name = sanitize_name(getStr(device_, "name"));
    for (const auto & match : matches) {
        std::string timestamp = now_isoformat();
        std::string today = today_str();
        std::string safe_rule_name = sanitize_name(
            match.contains("rule_name") && match["rule_name"].is_string()
                ? match["rule_name"].get<std::string>() : "");

        std::string roi_dir = g_paths->alerts_dir + "/" + today + "/" + safe_device_name +
                              "/" + safe_rule_name + "/roi";
        std::string orig_dir = g_paths->alerts_dir + "/" + today + "/" + safe_device_name +
                               "/" + safe_rule_name + "/original";
        std::error_code ec;
        fs::create_directories(roi_dir, ec);
        fs::create_directories(orig_dir, ec);

        std::string img_filename = "alert_" + std::to_string(now_ms()) + "_" +
                                   std::to_string(alert_count) + ".jpg";

        // ROI 存 224x224（cv2.resize 默认双线性）
        ImageRGB roi = resize_bilinear(person, 224, 224);
        {
            std::ofstream f(roi_dir + "/" + img_filename, std::ios::binary);
            auto jpg = encode_jpeg(roi, 95);
            f.write((const char *)jpg.data(), (std::streamsize)jpg.size());
        }
        // 原图画框（绿色 2px）
        {
            ImageRGB annotated = frame;
            draw_rect(annotated, x1, y1, x2, y2);
            std::ofstream f(orig_dir + "/" + img_filename, std::ios::binary);
            auto jpg = encode_jpeg(annotated, 95);
            f.write((const char *)jpg.data(), (std::streamsize)jpg.size());
        }

        printf("🚨 告警: %s 正样本分数: %s 负样本分数: %s\n",
               match.value("rule_name", "").c_str(),
               format_double(match.value("final_score", 0.0)).c_str(),
               match.contains("negative_scores") ? match["negative_scores"].dump().c_str() : "[]");

        std::string rel_roi = today + "/" + safe_device_name + "/" + safe_rule_name + "/roi/" + img_filename;
        std::string rel_orig = today + "/" + safe_device_name + "/" + safe_rule_name + "/original/" + img_filename;

        json bbox = json::array({x1, y1, x2, y2});

        // 检测日志
        json log_entry;
        log_entry["timestamp"] = timestamp;
        log_entry["device_id"] = device_.value("id", "");
        log_entry["device_name"] = device_.value("name", "");
        log_entry["rule_id"] = match.value("rule_id", "");
        log_entry["rule_name"] = match.value("rule_name", "");
        log_entry["positive"] = match.value("positive", "");
        log_entry["positive_list"] = match.contains("positive_list") ? match["positive_list"] : json::array();
        log_entry["negative_list"] = match.contains("negative_list") ? match["negative_list"] : json::array();
        log_entry["positive_scores"] = match.contains("positive_scores") ? match["positive_scores"] : json::array();
        log_entry["negative_scores"] = match.contains("negative_scores") ? match["negative_scores"] : json::array();
        log_entry["description"] = match.value("description", "");
        log_entry["base_description"] = match.value("base_description", "");
        log_entry["description_zh"] = match.value("description_zh", "");
        log_entry["attributes"] = match.contains("attributes") ? match["attributes"] : json::array();
        log_entry["score"] = match.value("final_score", 0.0);
        log_entry["negative_higher"] = match.value("negative_higher", false);
        log_entry["image"] = rel_roi;
        log_entry["original_image"] = rel_orig;
        log_entry["bbox"] = bbox;
        append_detection_log(log_entry);

        // 告警（完整规则信息）
        json alert;
        alert["id"] = "alert_" + std::to_string(now_ms() / 1000) + "_" + std::to_string(alert_count);
        alert["device_id"] = device_.value("id", "");
        alert["device_name"] = device_.value("name", "");
        alert["rule_id"] = match.value("rule_id", "");
        alert["rule_name"] = match.value("rule_name", "");
        alert["positive"] = match.value("positive", "");
        alert["positive_list"] = match.contains("positive_list") ? match["positive_list"] : json::array();
        alert["negative_list"] = match.contains("negative_list") ? match["negative_list"] : json::array();
        alert["positive_scores"] = match.contains("positive_scores") ? match["positive_scores"] : json::array();
        alert["negative_scores"] = match.contains("negative_scores") ? match["negative_scores"] : json::array();
        alert["description"] = match.value("description", "");
        alert["base_description"] = match.value("base_description", "");
        alert["description_zh"] = match.value("description_zh", "");
        alert["attributes"] = match.contains("attributes") ? match["attributes"] : json::array();
        alert["score"] = match.value("final_score", 0.0);
        alert["negative_higher"] = match.value("negative_higher", false);
        alert["timestamp"] = timestamp;
        alert["image"] = rel_roi;
        alert["original_image"] = rel_orig;
        alert["bbox"] = bbox;
        alert["low_quality"] = is_low_quality;
        alert["img_size"] = {{"width", w}, {"height", h}};

        if (on_alert) on_alert(alert);
    }
}
