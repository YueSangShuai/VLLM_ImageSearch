#include "stream_processor.h"
#include "frame.h"
#include "util.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

static std::string getStr(const json & j, const char * key, const std::string & def = "") {
    if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
    return def;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// ==================== 按设备过滤配置解析（devices.json 设备项，缺字段=不过滤） ====================

struct RoiRect {
    double x, y, w, h;   // 归一化 0..1
};

// roi_regions: [{"x":0.1,"y":0.2,"w":0.5,"h":0.6}, ...]；空/缺 = 全图。非法项跳过。
static std::vector<RoiRect> parseRoiRegions(const json & device) {
    std::vector<RoiRect> out;
    if (!device.contains("roi_regions") || !device["roi_regions"].is_array()) return out;
    for (const auto & r : device["roi_regions"]) {
        if (!r.is_object()) continue;
        auto num = [&r](const char * k, double & v) {
            if (!r.contains(k) || !r[k].is_number()) return false;
            v = r[k].get<double>();
            return true;
        };
        RoiRect rect{0, 0, 0, 0};
        if (!num("x", rect.x) || !num("y", rect.y) || !num("w", rect.w) || !num("h", rect.h))
            continue;
        if (rect.w <= 0 || rect.h <= 0) continue;
        out.push_back(rect);
    }
    return out;
}

// orientation_filter: ["Front","Side","Back"] 子集；空/缺 = 不限
static std::set<std::string> parseOrientationFilter(const json & device) {
    std::set<std::string> out;
    if (!device.contains("orientation_filter") || !device["orientation_filter"].is_array())
        return out;
    for (const auto & a : device["orientation_filter"]) {
        if (a.is_string()) out.insert(a.get<std::string>());
    }
    return out;
}

// min_box_w/min_h：低质量判定的宽高阈值，缺省 64/32（与历史硬编码一致）
static void parseMinBox(const json & device, int & w, int & h) {
    w = 64; h = 32;
    if (device.contains("min_box_w") && device["min_box_w"].is_number())
        w = (int)device["min_box_w"].get<double>();
    if (device.contains("min_box_h") && device["min_box_h"].is_number())
        h = (int)device["min_box_h"].get<double>();
}

// 行人框中心点落在任一区域内即保留
static bool centerInRoi(const Detection & det, const std::vector<RoiRect> & regions,
                        int frame_w, int frame_h) {
    double cx = (det.bbox[0] + det.bbox[2]) / 2.0 / frame_w;
    double cy = (det.bbox[1] + det.bbox[3]) / 2.0 / frame_h;
    for (const auto & r : regions) {
        if (cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h) return true;
    }
    return false;
}

// ---- 不规则区域（多边形）：与 roi_regions 共存，命中任一形状即保留 ----

struct RoiPoly {
    std::vector<double> x, y;   // 归一化 0..1，首尾隐式闭合
};

// 单设备上限：区域是每帧解析的，防手改 json 塞进几千个顶点把设备线程拖慢
static const size_t kMaxPolysPerDevice = 16;
static const size_t kMaxPolyVertices = 64;

// roi_polygons: [[{"x":..,"y":..}, ...], ...]；空/缺 = 全图。非法多边形整块丢弃。
static std::vector<RoiPoly> parseRoiPolys(const json & device) {
    std::vector<RoiPoly> out;
    if (!device.contains("roi_polygons") || !device["roi_polygons"].is_array()) return out;
    auto clamp01 = [](double v) { return v < 0 ? 0.0 : (v > 1.0 ? 1.0 : v); };
    for (const auto & raw : device["roi_polygons"]) {
        if (out.size() >= kMaxPolysPerDevice) break;
        if (!raw.is_array()) continue;
        RoiPoly p;
        for (const auto & pt : raw) {
            if (p.x.size() >= kMaxPolyVertices) break;
            if (!pt.is_object() || !pt.contains("x") || !pt.contains("y")) continue;
            if (!pt["x"].is_number() || !pt["y"].is_number()) continue;
            double x = clamp01(pt["x"].get<double>());
            double y = clamp01(pt["y"].get<double>());
            // 相邻重复点会让射线法的 (y_i>py)!=(y_j>py) 判定退化，直接丢
            if (!p.x.empty() && p.x.back() == x && p.y.back() == y) continue;
            p.x.push_back(x);
            p.y.push_back(y);
        }
        if (p.x.size() > 1 && p.x.front() == p.x.back() && p.y.front() == p.y.back()) {
            p.x.pop_back();  // 显式闭合的首尾同点：隐式闭合已覆盖
            p.y.pop_back();
        }
        if (p.x.size() < 3) continue;
        double area2 = 0;  // 鞋带公式：共线/面积极小的退化多边形没有判定意义
        for (size_t i = 0, j = p.x.size() - 1; i < p.x.size(); j = i++)
            area2 += p.x[j] * p.y[i] - p.x[i] * p.y[j];
        if (std::fabs(area2) < 1e-6) continue;
        out.push_back(std::move(p));
    }
    return out;
}

// even-odd 射线法。在归一化空间判定成立：查询点与各轴按同样方式缩放，包含关系不变。
static bool pointInPolygon(double px, double py, const RoiPoly & p) {
    bool inside = false;
    for (size_t i = 0, j = p.x.size() - 1; i < p.x.size(); j = i++) {
        if ((p.y[i] > py) != (p.y[j] > py)) {
            double xint = p.x[i] + (py - p.y[i]) / (p.y[j] - p.y[i]) * (p.x[j] - p.x[i]);
            if (px < xint) inside = !inside;
        }
    }
    return inside;
}

// 脚点判定：bbox 底边中点，判的是「人站在哪块地上」。
// 与矩形的「框中心」口径不同是刻意的：不规则区域通常圈的是地面范围，用框中心会在
// 人贴边界时把站在外面的人算进来。矩形沿用历史中心口径，已配好的行为不漂移。
static bool footInPoly(const Detection & det, const std::vector<RoiPoly> & polys,
                       int frame_w, int frame_h) {
    double fx = (det.bbox[0] + det.bbox[2]) / 2.0 / frame_w;
    double fy = det.bbox[3] / (double)frame_h;
    for (const auto & p : polys)
        if (pointInPolygon(fx, fy, p)) return true;
    return false;
}

struct RoiShapes {
    std::vector<RoiRect> rects;
    std::vector<RoiPoly> polys;
    bool empty() const { return rects.empty() && polys.empty(); }
};

static RoiShapes parseRoiShapes(const json & device) {
    RoiShapes s;
    s.rects = parseRoiRegions(device);
    s.polys = parseRoiPolys(device);
    return s;
}

static bool insideRoi(const Detection & det, const RoiShapes & shapes, int frame_w, int frame_h) {
    return centerInRoi(det, shapes.rects, frame_w, frame_h) ||
           footInPoly(det, shapes.polys, frame_w, frame_h);
}

// 证据图上叠画区域边界（青色），与绿色人框区分：拉图回来能直接判断人当时在不在区域内
static void drawRoiOverlay(ImageRGB & img, const RoiShapes & shapes) {
    for (const auto & r : shapes.rects) {
        draw_polyline(img, {
            {(int)(r.x * img.w), (int)(r.y * img.h)},
            {(int)((r.x + r.w) * img.w), (int)(r.y * img.h)},
            {(int)((r.x + r.w) * img.w), (int)((r.y + r.h) * img.h)},
            {(int)(r.x * img.w), (int)((r.y + r.h) * img.h)} }, true);
    }
    for (const auto & p : shapes.polys) {
        std::vector<std::pair<int, int>> pts;
        pts.reserve(p.x.size());
        for (size_t i = 0; i < p.x.size(); i++)
            pts.emplace_back((int)(p.x[i] * img.w), (int)(p.y[i] * img.h));
        draw_polyline(img, pts, true);
    }
}

// 朝向事后过滤：属性头输出的 Front/Side/Back 属性里没有任何一个落在 allowed 时判为不符。
// attributes 里完全没有朝向属性（属性头没判出来）→ 保留，不误杀。
static bool orientationAllowed(const json & match, const std::set<std::string> & allowed) {
    bool has_orient = false;
    if (match.contains("attributes") && match["attributes"].is_array()) {
        for (const auto & a : match["attributes"]) {
            if (!a.is_string()) continue;
            std::string s = a.get<std::string>();
            if (s == "Front" || s == "Side" || s == "Back") {
                has_orient = true;
                if (allowed.count(s)) return true;
            }
        }
    }
    return !has_orient;
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

StreamProcessor::StreamProcessor(const json & device, Detector * detector, Matcher * matcher,
                                 bool debug_capture)
    : device_(device), detector_(detector), matcher_(matcher), debug_capture_(debug_capture) {}

void StreamProcessor::updateDevice(const json & device) {
    std::lock_guard<std::mutex> lk(dev_mu_);
    device_ = device;
}

json StreamProcessor::deviceSnapshot() const {
    std::lock_guard<std::mutex> lk(dev_mu_);
    return device_;
}

void StreamProcessor::run() {
    // StreamProcessor.start()：ffmpeg 单帧拉取 + 指数退避重连
    json device = deviceSnapshot();
    std::string rtsp_url = device_rtsp_url(device);
    std::string device_name = getStr(device, "name", "Unknown");
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
    json device = deviceSnapshot();
    std::vector<Detection> detections = detector_->detect(frame);
    if (!detections.empty()) {
        printf("[%s] 检测到 %zu 个行人\n", getStr(device, "name", "Unknown").c_str(),
               detections.size());
    }

    // 检测区域过滤：roi_regions（矩形，判框中心）/ roi_polygons（不规则，判脚点）任一非空时，
    // 只保留落在其中的检出（省下 VLM 描述+匹配的大头）
    RoiShapes shapes = parseRoiShapes(device);
    if (!shapes.empty() && !detections.empty()) {
        std::vector<Detection> kept, outside;
        for (const auto & det : detections) {
            if (insideRoi(det, shapes, frame.w, frame.h)) kept.push_back(det);
            else outside.push_back(det);
        }
        if (kept.size() != detections.size())
            printf("[%s] ROI 区域过滤: 保留 %zu/%zu 个行人\n",
                   getStr(device, "name", "Unknown").c_str(), kept.size(), detections.size());
        detections.swap(kept);

        // 区域外行人也进调试页（排障：看区域圈得对不对、有没有误圈），但只落图，
        // 不走 VLM 描述与规则匹配——这正是 ROI 过滤想省的大头
        if (debug_capture_ && !outside.empty()) {
            int min_w, min_h;
            parseMinBox(device, min_w, min_h);
            for (const auto & det : outside) {
                try {
                    ImageRGB person;
                    if (!crop_region(frame, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                                     person))
                        continue;
                    bool is_low_quality = (person.w < min_w || person.h < min_h);
                    captureDebug(person, frame, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                                 is_low_quality, json::array(), /*in_region=*/false);
                } catch (const std::exception & e) {
                    printf("区域外调试捕获异常: %s\n", e.what());
                } catch (...) {
                    printf("区域外调试捕获异常\n");
                }
            }
        }
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

// VLM_DEBUG_CAPTURE=1：所有检测到的行人（无论是否命中规则）落盘 ROI/原图 + 描述快照，
// 写入 data/debug_log.json，web /debug 页展示。图片按 日期/设备/roi|original 分目录，
// json_store 截断 1000 条时会顺带删除被逐出的图片。
// in_region=false（区域外行人）：只落图，不取/不补描述快照，条目打 outside_roi 标。
// orient_dropped：朝向事后过滤丢弃的规则名（match() 先跑、朝向过滤后行，快照里这些规则
// 仍是 hit，改标 orientation 让页面能写明「朝向不符合」）。
void StreamProcessor::captureDebug(const ImageRGB & person, const ImageRGB & frame,
                                   int x1, int y1, int x2, int y2, bool low_quality,
                                   const json & matches, bool in_region,
                                   const json & orient_dropped) {
    json device = deviceSnapshot();
    // 空对象而非 null：null 上 value() 抛 type_error.306（实测踩中）
    json snap = json::object();
    if (in_region) {
        snap = matcher_->lastDebugSnapshot();
        // 未经描述路径（无规则等）时补一次描述生成
        if (!snap.contains("description")) snap = matcher_->describeForDebug(person);
    }

    bool matched = matches.is_array() && !matches.empty();
    int64_t dbg_count = now_ms() % 1000000;
    std::string safe_device_name = sanitize_name(getStr(device, "name"));
    std::string today = today_str();

    std::string roi_dir = g_paths->debug_dir + "/" + today + "/" + safe_device_name + "/roi";
    std::string orig_dir = g_paths->debug_dir + "/" + today + "/" + safe_device_name + "/original";
    std::error_code ec;
    fs::create_directories(roi_dir, ec);
    fs::create_directories(orig_dir, ec);

    std::string img_filename = "debug_" + std::to_string(now_ms()) + "_" +
                               std::to_string(dbg_count) + ".jpg";
    {
        std::ofstream f(roi_dir + "/" + img_filename, std::ios::binary);
        auto jpg = encode_jpeg(resize_bilinear(person, 224, 224), 95);
        f.write((const char *)jpg.data(), (std::streamsize)jpg.size());
    }
    {
        ImageRGB annotated = frame;
        draw_rect(annotated, x1, y1, x2, y2);
        drawRoiOverlay(annotated, parseRoiShapes(device));
        std::ofstream f(orig_dir + "/" + img_filename, std::ios::binary);
        auto jpg = encode_jpeg(annotated, 90);
        f.write((const char *)jpg.data(), (std::streamsize)jpg.size());
    }

    json rules = json::array();
    if (matched) {
        for (const auto & m : matches) {
            rules.push_back({{"rule_name", m.value("rule_name", "")},
                             {"score", m.value("final_score", 0.0)}});
        }
    }

    // 逐规则得分明细（含未命中/冷却中）：落盘 + stdout 打印；区域外未做匹配，不打
    json rule_details = snap.contains("rules") ? snap["rules"] : json::array();
    // 朝向过滤发生在 match() 之后：快照里被丢弃的规则仍是 hit，改标 orientation
    if (orient_dropped.is_array() && !orient_dropped.empty()) {
        for (auto & d : rule_details) {
            if (d.value("status", "") != "hit" || !d.contains("rule_name")) continue;
            std::string rn = d.value("rule_name", "");
            for (const auto & o : orient_dropped)
                if (o.is_string() && o.get<std::string>() == rn) {
                    d["status"] = "orientation";
                    break;
                }
        }
    }
    if (in_region) {
        std::string hit_s, miss_s;
        for (const auto & d : rule_details) {
            std::string item = d.value("rule_name", "?") + "(" +
                               format_double(d.value("score", 0.0)) + "/" +
                               format_double(d.value("threshold", 0.5)) + "阈)";
            if (d.value("type", "") == "symbolic") {
                auto missing = d.contains("missing_attrs") ? d["missing_attrs"] : json::array();
                if (!missing.empty()) {
                    std::string ms;
                    for (const auto & a : missing) ms += (ms.empty() ? "" : ",") + a.get<std::string>();
                    item += "[缺:" + ms + "]";
                }
            }
            std::string st = d.value("status", "");
            if (st == "hit") hit_s += (hit_s.empty() ? "" : " ") + item;
            else miss_s += (miss_s.empty() ? "" : " ") + item +
                           (st == "cooldown" ? "(冷却)" : st == "no_embedding" ? "(无向量)" :
                            st == "orientation" ? "(朝向不符)" : "");
        }
        printf("[%s] 🐞 匹配明细 | 命中: %s | 未命中: %s\n",
               sanitize_name(getStr(device, "name", "Unknown")).c_str(),
               hit_s.empty() ? "无" : hit_s.c_str(),
               miss_s.empty() ? "无" : miss_s.c_str());
    }

    json entry;
    entry["timestamp"] = now_isoformat();
    entry["device_id"] = device.value("id", "");
    entry["device_name"] = device.value("name", "");
    entry["matched"] = matched;
    entry["matched_rules"] = rules;
    entry["rule_details"] = rule_details;   // 全部规则得分与未命中属性（/debug 页展示）
    entry["description"] = snap.value("description", "");
    entry["description_zh"] = snap.value("description_zh", "");
    entry["attributes"] = snap.contains("attributes") ? snap["attributes"] : json::array();
    entry["low_quality"] = low_quality;
    entry["image"] = today + "/" + safe_device_name + "/roi/" + img_filename;
    entry["original_image"] = today + "/" + safe_device_name + "/original/" + img_filename;
    entry["bbox"] = json::array({x1, y1, x2, y2});
    if (!in_region) entry["outside_roi"] = true;   // 区域内条目无此字段，前端按 false 处理
    if (orient_dropped.is_array() && !orient_dropped.empty())
        entry["orientation_filtered"] = orient_dropped;  // 被朝向过滤丢弃的规则名
    append_debug_log(entry);
}

void StreamProcessor::processDetection(const Detection & det, const ImageRGB & frame) {
    // _process_detection：ROI 匹配 -> 告警落盘 -> 日志/回调
    json device = deviceSnapshot();
    int x1 = det.bbox[0], y1 = det.bbox[1], x2 = det.bbox[2], y2 = det.bbox[3];

    ImageRGB person;
    if (!crop_region(frame, x1, y1, x2, y2, person)) return;
    int w = person.w, h = person.h;
    // 低质量长宽阈值按设备可配（min_box_w/min_h），照常告警，仅页面标注
    int min_w, min_h;
    parseMinBox(device, min_w, min_h);
    bool is_low_quality = (w < min_w || h < min_h);

    json matches = matcher_->match(person);

    // 朝向事后过滤：orientation_filter 非空时，命中的属性里没有任何允许朝向则丢弃该条。
    // 丢弃的规则名记进 orient_dropped，调试页据此标「朝向不符」（否则只剩个笼统的未命中）
    std::set<std::string> allowed_orient = parseOrientationFilter(device);
    json orient_dropped = json::array();
    if (!allowed_orient.empty() && matches.is_array()) {
        json kept = json::array();
        std::vector<std::string> dropped_ids;
        for (auto & m : matches) {
            if (orientationAllowed(m, allowed_orient)) {
                kept.push_back(m);
            } else {
                printf("[%s] 朝向过滤丢弃: %s（朝向不在勾选范围）\n",
                       getStr(device, "name", "Unknown").c_str(),
                       m.value("rule_name", "?").c_str());
                orient_dropped.push_back(m.value("rule_name", "?"));
                std::string rid = m.value("rule_id", "");
                if (!rid.empty()) dropped_ids.push_back(rid);
            }
        }
        matches.swap(kept);
        // match() 不知道朝向过滤、触发即启动冷却；被丢弃的命中不该占用规则冷却，
        // 否则背面人反复命中「黑衣服男生」会把冷却一直顶住，朝向对的人反而报不出
        if (!dropped_ids.empty()) matcher_->undoTrigger(dropped_ids);
    }

    if (debug_capture_) {
        try {
            captureDebug(person, frame, x1, y1, x2, y2, is_low_quality, matches,
                         /*in_region=*/true, orient_dropped);
        } catch (const std::exception & e) {
            printf("调试捕获异常: %s\n", e.what());
        } catch (...) {
            printf("调试捕获异常\n");
        }
    }

    if (!matches.is_array() || matches.empty()) return;

    int64_t alert_count = now_ms() % 1000000;

    std::string safe_device_name = sanitize_name(getStr(device, "name"));
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
        // 原图画框（绿色 2px）+ 青色区域边界
        {
            ImageRGB annotated = frame;
            draw_rect(annotated, x1, y1, x2, y2);
            drawRoiOverlay(annotated, parseRoiShapes(device));
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
        log_entry["device_id"] = device.value("id", "");
        log_entry["device_name"] = device.value("name", "");
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
        alert["device_id"] = device.value("id", "");
        alert["device_name"] = device.value("name", "");
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
