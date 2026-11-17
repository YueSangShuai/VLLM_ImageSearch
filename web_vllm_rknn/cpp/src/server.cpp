#include "server.h"
#include "frame.h"
#include "stream_processor.h"
#include "util.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

// ==================== 小工具 ====================

static void res_json(httplib::Response & res, const json & j, int status = 200) {
    res.status = status;
    // Flask jsonify 默认 ensure_ascii=True，nlohmann dump() 默认同样转义非 ASCII，行为一致
    res.set_content(j.dump(), "application/json");
}

static json parse_body(const httplib::Request & req) {
    json j = json::parse(req.body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return json::object();
    return j;
}

static std::string body_str(const json & d, const char * key, const std::string & def = "") {
    if (d.contains(key) && d[key].is_string()) return d[key].get<std::string>();
    return def;
}

static double body_num(const json & d, const char * key, double def) {
    if (!d.contains(key)) return def;
    const json & v = d[key];
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) { try { return std::stod(v.get<std::string>()); } catch (...) { return def; } }
    return def;
}

static json body_val(const json & d, const char * key, const json & def) {
    if (d.contains(key)) return d[key];
    return def;
}

// ---- 按设备过滤配置白名单清洗（/api/devices PUT ← /filters 页保存） ----
// roi_regions: [{x,y,w,h} 归一化 0..1]，逐夹紧、非法项丢弃
static json sanitizeRoiRegions(const json & in) {
    json out = json::array();
    if (!in.is_array()) return out;
    auto clamp01 = [](double v) { return v < 0 ? 0.0 : (v > 1.0 ? 1.0 : v); };
    for (const auto & r : in) {
        if (!r.is_object()) continue;
        double x = 0, y = 0, w = 0, h = 0;
        auto num = [&r](const char * k, double & v) {
            if (!r.contains(k) || !r[k].is_number()) return false;
            v = r[k].get<double>();
            return true;
        };
        if (!num("x", x) || !num("y", y) || !num("w", w) || !num("h", h)) continue;
        x = clamp01(x); y = clamp01(y); w = clamp01(w); h = clamp01(h);
        if (w <= 0 || h <= 0) continue;
        out.push_back({{"x", x}, {"y", y}, {"w", w}, {"h", h}});
    }
    return out;
}

// roi_polygons: [[{x,y},{x,y},...] 归一化 0..1]，外层层数与每块顶点数均有上限
// （区域是每帧解析的，上限防手改 json 把设备线程拖慢）。上限与 parseRoiPolys 保持一致。
static json sanitizeRoiPolys(const json & in) {
    json out = json::array();
    if (!in.is_array()) return out;
    auto clamp01 = [](double v) { return v < 0 ? 0.0 : (v > 1.0 ? 1.0 : v); };
    for (const auto & raw : in) {
        if (out.size() >= 16) break;
        if (!raw.is_array()) continue;
        json pts = json::array();
        for (const auto & pt : raw) {
            if (pts.size() >= 64) break;
            if (!pt.is_object() || !pt.contains("x") || !pt.contains("y")) continue;
            if (!pt["x"].is_number() || !pt["y"].is_number()) continue;
            double x = clamp01(pt["x"].get<double>());
            double y = clamp01(pt["y"].get<double>());
            if (!pts.empty()) {   // 相邻重复点：射线法判定会退化
                const json & last = pts.back();
                if (last["x"].get<double>() == x && last["y"].get<double>() == y) continue;
            }
            pts.push_back({{"x", x}, {"y", y}});
        }
        if (pts.size() > 1) {     // 首尾同点 = 显式闭合，运行侧隐式闭合已覆盖
            const json & f = pts.front();
            const json & b = pts.back();
            if (f["x"].get<double>() == b["x"].get<double>() &&
                f["y"].get<double>() == b["y"].get<double>()) pts.erase(pts.size() - 1);
        }
        if (pts.size() < 3) continue;
        double area2 = 0;         // 鞋带公式：共线/面积极小的退化多边形丢弃
        size_t n = pts.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++)
            area2 += pts[j]["x"].get<double>() * pts[i]["y"].get<double>() -
                     pts[i]["x"].get<double>() * pts[j]["y"].get<double>();
        if (area2 < 0) area2 = -area2;
        if (area2 < 1e-6) continue;
        out.push_back(pts);
    }
    return out;
}

// orientation_filter: Front/Side/Back 子集，其余丢弃
static json sanitizeOrientationFilter(const json & in) {
    json out = json::array();
    if (!in.is_array()) return out;
    for (const auto & a : in) {
        if (a.is_string()) {
            std::string s = a.get<std::string>();
            if (s == "Front" || s == "Side" || s == "Back") out.push_back(s);
        }
    }
    return out;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool safe_relpath(const std::string & s) {
    return !s.empty() && s.find("..") == std::string::npos && s[0] != '/';
}

// 调试捕获开启时，在页面导航最后一个链接后追加 /debug 入口
//（导航 html 在 web_vllm_cpp/templates 共享，不在本项目源码里改）
static void injectDebugNavLink(std::string & html) {
    size_t nav = html.find("</nav>");
    if (nav == std::string::npos) return;
    size_t a = html.rfind("</a>", nav);
    if (a == std::string::npos) return;
    a += 4;
    html.insert(a, "\n            <a href=\"/debug\" class=\"nav-link\">调试捕获</a>");
}

// 「区域与过滤」为 rknn 特有页（框选 ROI/朝向过滤/低质量阈值），常驻注入导航
static void injectFiltersNavLink(std::string & html) {
    size_t nav = html.find("</nav>");
    if (nav == std::string::npos) return;
    size_t a = html.rfind("</a>", nav);
    if (a == std::string::npos) return;
    a += 4;
    html.insert(a, "\n            <a href=\"/filters\" class=\"nav-link\">区域与过滤</a>");
}

// 与 app.py test_rtsp_connection 对齐：主码流 + URL 编码，能取到 1 帧即 online
static std::string test_rtsp_connection(const json & device) {
    std::string port = device.value("port", std::string("554"));
    if (port.empty()) port = "554";
    std::string url = build_rtsp_url(device.value("username", ""), device.value("password", ""),
                                     device.value("ip", ""), port,
                                     device.value("main_stream_path", std::string("/ch01")));
    return grab_frame_jpeg(url, 5).empty() ? "offline" : "online";
}

// ==================== Server ====================

Server::Server(const Config & cfg, Monitor * monitor) : cfg_(cfg), monitor_(monitor) {
    // flask-cors 全局允许跨域
    svr_.set_default_headers({{"Access-Control-Allow-Origin", "*"}});
    setup_routes();   // 并发任务队列（ThreadPool）在 setup_routes 里配置
}

Server::~Server() { stop(); }

void Server::stop() {
    hb_running_ = false;
    if (hb_.joinable()) hb_.join();
}

void Server::run() {
    hb_running_ = true;
    hb_ = std::thread([this] { heartbeat_loop(); });
    printf(" * Running on http://0.0.0.0:%d\n", cfg_.port);
    svr_.listen("0.0.0.0", cfg_.port);
}

// 对应 app.py device_heartbeat：每 5 秒检测所有设备，状态变更才回写
void Server::heartbeat_loop() {
    while (hb_running_) {
        for (int i = 0; i < 50 && hb_running_; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!hb_running_) break;
        try {
            json devices = load_json_array(g_paths->devices_file);
            // 逐台 RTSP 探测最长 5s×N 台，这期间 /filters、PUT /api/devices 都可能改文件。
            // 曾按「读整份→改→整体写回」做，把并发写入的 roi_* 等字段整片覆盖回旧值
            // （实测把刚存的全图矩形「复活」，区域形同失效）。故这里只暂存 status，
            // 写前重读文件、按 id 只合并 status，别的字段一概不碰。
            std::map<std::string, std::string> new_status;   // device id -> status
            for (auto & device : devices) {
                if (!device.is_object()) continue;
                std::string old_status = device.value("status", std::string("offline"));
                std::string probe = test_rtsp_connection(device);
                if (old_status != probe) {
                    new_status[device.value("id", "")] = probe;
                    printf("📡 设备 '%s' 状态变更: %s -> %s\n",
                           device.value("name", "").c_str(), old_status.c_str(), probe.c_str());
                }
            }
            if (new_status.empty()) continue;

            json fresh = load_json_array(g_paths->devices_file);
            bool updated = false;
            for (auto & device : fresh) {
                if (!device.is_object()) continue;
                auto it = new_status.find(device.value("id", ""));
                if (it == new_status.end() || device.value("status", "") == it->second)
                    continue;                       // 设备已被删除或状态又被别处改了，不越权
                device["status"] = it->second;
                updated = true;
            }
            if (updated) save_json_atomic(g_paths->devices_file, fresh);
        } catch (const std::exception & e) {
            printf("心跳检测异常: %s\n", e.what());
        }
    }
}

// URL_PREFIX 非空时把 HTML/JS 里的绝对路径统一加上前缀。token 表来自模板/页面
// 实际字面量清单（fetch/href/src 一律引号起头）；只替换紧跟引号符的路径，
// 文本里的裸路径不会命中——正好不想动正文。
void Server::applyPrefix(std::string &body) const {
    if (cfg_.url_prefix.empty()) return;
    static const char *roots[] = {"/api/", "/data/", "/static/", "/debug", "/filters",
                                  "/devices", "/rules", "/alerts"};
    auto replace_all = [](std::string &s, const std::string &from, const std::string &to) {
        for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size())
            s.replace(pos, from.size(), to);
    };
    for (char q : {'"', '\'', '`'}) {
        for (const char *root : roots) {
            std::string from(1, q); from += root;
            std::string to(1, q); to += cfg_.url_prefix; to += root;
            replace_all(body, from, to);
        }
    }
    replace_all(body, "href=\"/\"", "href=\"" + cfg_.url_prefix + "/\"");
}

void Server::setup_routes() {
    // 反向代理前缀（URL_PREFIX，如 /x-search）：在路由分发前剥掉，路由表不感知前缀。
    // 本版 httplib 的 pre-routing 回调给的是 const Request&，但底层对象本身非 const，
    // const_cast 改写 path 良定义。不带前缀的原路径照常兼容（代理已剥 / 本机脚本沿用）。
    if (!cfg_.url_prefix.empty()) {
        svr_.set_pre_routing_handler([p = cfg_.url_prefix](const httplib::Request &req,
                                                           httplib::Response &) {
            auto &r = const_cast<httplib::Request &>(req);
            if (r.path == p) r.path = "/";
            else if (r.path.size() > p.size() && r.path.compare(0, p.size(), p) == 0 &&
                     r.path[p.size()] == '/')
                r.path = r.path.substr(p.size());
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    // cpp-httplib 默认单线程串行处理：任一慢 handler（如 PUT /api/tasks 等
    // matcher 锁、读大 JSON）会冻结全部端点，前端表现为整个平台卡住。换 4 线程
    // 任务队列后慢请求只堵自己。共享状态均有各自锁（monitor/matcher/json_store）。
    svr_.new_task_queue = [] { return new httplib::ThreadPool(8); };

    // flask-cors 会自动应答 OPTIONS 预检；httplib 需显式兜底（同源前端不触发，跨源调用会）
    svr_.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });

    // ==================== 页面路由 ====================
    auto page = [this](const std::string & name) -> std::function<void(const httplib::Request &, httplib::Response &)> {
        return [this, name](const httplib::Request &, httplib::Response & res) {
            std::string html = read_file(cfg_.static_dir + "/" + name + ".html");
            if (html.empty()) { res.status = 404; res.set_content("Not Found", "text/plain"); return; }
            // 调试捕获开启时给公共导航注入 /debug 入口（模板与 web_vllm_cpp 共用，源码不动）
            if (cfg_.debug_capture && name != "debug") injectDebugNavLink(html);
            // debug/filters 是 rknn 特有页、自带完整导航，不重复注入
            if (name != "debug" && name != "filters") injectFiltersNavLink(html);
            applyPrefix(html);   // 导航注入之后再重写绝对路径，注入的链接一并带上前缀
            res.set_content(html, "text/html; charset=utf-8");
        };
    };
    svr_.Get("/", page("index"));
    svr_.Get("/devices", page("devices"));
    svr_.Get("/rules", page("rules"));
    svr_.Get("/alerts", page("alerts"));
    svr_.Get("/debug", page("debug"));
    svr_.Get("/filters", page("filters"));

    // 静态资源：/static/<path> -> static_dir/static/<path>
    svr_.Get("/static/(.*)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string rel = req.matches[1];
        if (!safe_relpath(rel)) { res.status = 404; return; }
        std::string body = read_file(cfg_.static_dir + "/static/" + rel);
        if (body.empty()) { res.status = 404; return; }
        std::string mime = "application/octet-stream";
        if (rel.size() > 4 && rel.compare(rel.size() - 4, 4, ".css") == 0) mime = "text/css";
        else if (rel.size() > 3 && rel.compare(rel.size() - 3, 3, ".js") == 0) mime = "application/javascript";
        else if (rel.size() > 4 && rel.compare(rel.size() - 4, 4, ".html") == 0) mime = "text/html; charset=utf-8";
        if (mime != "application/octet-stream") applyPrefix(body);   // 文本资源同样重写；二进制跳过
        res.set_content(body, mime.c_str());
    });

    // ==================== 设备管理 ====================
    svr_.Get("/api/devices", [](const httplib::Request &, httplib::Response & res) {
        res_json(res, load_json_array(g_paths->devices_file));
    });

    svr_.Post("/api/devices", [](const httplib::Request & req, httplib::Response & res) {
        json data = parse_body(req);
        json device;
        device["id"] = uuid4();
        device["name"] = body_str(data, "name");
        device["ip"] = body_str(data, "ip");
        device["port"] = body_str(data, "port", "554");
        device["username"] = body_str(data, "username");
        device["password"] = body_str(data, "password");
        device["main_stream_path"] = body_str(data, "main_stream_path", "/ch01");
        device["sub_stream_path"] = body_str(data, "sub_stream_path", "/ch01_sub");
        device["stream_type"] = body_str(data, "stream_type", "main");
        device["status"] = "offline";
        device["created_at"] = now_isoformat();

        printf("📹 设备: %s\n🔗 测试RTSP: rtsp://%s:%s@%s:%s%s\n",
               device["name"].get<std::string>().c_str(),
               device["username"].get<std::string>().c_str(),
               device["password"].get<std::string>().c_str(),
               device["ip"].get<std::string>().c_str(),
               device["port"].get<std::string>().c_str(),
               device["main_stream_path"].get<std::string>().c_str());
        device["status"] = test_rtsp_connection(device);
        printf("✅ 设备状态: %s\n", device["status"].get<std::string>().c_str());

        json devices = load_json_array(g_paths->devices_file);
        devices.push_back(device);
        save_json_atomic(g_paths->devices_file, devices);
        res_json(res, device);
    });

    svr_.Put("/api/devices/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        json data = parse_body(req);
        std::string id = req.matches[1];
        json devices = load_json_array(g_paths->devices_file);
        for (auto & d : devices) {
            if (d.is_object() && d.value("id", "") == id) {
                d["name"] = body_str(data, "name", d.value("name", ""));
                d["ip"] = body_str(data, "ip", d.value("ip", ""));
                d["port"] = body_str(data, "port", d.value("port", "554"));
                d["username"] = body_str(data, "username", d.value("username", ""));
                d["password"] = body_str(data, "password", d.value("password", ""));
                d["main_stream_path"] = body_str(data, "main_stream_path", d.value("main_stream_path", "/ch01"));
                d["sub_stream_path"] = body_str(data, "sub_stream_path", d.value("sub_stream_path", "/ch01_sub"));
                d["stream_type"] = body_str(data, "stream_type", d.value("stream_type", "main"));
                // 按设备过滤配置（/filters 页保存）：缺字段保持原值
                if (data.contains("roi_regions"))
                    d["roi_regions"] = sanitizeRoiRegions(data["roi_regions"]);
                if (data.contains("roi_polygons"))
                    d["roi_polygons"] = sanitizeRoiPolys(data["roi_polygons"]);
                if (data.contains("orientation_filter"))
                    d["orientation_filter"] = sanitizeOrientationFilter(data["orientation_filter"]);
                for (const char * key : {"min_box_w", "min_box_h"}) {
                    if (data.contains(key)) {
                        double v = body_num(data, key, -1);
                        if (v >= 0 && v <= 4096) d[key] = (int)v;
                    }
                }
                break;
            }
        }
        save_json_atomic(g_paths->devices_file, devices);
        monitor_->refreshRulesAsync();   // 设备配置热更新走同一条后台刷新链
        res_json(res, {{"success", true}});
    });

    svr_.Delete("/api/devices/([^/]+)", [](const httplib::Request & req, httplib::Response & res) {
        std::string id = req.matches[1];
        json devices = load_json_array(g_paths->devices_file);
        json kept = json::array();
        for (auto & d : devices)
            if (!(d.is_object() && d.value("id", "") == id)) kept.push_back(d);
        save_json_atomic(g_paths->devices_file, kept);
        res_json(res, {{"success", true}});
    });

    // ==================== 任务管理 ====================
    // （规则随任务走 /api/tasks；独立的 /api/rules CRUD 已删——前端与板端流程均不再使用）
    svr_.Get("/api/tasks", [](const httplib::Request &, httplib::Response & res) {
        res_json(res, load_json_array(g_paths->tasks_file));
    });

    svr_.Post("/api/tasks", [this](const httplib::Request & req, httplib::Response & res) {
        json data = parse_body(req);
        json task;
        task["id"] = data.contains("id") && data["id"].is_string() ? data["id"] : json(uuid4());
        task["name"] = body_str(data, "name");
        task["device_id"] = body_str(data, "device_id");
        task["device_name"] = body_str(data, "device_name");
        task["device"] = body_val(data, "device", json::object());
        task["rules"] = body_val(data, "rules", json::array());
        task["created_at"] = now_isoformat();

        json tasks = load_json_array(g_paths->tasks_file);
        tasks.push_back(task);
        save_json_atomic(g_paths->tasks_file, tasks);
        monitor_->refreshRulesAsync();
        res_json(res, task);
    });

    svr_.Get("/api/tasks/([^/]+)", [](const httplib::Request & req, httplib::Response & res) {
        std::string id = req.matches[1];
        for (auto & t : load_json_array(g_paths->tasks_file)) {
            if (t.is_object() && t.value("id", "") == id) { res_json(res, t); return; }
        }
        res_json(res, {{"error", "Task not found"}}, 404);
    });

    svr_.Put("/api/tasks/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        json data = parse_body(req);
        std::string id = req.matches[1];
        json tasks = load_json_array(g_paths->tasks_file);
        for (auto & t : tasks) {
            if (t.is_object() && t.value("id", "") == id) {
                t["name"] = body_str(data, "name", t.value("name", ""));
                t["device_id"] = body_str(data, "device_id", t.value("device_id", ""));
                t["device_name"] = body_str(data, "device_name", t.value("device_name", ""));
                t["device"] = data.contains("device") ? data["device"]
                                                      : t.value("device", json::object());
                if (data.contains("rules")) t["rules"] = data["rules"];
                break;
            }
        }
        save_json_atomic(g_paths->tasks_file, tasks);
        monitor_->refreshRulesAsync();   // 后台刷新：不同步等 match() 放 matcher 锁
        res_json(res, {{"success", true}});
    });

    svr_.Delete("/api/tasks/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string id = req.matches[1];
        json tasks = load_json_array(g_paths->tasks_file);
        json kept = json::array();
        for (auto & t : tasks)
            if (!(t.is_object() && t.value("id", "") == id)) kept.push_back(t);
        save_json_atomic(g_paths->tasks_file, kept);
        monitor_->refreshRulesAsync();   // 此前删任务不刷新，规则残留到下次重启
        res_json(res, {{"success", true}});
    });

    // ==================== 视频流 ====================
    svr_.Get("/api/video/frame/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string device_id = req.matches[1];
        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();

        // ① 1 秒内的缓存直接回；② 失败退避期内只回旧帧；③ 同设备已有线程在跑 ffmpeg
        //    时不再排队拉取、直接回旧帧——慢/离线设备的 10fps 轮询不再吃光 HTTP 线程池
        std::vector<uint8_t> stale;
        bool do_grab = false;
        {
            std::lock_guard<std::mutex> lk(fc_mu_);
            auto it = frame_cache_.find(device_id);
            if (it != frame_cache_.end()) {
                if (now - it->second.time_ms < 1000) {
                    res.set_content((const char *)it->second.jpeg.data(), it->second.jpeg.size(), "image/jpeg");
                    return;
                }
                stale = it->second.jpeg;
            }
            auto bo = fc_backoff_until.find(device_id);
            if (bo == fc_backoff_until.end() || now >= bo->second)
                do_grab = fc_grabbing_.insert(device_id).second;
        }
        if (!do_grab) {
            if (stale.empty()) stale = placeholder_jpeg();
            res.set_content((const char *)stale.data(), stale.size(), "image/jpeg");
            return;
        }

        // 兜底：中途任何异常/提前返回都要归还在飞标记，否则该设备预览永久卡旧帧
        struct GrabGuard {
            Server * s; std::string id;
            ~GrabGuard() { std::lock_guard<std::mutex> lk(s->fc_mu_); s->fc_grabbing_.erase(id); }
        } guard{this, device_id};

        std::vector<uint8_t> jpeg;
        for (auto & d : load_json_array(g_paths->devices_file)) {
            if (d.is_object() && d.value("id", "") == device_id) {
                // 预览用未编码 URL（与 app.py 一致）；预览超时收敛到 5s，
                // 离线设备最坏只占一个线程 5s，退避期再挡住后续请求
                jpeg = grab_frame_jpeg(device_rtsp_url_raw(d), 5);
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lk(fc_mu_);
            if (jpeg.empty()) {
                frame_cache_[device_id] = {now, placeholder_jpeg()};
                fc_backoff_until[device_id] = now + 5000;   // 失败 5s 退避
            } else {
                frame_cache_[device_id] = {now, jpeg};
                fc_backoff_until.erase(device_id);
            }
        }
        if (jpeg.empty()) jpeg = placeholder_jpeg();
        res.set_content((const char *)jpeg.data(), jpeg.size(), "image/jpeg");
    });

    // ==================== 告警 / 日志 ====================
    // 告警按天分文件存储；所有接口支持 ?date=YYYY-MM-DD（缺省 = 今天）
    auto alerts_file = [](const httplib::Request & req, httplib::Response & res) -> std::string {
        std::string date = req.get_param_value("date");
        if (!date.empty() && !valid_alert_date(date)) {
            res.status = 400;
            return "";
        }
        return g_paths->alerts_file_for(date);
    };

    svr_.Get("/api/alerts/dates", [](const httplib::Request &, httplib::Response & res) {
        // 有告警文件的日期（新→旧），供前端日期下拉框用；今天永远排第一
        json dates = json::array();
        std::vector<std::string> found;
        std::error_code ec;
        for (auto & e : fs::directory_iterator(g_paths->alerts_history_dir, ec)) {
            if (!e.is_regular_file()) continue;
            std::string name = e.path().filename().string();
            const std::string prefix = "alerts_", suffix = ".json";
            if (name.size() > prefix.size() + suffix.size() &&
                name.rfind(prefix, 0) == 0 &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                std::string d = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
                if (valid_alert_date(d)) found.push_back(d);
            }
        }
        std::sort(found.begin(), found.end(), std::greater<std::string>());
        dates.push_back(today_str());
        for (auto & d : found)
            if (d != today_str()) dates.push_back(d);
        res_json(res, dates);
    });

    svr_.Get("/api/alerts", [alerts_file](const httplib::Request & req, httplib::Response & res) {
        std::string file = alerts_file(req, res);
        if (file.empty() && res.status != 200) return;
        res_json(res, load_json_array(file));
    });

    svr_.Post("/api/alerts/clear", [alerts_file](const httplib::Request & req, httplib::Response & res) {
        std::string file = alerts_file(req, res);
        if (file.empty() && res.status != 200) return;
        save_json_atomic(file, json::array());
        res_json(res, {{"success", true}});
    });

    svr_.Delete("/api/alerts/([^/]+)", [alerts_file](const httplib::Request & req, httplib::Response & res) {
        std::string file = alerts_file(req, res);
        if (file.empty() && res.status != 200) return;
        std::string id = req.matches[1];
        json kept = json::array();
        for (auto & a : load_json_array(file))
            if (!(a.is_object() && a.value("id", "") == id)) kept.push_back(a);
        save_json_atomic(file, kept);
        res_json(res, {{"success", true}});
    });

    // ==================== 调试捕获（VLM_DEBUG_CAPTURE）====================
    svr_.Get("/api/debug", [](const httplib::Request & req, httplib::Response & res) {
        json logs = load_json_array(g_paths->debug_log_file);
        // ?limit=N 截断（默认全部，最新在前），前端分页可选
        std::string limit_s = req.get_param_value("limit");
        if (!limit_s.empty()) {
            try {
                size_t limit = (size_t)std::stoul(limit_s);
                if (logs.is_array() && logs.size() > limit)
                    logs.get_ref<json::array_t &>().resize(limit);
            } catch (...) {}
        }
        res_json(res, logs);
    });

    svr_.Post("/api/debug/clear", [](const httplib::Request &, httplib::Response & res) {
        // 清空日志并尽量清掉调试图片目录（板端磁盘友好）
        std::error_code ec;
        if (fs::exists(g_paths->debug_dir, ec)) fs::remove_all(g_paths->debug_dir, ec);
        fs::create_directories(g_paths->debug_dir, ec);
        save_json_atomic(g_paths->debug_log_file, json::array());
        res_json(res, {{"success", true}});
    });

    // 调试图片（与告警图同款 1 小时缓存）
    svr_.Get("/data/debug_images/(.*)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string rel = req.matches[1];
        if (!safe_relpath(rel)) { res.status = 404; return; }
        std::string body = read_file(g_paths->debug_dir + "/" + rel);
        if (body.empty()) { res.status = 404; return; }
        res.set_header("Cache-Control", "public, max-age=3600");
        res.set_content(body, "image/jpeg");
    });

    // 告警图片（1 小时浏览器缓存，对齐 app.py serve_alert_image）
    svr_.Get("/data/alerts_images/(.*)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string rel = req.matches[1];
        if (!safe_relpath(rel)) { res.status = 404; return; }
        std::string body = read_file(g_paths->alerts_dir + "/" + rel);
        if (body.empty()) { res.status = 404; return; }
        res.set_header("Cache-Control", "public, max-age=3600");
        res.set_content(body, "image/jpeg");
    });

    // ==================== 监控状态 ====================
    svr_.Get("/api/monitoring/status", [this](const httplib::Request &, httplib::Response & res) {
        std::string model_name = getenv_str("VLM_MODEL_NAME", "");
        std::string mmproj = getenv_str("VLM_MMPROJ_PATH", "");
        auto rstrip = [](std::string s) {
            while (!s.empty() && (s.back() == '/' || s.back() == '\\')) s.pop_back();
            return s;
        };
        json model;
        model["type"] = "RKNN/RKLLM";   // 本项目唯一后端（GGUF 路线已删）
        model["match_mode"] = tolower_ascii(getenv_str("VLM_MATCH_MODE", "combined"));
        model["model_name"] = model_name;
        model["model_basename"] = model_name.empty() ? ""
                            : fs::path(rstrip(model_name)).filename().string();
        model["mmproj_path"] = mmproj;
        model["mmproj_basename"] = mmproj.empty() ? "" : fs::path(mmproj).filename().string();

        json out;
        out["running"] = monitor_->isRunning();
        out["device_count"] = monitor_->deviceCount();
        out["debug_capture"] = cfg_.debug_capture;
        out["model"] = model;
        res_json(res, out);
    });

    svr_.Post("/api/monitoring/start", [this](const httplib::Request &, httplib::Response & res) {
        res_json(res, monitor_->start());
    });

    svr_.Post("/api/monitoring/stop", [this](const httplib::Request &, httplib::Response & res) {
        res_json(res, monitor_->stop());
    });
}
