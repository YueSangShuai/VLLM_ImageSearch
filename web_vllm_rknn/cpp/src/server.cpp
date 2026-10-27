#include "server.h"
#include "frame.h"
#include "stream_processor.h"
#include "util.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
    setup_routes();
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
            bool updated = false;
            for (auto & device : devices) {
                if (!device.is_object()) continue;
                std::string old_status = device.value("status", std::string("offline"));
                std::string new_status = test_rtsp_connection(device);
                if (old_status != new_status) {
                    device["status"] = new_status;
                    updated = true;
                    printf("📡 设备 '%s' 状态变更: %s -> %s\n",
                           device.value("name", "").c_str(), old_status.c_str(), new_status.c_str());
                }
            }
            if (updated) save_json_atomic(g_paths->devices_file, devices);
        } catch (const std::exception & e) {
            printf("心跳检测异常: %s\n", e.what());
        }
    }
}

void Server::setup_routes() {
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
            res.set_content(html, "text/html; charset=utf-8");
        };
    };
    svr_.Get("/", page("index"));
    svr_.Get("/devices", page("devices"));
    svr_.Get("/rules", page("rules"));
    svr_.Get("/alerts", page("alerts"));

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

    svr_.Put("/api/devices/([^/]+)", [](const httplib::Request & req, httplib::Response & res) {
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
                break;
            }
        }
        save_json_atomic(g_paths->devices_file, devices);
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

    // ==================== 规则管理 ====================
    svr_.Get("/api/rules", [](const httplib::Request &, httplib::Response & res) {
        res_json(res, load_json_array(g_paths->rules_file));
    });

    svr_.Post("/api/rules", [this](const httplib::Request & req, httplib::Response & res) {
        json data = parse_body(req);
        json rule;
        rule["id"] = uuid4();
        rule["name"] = body_str(data, "name");
        rule["text"] = body_str(data, "text");
        rule["positive"] = body_val(data, "positive", json::array());
        rule["negative"] = body_val(data, "negative", json::array());
        rule["threshold"] = body_num(data, "threshold", 0.5);
        rule["enabled"] = body_val(data, "enabled", json(true));
        rule["created_at"] = now_isoformat();

        json rules = load_json_array(g_paths->rules_file);
        rules.push_back(rule);
        save_json_atomic(g_paths->rules_file, rules);

        // 同步追加进所有 tasks.json 任务（与 Python 版行为一致）
        if (fs::exists(g_paths->tasks_file)) {
            json tasks = load_json_array(g_paths->tasks_file);
            for (auto & task : tasks) {
                if (!task.contains("rules") || !task["rules"].is_array())
                    task["rules"] = json::array();
                json tr;
                tr["id"] = rule["id"];
                tr["name"] = rule["name"];
                tr["text"] = rule.value("text", "");
                tr["positive"] = rule.value("positive", json::array());
                tr["negative"] = rule.value("negative", json::array());
                tr["threshold"] = rule.value("threshold", 0.5);
                tr["score_diff"] = 0;
                tr["enabled"] = rule.value("enabled", json(true));
                task["rules"].push_back(tr);
            }
            save_json_atomic(g_paths->tasks_file, tasks);
        }
        monitor_->refreshRules();
        res_json(res, rule);
    });

    svr_.Put("/api/rules/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        json data = parse_body(req);
        std::string id = req.matches[1];
        json rules = load_json_array(g_paths->rules_file);
        for (auto & r : rules) {
            if (r.is_object() && r.value("id", "") == id) {
                r["name"] = body_str(data, "name", r.value("name", ""));
                r["text"] = body_str(data, "text", r.value("text", ""));
                r["positive"] = data.contains("positive") ? data["positive"]
                                                          : r.value("positive", json::array());
                r["negative"] = data.contains("negative") ? data["negative"] : r.value("negative", json::array());
                r["threshold"] = data.contains("threshold") ? body_num(data, "threshold", 0.5)
                                                            : r.value("threshold", 0.5);
                r["enabled"] = data.contains("enabled") ? body_val(data, "enabled", json(true))
                                                        : r.value("enabled", json(true));
                break;
            }
        }
        save_json_atomic(g_paths->rules_file, rules);

        // 同步更新 tasks.json 中所有引用
        if (fs::exists(g_paths->tasks_file)) {
            json tasks = load_json_array(g_paths->tasks_file);
            for (auto & task : tasks) {
                if (!task.contains("rules") || !task["rules"].is_array()) continue;
                for (auto & r : task["rules"]) {
                    if (r.is_object() && r.value("id", "") == id) {
                        r["name"] = body_str(data, "name", r.value("name", ""));
                        r["text"] = body_str(data, "text", r.value("text", ""));
                        r["positive"] = data.contains("positive") ? data["positive"]
                                                                  : r.value("positive", json::array());
                        r["negative"] = data.contains("negative") ? data["negative"]
                                                                  : r.value("negative", json::array());
                        r["threshold"] = data.contains("threshold") ? body_num(data, "threshold", 0.5)
                                                                    : r.value("threshold", 0.5);
                        r["score_diff"] = data.contains("score_diff") ? body_num(data, "score_diff", 0)
                                                                      : r.value("score_diff", 0);
                        r["enabled"] = data.contains("enabled") ? body_val(data, "enabled", json(true))
                                                                : r.value("enabled", json(true));
                    }
                }
            }
            save_json_atomic(g_paths->tasks_file, tasks);
        }
        monitor_->refreshRules();
        res_json(res, {{"success", true}});
    });

    svr_.Delete("/api/rules/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string id = req.matches[1];
        json rules = load_json_array(g_paths->rules_file);
        json kept = json::array();
        for (auto & r : rules)
            if (!(r.is_object() && r.value("id", "") == id)) kept.push_back(r);
        save_json_atomic(g_paths->rules_file, kept);
        monitor_->refreshRules();
        res_json(res, {{"success", true}});
    });

    // ==================== 任务管理 ====================
    svr_.Get("/api/tasks", [](const httplib::Request &, httplib::Response & res) {
        res_json(res, load_json_array(g_paths->tasks_file));
    });

    svr_.Post("/api/tasks", [](const httplib::Request & req, httplib::Response & res) {
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
        monitor_->refreshRules();
        res_json(res, {{"success", true}});
    });

    svr_.Delete("/api/tasks/([^/]+)", [](const httplib::Request & req, httplib::Response & res) {
        std::string id = req.matches[1];
        json tasks = load_json_array(g_paths->tasks_file);
        json kept = json::array();
        for (auto & t : tasks)
            if (!(t.is_object() && t.value("id", "") == id)) kept.push_back(t);
        save_json_atomic(g_paths->tasks_file, kept);
        res_json(res, {{"success", true}});
    });

    // ==================== 视频流 ====================
    svr_.Get("/api/video/frame/([^/]+)", [this](const httplib::Request & req, httplib::Response & res) {
        std::string device_id = req.matches[1];
        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();
        {
            std::lock_guard<std::mutex> lk(fc_mu_);
            auto it = frame_cache_.find(device_id);
            if (it != frame_cache_.end() && now - it->second.time_ms < 1000) {
                res.set_content((const char *)it->second.jpeg.data(), it->second.jpeg.size(), "image/jpeg");
                return;
            }
        }
        std::vector<uint8_t> jpeg;
        for (auto & d : load_json_array(g_paths->devices_file)) {
            if (d.is_object() && d.value("id", "") == device_id) {
                // 预览用未编码 URL（与 app.py 一致）
                jpeg = grab_frame_jpeg(device_rtsp_url_raw(d), 10);
                break;
            }
        }
        if (jpeg.empty()) jpeg = placeholder_jpeg();
        {
            std::lock_guard<std::mutex> lk(fc_mu_);
            frame_cache_[device_id] = {now, jpeg};
        }
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

    svr_.Get("/api/logs", [](const httplib::Request &, httplib::Response & res) {
        res_json(res, load_json_array(g_paths->log_file));
    });

    svr_.Post("/api/logs/clear", [](const httplib::Request &, httplib::Response & res) {
        save_json_atomic(g_paths->log_file, json::array());
        res_json(res, {{"success", true}});
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
        std::string fmt = tolower_ascii(getenv_str("VLM_MODEL_FORMAT", ""));
        std::string mmproj = getenv_str("VLM_MMPROJ_PATH", "");
        auto rstrip = [](std::string s) {
            while (!s.empty() && (s.back() == '/' || s.back() == '\\')) s.pop_back();
            return s;
        };
        json model;
        bool is_gguf = fmt == "gguf" ||
            (model_name.size() > 5 && model_name.size() >= 5 &&
             tolower_ascii(model_name).rfind(".gguf") == model_name.size() - 5);
        model["type"] = is_gguf ? "GGUF" : "HuggingFace";
        model["match_mode"] = tolower_ascii(getenv_str("VLM_MATCH_MODE", "combined"));
        model["model_name"] = model_name;
        model["model_basename"] = model_name.empty() ? ""
                            : fs::path(rstrip(model_name)).filename().string();
        model["mmproj_path"] = mmproj;
        model["mmproj_basename"] = mmproj.empty() ? "" : fs::path(mmproj).filename().string();

        json out;
        out["running"] = monitor_->isRunning();
        out["device_count"] = monitor_->deviceCount();
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
