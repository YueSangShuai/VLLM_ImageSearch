#include "server.h"
#include "frame.h"   // url_encode
#include "util.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

void res_json(httplib::Response &res, const json &j, int status = 200) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
    res.set_header("Access-Control-Allow-Origin", "*");
}

std::string mime_of(const std::string &name) {
    auto dot = name.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "png") return "image/png";
    if (ext == "webp") return "image/webp";
    return "image/jpeg";
}

// 图库相对路径安全性：禁绝对路径与 ".." 段
bool safe_rel(const std::string &rel) {
    if (rel.empty() || rel[0] == '/') return false;
    if (rel.find("..") != std::string::npos) return false;
    if (rel.find('\\') != std::string::npos) return false;
    return true;
}

int64_t now_epoch() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string sanitize_filename(std::string s) {
    for (auto &c : s)
        if (!(isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-')) c = '_';
    while (s.find("__") != std::string::npos) s.replace(s.find("__"), 2, "_");
    if (s.size() > 120) s = s.substr(s.size() - 120);
    return s;
}

}  // namespace

void Server::run() {
    setup_routes();
    fprintf(stderr, "web_text2img 监听 0.0.0.0:%d（gallery=%s）\n", cfg_.port,
            cfg_.gallery_dir.c_str());
    svr_.listen("0.0.0.0", cfg_.port);
}

void Server::setup_routes() {
    svr_.new_task_queue = [] { return new httplib::ThreadPool(4); };

    // 反向代理前缀（URL_PREFIX）：路由分发前剥掉，路由表不感知前缀
    if (!cfg_.url_prefix.empty()) {
        svr_.set_pre_routing_handler([p = cfg_.url_prefix](const httplib::Request &req,
                                                           httplib::Response &) {
            auto &r = const_cast<httplib::Request &>(req);
            if (r.path == p) { r.path = "/"; return httplib::Server::HandlerResponse::Unhandled; }
            if (r.path.size() > p.size() && r.path.compare(0, p.size(), p) == 0 &&
                r.path[p.size()] == '/') {
                r.path = r.path.substr(p.size());
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    svr_.Options(".*", [](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });

    const std::string &gallery_dir = cfg_.gallery_dir;

    // ---------- 页面 ----------
    svr_.Get("/", [this](const httplib::Request &, httplib::Response &res) {
        std::ifstream f(static_path_for("search.html"));
        if (!f) { res_json(res, {{"error", "search.html 缺失"}}, 404); return; }
        std::stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/html; charset=utf-8");
    });
    svr_.Get("/static/(.*)", [this](const httplib::Request &req, httplib::Response &res) {
        std::string name = req.matches[1];
        if (!safe_rel(name)) { res.status = 400; return; }
        std::ifstream f(static_path_for(name), std::ios::binary);
        if (!f) { res.status = 404; return; }
        std::stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/plain; charset=utf-8");
    });

    // ---------- 结果缩略图回读：/img?file=<相对路径> ----------
    svr_.Get("/img", [&](const httplib::Request &req, httplib::Response &res) {
        std::string rel = req.get_param_value("file");
        if (!safe_rel(rel)) { res.status = 400; return; }
        fs::path p = fs::path(gallery_dir) / rel;
        std::error_code ec;
        if (!fs::exists(p, ec)) { res.status = 404; return; }
        std::ifstream f(p, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), mime_of(rel).c_str());
        res.set_header("Cache-Control", "max-age=86400");
    });

    // ---------- 检索 ----------
    svr_.Post("/api/search", [&](const httplib::Request &req, httplib::Response &res) {
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res_json(res, {{"error", "body 需为 JSON"}}, 400);
            return;
        }
        std::string text = body.value("text", "");
        int topk = body.value("topk", cfg_.default_topk);
        if (topk < 1) topk = 1;
        if (topk > 200) topk = 200;
        if (trim(text).empty()) { res_json(res, {{"error", "text 必填"}}, 400); return; }
        try {
            auto t0 = std::chrono::steady_clock::now();
            auto hits = svc_.search(text, topk);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
            json results = json::array();
            for (auto &[file, score] : hits)
                results.push_back({{"file", file},
                                   {"score", std::round((double)score * 10000.0) / 10000.0},
                                   {"url", "/img?file=" + url_encode(file)}});
            res_json(res, {{"success", true}, {"text", text}, {"results", results},
                           {"elapsed_ms", ms}, {"gallery_size", svc_.gallery_size()}});
        } catch (const std::exception &e) {
            res_json(res, {{"error", e.what()}}, 500);
        }
    });

    // ---------- 上传入库（multipart 字段 image，兼容 file） ----------
    svr_.Post("/api/enroll", [&](const httplib::Request &req, httplib::Response &res) {
        httplib::MultipartFormData f;
        bool found = false;
        if (req.has_file("image")) { f = req.get_file_value("image"); found = true; }
        else if (req.has_file("file")) { f = req.get_file_value("file"); found = true; }
        if (!found) { res_json(res, {{"error", "multipart 字段 image 缺失"}}, 400); return; }
        if (f.content.empty() || f.content.size() > 20 * 1024 * 1024) {
            res_json(res, {{"error", "内容为空或超过 20MB"}}, 400);
            return;
        }
        std::string name = sanitize_filename(f.filename.empty() ? "upload.jpg" : f.filename);
        std::random_device rd;
        auto uniq = std::to_string(now_epoch()) + "_" + std::to_string(rd() % 10000) + "_" + name;
        fs::path up = fs::path(gallery_dir) / "uploads";
        std::error_code ec;
        fs::create_directories(up, ec);
        fs::path dst = up / uniq;
        {
            std::ofstream o(dst, std::ios::binary);
            o.write(f.content.data(), (std::streamsize)f.content.size());
        }
        try {
            svc_.enroll(dst.string());
            std::string rel = (fs::path("uploads") / uniq).string();
            res_json(res, {{"success", true}, {"file", rel}, {"url", "/img?file=" + url_encode(rel)},
                           {"gallery_size", svc_.gallery_size()}});
        } catch (const std::exception &e) {
            fs::remove(dst, ec);
            res_json(res, {{"error", e.what()}}, 500);
        }
    });

    // ---------- 手动触发重建（新图批量放入目录后调用） ----------
    svr_.Post("/api/rebuild", [&](const httplib::Request &, httplib::Response &res) {
        if (svc_.building()) {
            res_json(res, {{"success", false}, {"reason", "正在建库中"}}, 409);
            return;
        }
        std::thread([&svc = svc_] {
            try { svc.build(); } catch (const std::exception &e) {
                fprintf(stderr, "rebuild 失败: %s\n", e.what());
            }
        }).detach();
        res_json(res, {{"success", true}, {"gallery_size", svc_.gallery_size()}});
    });

    // ---------- 状态 ----------
    svr_.Get("/api/status", [&](const httplib::Request &, httplib::Response &res) {
        res_json(res, {{"building", svc_.building()},
                       {"gallery_size", svc_.gallery_size()},
                       {"gallery_total", svc_.gallery_total()},
                       {"gallery_dir", cfg_.gallery_dir},
                       {"default_topk", cfg_.default_topk},
                       {"port", cfg_.port}});
    });
    svr_.Get("/api/system_info", [&](const httplib::Request &, httplib::Response &res) {
        res_json(res, {{"service", "web_text2img_rknn"}, {"gallery_size", svc_.gallery_size()}});
    });
}

std::string Server::static_path_for(const std::string &name) const {
    return cfg_.static_dir + "/" + name;
}
