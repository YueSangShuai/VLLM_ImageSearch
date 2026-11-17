#pragma once
// HTTP 服务：cpp-httplib，路由为 web_vllm_rknn server.cpp 的检索特化裁剪版
//（保留 URL_PREFIX 剥前缀、ThreadPool、res_json、CORS 兜底模式）。
#include "config.h"
#include "search_service.h"
#include <httplib.h>
#include <memory>

class Server {
public:
    Server(const Config &cfg, SearchService &svc) : cfg_(cfg), svc_(svc) {}
    void run();

private:
    void setup_routes();
    std::string static_path_for(const std::string &name) const;

    Config cfg_;
    SearchService &svc_;
    httplib::Server svr_;
};
