#pragma once
// 对应 app.py：cpp-httplib 实现全部 REST 路由/页面路由 + 设备心跳线程
#include "config.h"
#include "json_store.h"
#include "monitor.h"

#include "httplib.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Server {
public:
    Server(const Config & cfg, Monitor * monitor);
    ~Server();

    void run();    // 阻塞监听（内部先启动心跳线程）
    void stop();

private:
    void setup_routes();
    void heartbeat_loop();

    // 视频帧缓存（1 秒）
    struct FrameCache { int64_t time_ms = 0; std::vector<uint8_t> jpeg; };
    std::mutex fc_mu_;
    std::map<std::string, FrameCache> frame_cache_;

    Config cfg_;
    Monitor * monitor_;
    httplib::Server svr_;
    std::atomic<bool> hb_running_{false};
    std::thread hb_;
};
