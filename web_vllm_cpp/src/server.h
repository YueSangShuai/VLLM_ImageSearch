#pragma once
// 对应 app.py：cpp-httplib 实现全部 REST 路由/页面路由 + 设备心跳线程
#include "config.h"
#include "json_store.h"
#include "monitor.h"

#include "httplib.h"

#include <atomic>
#include <map>
#include <mutex>
#include <set>
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

    // 视频帧缓存（1 秒）。ffmpeg 单帧拉取最坏 5s：为防 10fps 轮询把慢/离线设备的
    // 拉取堆满 HTTP 线程池（整站假死），同一设备并发请求只放行一个线程真拉帧
    // （fc_grabbing_ 在飞去重，其余请求直接回旧帧），失败后 5s 退避期内也只回缓存帧。
    struct FrameCache { int64_t time_ms = 0; std::vector<uint8_t> jpeg; };
    std::mutex fc_mu_;
    std::map<std::string, FrameCache> frame_cache_;
    std::set<std::string> fc_grabbing_;              // 正在 ffmpeg 取帧的设备
    std::map<std::string, int64_t> fc_backoff_until;  // 取帧失败退避截止时间戳(ms)

    Config cfg_;
    Monitor * monitor_;
    httplib::Server svr_;
    std::atomic<bool> hb_running_{false};
    std::thread hb_;
};
