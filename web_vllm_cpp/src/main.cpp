// 启动序列对齐 app.py：加载模型 -> 自动启动监控 -> HTTP 服务常驻
#include "config.h"
#include "json_store.h"
#include "monitor.h"
#include "server.h"

#include <cstdio>
#include <filesystem>
#include <memory>

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    Config cfg = Config::load();

    g_paths = new Paths(Paths::from_data_dir(cfg.data_dir));

    printf("🚀 预加载Qwen-VL + BGE模型...\n");
    auto monitor = std::make_unique<Monitor>(cfg);
    printf("✅ Qwen-VL + BGE模型加载完成\n");

    printf("🚀 自动启动监控...\n");
    json start_res = monitor->start();
    printf("✅ 监控已启动: %s\n", start_res.value("message", "").c_str());

    Server server(cfg, monitor.get());
    server.run();   // 阻塞；心跳线程由 Server 内部管理

    server.stop();
    monitor->stop();
    return 0;
}
