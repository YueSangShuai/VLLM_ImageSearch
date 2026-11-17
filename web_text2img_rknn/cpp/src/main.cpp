// 入口：加载双塔 → 后台建库 → HTTP 服务（边建边查）。
#include "config.h"
#include "search_service.h"
#include "server.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>

int main() {
    try {
        Config cfg = Config::load();
        std::filesystem::create_directories(cfg.gallery_dir);
        std::filesystem::create_directories(cfg.data_dir + "/gallery");

        SearchService svc(cfg.image_rknn, cfg.text_rknn, cfg.vocab_path, cfg.text_max_len,
                          cfg.gallery_dir, cfg.data_dir + "/gallery");

        // 首次/增量建库放后台：10 万张全量 ~2h，不能挡服务
        std::thread([&svc] {
            try { svc.build(); } catch (const std::exception &e) {
                fprintf(stderr, "启动建库失败: %s\n", e.what());
            }
        }).detach();

        Server server(cfg, svc);
        server.run();
    } catch (const std::exception &e) {
        fprintf(stderr, "启动失败: %s\n", e.what());
        return 1;
    }
    return 0;
}
