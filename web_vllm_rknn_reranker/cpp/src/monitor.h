#pragma once
// 对应 processor.py MonitoringManager（含共享的 VLM/BGE 单例，即 python 的 _shared_* 机制）
#include "detector.h"
#include "json_store.h"
#include "matcher.h"
#include "reranker_scorer.h"
#include "stream_processor.h"
#include "text_embedder.h"
#include "vlm_describer.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Monitor {
public:
    explicit Monitor(const Config & cfg);   // 加载模型（YOLO；VLM/BGE 见 startDevice 惰性说明）
    ~Monitor();

    json start();            // {"success":true,"message":"已启动 N 个设备监控"}
    json refreshRules();     // 重新读 tasks.json 并刷新各设备规则
    json stop();

    bool isRunning() const;
    size_t deviceCount() const;

    void startDevice(const std::string & device_id);
    void stopDevice(const std::string & device_id);

private:
    void handleAlert(const json & alert);   // 对应 handle_alert：prepend_alert

    Config cfg_;
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<VlmDescriber> describer_;    // reranker 模式下不加载
    std::unique_ptr<TextEmbedder> embedder_;     // reranker 模式下不加载
    std::unique_ptr<RerankerScorer> reranker_;   // 仅 match_mode=reranker

    json devices_ = json::array();          // devices.json 快照（DeviceManager.load_devices）
    json tasks_ = json::array();

    struct Runner {
        std::unique_ptr<Matcher> matcher;
        std::unique_ptr<StreamProcessor> proc;
        std::thread th;
    };
    std::map<std::string, std::unique_ptr<Runner>> processors_;  // 保持设备顺序（map 按 id 排序，与 python dict 顺序可能不同，仅影响 message 无关字段）

    mutable std::mutex mu_;
};
