#pragma once
// 对应 processor.py MonitoringManager（含共享的 VLM/BGE 单例，即 python 的 _shared_* 机制）
#include "detector.h"
#include "json_store.h"
#include "matcher.h"
#include "reranker.h"
#include "stream_processor.h"
#include "text_embedder.h"
#include "vlm_describer.h"

#include <atomic>
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
    json refreshRules();     // 重新读 tasks.json 并刷新各设备规则（同步，会等 match 放锁）
    void refreshRulesAsync();// 后台线程刷新；已有刷新在跑时合并请求（HTTP 线程零等待）
    json stop();

    bool isRunning() const;
    size_t deviceCount() const;

    void startDevice(const std::string & device_id);
    void stopDevice(const std::string & device_id);

    // persistent 拉流心跳（online 判定改用帧流）：返回该设备距上一帧的毫秒数；
    // -1 = 该设备没有在跑常驻拉流（未监控/oneshot/还没出过帧）→ 心跳循环回退 RTSP 探测
    int64_t frameHeartbeatAge(const std::string & device_id);

private:
    void handleAlert(const json & alert);   // 对应 handle_alert：prepend_alert

    Config cfg_;
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<VlmDescriber> describer_;
    std::unique_ptr<TextEmbedder> embedder_;
    // 二次核验全局单例（与 describer_/embedder_ 并列）。RERANKER_MODEL_PATH 空=不构造
    std::unique_ptr<Reranker> reranker_;

    json devices_ = json::array();          // devices.json 快照（DeviceManager.load_devices）
    json tasks_ = json::array();

    // shared_ptr：refreshRules 在锁外取 runner->matcher/proc 做更新时，
    // 即便并发 stopDevice 已从 map 摘除，引用计数也保证其存活到本轮刷完
    struct Runner {
        std::unique_ptr<Matcher> matcher;
        std::unique_ptr<StreamProcessor> proc;
        std::thread th;
    };
    std::map<std::string, std::shared_ptr<Runner>> processors_;  // 保持设备顺序（map 按 id 排序，与 python dict 顺序可能不同，仅影响 message 无关字段）

    mutable std::mutex mu_;
    std::atomic<bool> refreshing_{false};
    std::atomic<bool> refresh_again_{false};   // 刷新期间又有请求 → 跑完当前这轮再刷一轮
};
