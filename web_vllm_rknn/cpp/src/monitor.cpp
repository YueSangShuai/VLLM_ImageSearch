#include "monitor.h"
#include "util.h"

#include <cstdio>

// MonitoringManager：设备来自 devices.json，规则来自 tasks.json（按 device_id 合并）
// VLM/BGE 全局单例（python 用类属性 _shared_vlm_describer/_shared_text_embedder 实现，
// 这里等价地由 Monitor 持有；Detector 也是全设备共享一份，对应 python 的 self.detector）

Monitor::Monitor(const Config & cfg) : cfg_(cfg) {
    devices_ = load_json_array(g_paths->devices_file);

    printf("加载YOLO模型: %s\n", cfg.yolo_onnx_path.c_str());
    detector_.reset(new Detector(cfg.yolo_onnx_path, cfg.det_conf, cfg.det_iou));

    printf("加载Qwen-VL描述模型: %s (模式: %s)\n", cfg.vlm_model_path.c_str(), cfg.match_mode.c_str());
    describer_.reset(new VlmDescriber(cfg.vlm_model_path, cfg.mmproj_path, cfg.n_ctx,
                                      cfg.vlm_temperature, cfg.vlm_repetition_penalty));
    describer_->setAttrHead(cfg.attr_head_path);   // 可选件：加载失败自动走旧路径

    printf("加载BGE文本向量模型: %s\n", cfg.bge_onnx_path.c_str());
    embedder_.reset(new TextEmbedder(cfg.bge_onnx_path, cfg.bge_vocab_path, cfg.bge_max_length));

    // 二次核验（可选件）：路径为空=整个功能禁用；init 失败 Reranker 自带降级（ok()=false，核验一律放行）
    if (!cfg.reranker_model_path.empty()) {
        printf("加载Qwen3-Reranker二次核验模型: %s\n", cfg.reranker_model_path.c_str());
        reranker_.reset(new Reranker(cfg.reranker_model_path, cfg.reranker_default_threshold));
    } else {
        printf("二次核验未启用（RERANKER_MODEL_PATH 未设置）\n");
    }
}

Monitor::~Monitor() {
    stop();
}

bool Monitor::isRunning() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !processors_.empty();
}

size_t Monitor::deviceCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    return processors_.size();
}

void Monitor::handleAlert(const json & alert) {
    prepend_alert(*g_paths, alert);   // 头插 + 原子写（json_store 内部加锁）
}

void Monitor::startDevice(const std::string & device_id) {
    // 快照读：设备与规则在 mu_ 内取，建模/BGE 编码挪到锁外（updateRules 首次要
    // 编码全部规则文本，秒级，压在 mu_ 里会拖住 status/refresh 等全部接口）
    json device = json::object();
    json device_rules = json::array();
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (processors_.count(device_id)) return;
        bool found = false;
        for (const auto & d : devices_) {
            if (d.is_object() && d.value("id", "") == device_id) { device = d; found = true; break; }
        }
        if (!found) return;

        // 该设备规则 = 所有关联任务规则的并集（python 还会给 rule 补 device_id 字段）
        for (auto & task : tasks_) {
            if (task.is_object() && task.value("device_id", "") == device_id) {
                for (auto & rule : task.value("rules", json::array())) {
                    rule["device_id"] = device_id;
                    device_rules.push_back(rule);
                }
            }
        }
    }

    std::string name = device.value("name", "?");
    printf("[%s] 加载了 %zu 条规则\n", name.c_str(), device_rules.size());

    auto runner = std::make_shared<Runner>();
    runner->matcher = std::make_unique<Matcher>(describer_.get(), embedder_.get(), reranker_.get(), cfg_);
    runner->matcher->updateRules(device_rules);
    runner->proc = std::make_unique<StreamProcessor>(device, detector_.get(), runner->matcher.get(),
                                                     cfg_.debug_capture);
    if (cfg_.debug_capture) printf("[%s] 🐞 调试捕获已开启（web /debug 页）\n", name.c_str());
    StreamProcessor * proc = runner->proc.get();
    proc->on_alert = [this](const json & a) { handleAlert(a); };
    StreamProcessor * raw = proc;
    runner->th = std::thread([raw]() { raw->run(); });

    // 并发重复 start 兜底：后到者自毁，不让两个线程抢同一设备
    bool inserted = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!processors_.count(device_id)) { processors_[device_id] = runner; inserted = true; }
    }
    if (!inserted) {
        runner->proc->stop();
        if (runner->th.joinable()) runner->th.join();
    }
}

int64_t Monitor::frameHeartbeatAge(const std::string & device_id) {
    std::shared_ptr<Runner> runner;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = processors_.find(device_id);
        if (it == processors_.end()) return -1;
        runner = it->second;   // 引用计数保活，同 refreshRules 的并发 stopDevice 约定
    }
    if (!runner->proc->pullActive()) return -1;
    return runner->proc->frameAgeMs();
}

void Monitor::stopDevice(const std::string & device_id) {
    std::shared_ptr<Runner> runner;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = processors_.find(device_id);
        if (it == processors_.end()) return;
        runner = it->second;
        processors_.erase(it);
    }
    runner->proc->stop();
    // oneshot：取帧最多 ~10s 后退出；persistent：帧槽 close 即醒 + 读流线程 200ms 内
    // 退出并杀 ffmpeg，run() 末尾 join 读流线程，这一 join 把两个线程都收齐
    if (runner->th.joinable()) runner->th.join();
}

json Monitor::start() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        tasks_ = load_json_array(g_paths->tasks_file);
        devices_ = load_json_array(g_paths->devices_file);   // 运行中新增/编辑的设备即刻可见
    }
    // 与 python 一致：遍历启动时加载的设备列表
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto & d : devices_)
            if (d.is_object() && d.contains("id")) ids.push_back(d["id"].get<std::string>());
    }
    for (const auto & id : ids) startDevice(id);

    json res;
    res["success"] = true;
    res["message"] = "已启动 " + std::to_string(deviceCount()) + " 个设备监控";
    return res;
}

json Monitor::refreshRules() {
    // 锁内只读文件+快照 runner 引用；BGE 编码（updateRules 对新文本）与逐设备
    // 更新全部在 mu_ 锁外——否则刷新期间 /api/monitoring/status、start/stop 排队
    json tasks, devices;
    std::vector<std::pair<std::string, std::shared_ptr<Runner>>> runners;
    {
        std::lock_guard<std::mutex> lk(mu_);
        tasks = load_json_array(g_paths->tasks_file);
        tasks_ = tasks;
        // 顺带刷新设备快照：PUT /api/devices 保存后走同一条异步链，
        // ROI 区域/朝向过滤/低质量阈值等按设备配置无需重启即可生效
        devices = load_json_array(g_paths->devices_file);
        devices_ = devices;
        runners.reserve(processors_.size());
        for (auto & kv : processors_) runners.emplace_back(kv.first, kv.second);
    }
    printf("🔄 刷新监控规则...\n");
    size_t total = 0;
    for (auto & kv : runners) {
        const std::string & device_id = kv.first;
        Runner * runner = kv.second.get();   // shared_ptr 拷贝保活，并发 stopDevice 安全
        for (const auto & d : devices) {
            if (d.is_object() && d.value("id", "") == device_id) {
                runner->proc->updateDevice(d);
                break;
            }
        }
        json device_rules = json::array();
        for (auto & task : tasks) {
            if (task.is_object() && task.value("device_id", "") == device_id) {
                for (auto & rule : task.value("rules", json::array())) {
                    rule["device_id"] = device_id;
                    device_rules.push_back(rule);
                }
            }
        }
        runner->matcher->updateRules(device_rules);
        printf("  设备 %s 更新了 %zu 条规则:\n", runner->proc->deviceInfo().value("name", "?").c_str(),
               device_rules.size());
        for (const auto & rule : device_rules)
            printf("    - %s\n", rule.value("name", "未知").c_str());
        total += device_rules.size();
    }
    printf("✅ 规则刷新完成，共 %zu 条规则\n", total);

    json res;
    res["success"] = true;
    res["message"] = "已刷新 " + std::to_string(total) + " 条规则";
    return res;
}

void Monitor::refreshRulesAsync() {
    // HTTP 线程零等待：后台线程执行 refreshRules（其中 matcher 重编码可能要几秒）。
    // 已有刷新在跑时只置重刷标记——refreshRules 每次都重读最新 tasks.json，
    // 补一轮即可覆盖期间的新保存，不会丢更新也不会并发重入。
    bool expect = false;
    if (!refreshing_.compare_exchange_strong(expect, true)) {
        refresh_again_ = true;
        return;
    }
    std::thread([this] {
        do {
            refresh_again_ = false;
            try {
                refreshRules();
            } catch (const std::exception & e) {
                printf("规则刷新失败: %s\n", e.what());
            }
        } while (refresh_again_);
        refreshing_ = false;
    }).detach();
}

json Monitor::stop() {
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto & kv : processors_) ids.push_back(kv.first);
    }
    for (const auto & id : ids) stopDevice(id);

    json res;
    res["success"] = true;
    res["message"] = "已停止所有监控";
    return res;
}
