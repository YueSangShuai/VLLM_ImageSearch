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

    std::string device = cfg.device;
    if (device.rfind("cuda", 0) == 0) {
        // 无 GPU 环境自动降级为 CPU（python 按 torch.cuda.is_available 判断）
        // C++ 侧：llama 后端枚举里找不到 CUDA 设备时，common_init_from_params 会自然回退，
        // 这里只需保证 n_gpu_layers 语义在 VlmDescriber 内部一致。
    }

    printf("加载Qwen-VL描述模型: %s (模式: %s)\n", cfg.vlm_model_path.c_str(), cfg.match_mode.c_str());
    describer_.reset(new VlmDescriber(cfg.vlm_model_path, cfg.mmproj_path, device,
                                      cfg.n_gpu_layers, cfg.n_ctx,
                                      cfg.vlm_temperature, cfg.vlm_repetition_penalty));

    printf("加载BGE文本向量模型: %s\n", cfg.bge_onnx_path.c_str());
    embedder_.reset(new TextEmbedder(cfg.bge_onnx_path, cfg.bge_vocab_path, cfg.bge_max_length));
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
    std::lock_guard<std::mutex> lk(mu_);
    if (processors_.count(device_id)) return;

    json device = json::object();
    bool found = false;
    for (const auto & d : devices_) {
        if (d.is_object() && d.value("id", "") == device_id) { device = d; found = true; break; }
    }
    if (!found) return;

    // 该设备规则 = 所有关联任务规则的并集（python 还会给 rule 补 device_id 字段）
    json device_rules = json::array();
    for (auto & task : tasks_) {
        if (task.is_object() && task.value("device_id", "") == device_id) {
            for (auto & rule : task.value("rules", json::array())) {
                rule["device_id"] = device_id;
                device_rules.push_back(rule);
            }
        }
    }

    std::string name = device.value("name", "?");
    printf("[%s] 加载了 %zu 条规则\n", name.c_str(), device_rules.size());

    auto runner = std::make_unique<Runner>();
    runner->matcher = std::make_unique<Matcher>(describer_.get(), embedder_.get(), cfg_);
    runner->matcher->updateRules(device_rules);
    runner->proc = std::make_unique<StreamProcessor>(device, detector_.get(), runner->matcher.get());
    StreamProcessor * proc = runner->proc.get();
    proc->on_alert = [this](const json & a) { handleAlert(a); };
    StreamProcessor * raw = proc;
    runner->th = std::thread([raw]() { raw->run(); });

    processors_[device_id] = std::move(runner);
}

void Monitor::stopDevice(const std::string & device_id) {
    std::unique_ptr<Runner> runner;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = processors_.find(device_id);
        if (it == processors_.end()) return;
        runner = std::move(it->second);
        processors_.erase(it);
    }
    runner->proc->stop();
    if (runner->th.joinable()) runner->th.join();   // 取帧最多 ~10s 后退出
}

json Monitor::start() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        tasks_ = load_json_array(g_paths->tasks_file);
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
    json tasks;
    {
        std::lock_guard<std::mutex> lk(mu_);
        tasks = load_json_array(g_paths->tasks_file);
        tasks_ = tasks;
    }
    printf("🔄 刷新监控规则...\n");
    size_t total = 0;
    std::lock_guard<std::mutex> lk(mu_);
    for (auto & kv : processors_) {
        const std::string & device_id = kv.first;
        json device_rules = json::array();
        for (auto & task : tasks_) {
            if (task.is_object() && task.value("device_id", "") == device_id) {
                for (auto & rule : task.value("rules", json::array())) {
                    rule["device_id"] = device_id;
                    device_rules.push_back(rule);
                }
            }
        }
        kv.second->matcher->updateRules(device_rules);
        printf("  设备 %s 更新了 %zu 条规则:\n", kv.second->proc->deviceInfo().value("name", "?").c_str(),
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
