#pragma once
// JSON 文件存储：路径与行为逐一对齐 processor.py / app.py
#include <nlohmann/json.hpp>
#include <string>
#include <mutex>
#include <vector>

using json = nlohmann::json;

struct Paths {
    std::string data_dir;
    std::string devices_file, rules_file, tasks_file, log_file;
    std::string alerts_dir;  // alerts_images/
    std::string alerts_history_dir;

    static Paths from_data_dir(const std::string &data_dir);
    std::string today_alerts_file() const;               // alerts_history/alerts_YYYY-MM-DD.json
    std::string alerts_file_for(const std::string &date) const;  // date 空 = 今天
};

// YYYY-MM-DD 严格校验（防目录穿越，date 会拼进文件名）
bool valid_alert_date(const std::string &date);

// 全局文件锁（对应 _log_lock）与共享路径（main 启动时设置）
extern std::mutex g_io_lock;
extern Paths *g_paths;

json load_json_array(const std::string &path);          // 缺文件/损坏返回 []
void save_json_atomic(const std::string &path, const json &v, int indent = 2);
void append_detection_log(const json &entry);           // 头插，截断 1000
void prepend_alert(const Paths &p, const json &alert);  // 与 MonitoringManager.handle_alert 一致
