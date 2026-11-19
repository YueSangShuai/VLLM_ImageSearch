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
    std::string debug_dir;        // debug_images/（VLM_DEBUG_CAPTURE 开启时使用）
    std::string debug_log_file;   // debug_log.json（旧版数组格式，升级后只读兼容、不再写入）
    std::string debug_log_jsonl;  // debug_log.jsonl（现格式：一行一条 JSON，追加写）

    static Paths from_data_dir(const std::string &data_dir);
    std::string today_alerts_file() const;               // alerts_history/alerts_YYYY-MM-DD.json
    std::string alerts_file_for(const std::string &date) const;  // date 空 = 今天
};

// YYYY-MM-DD 严格校验（防目录穿越，date 会拼进文件名）
bool valid_alert_date(const std::string &date);

// 全局文件锁（对应 _log_lock）与共享路径（main 启动时设置）
extern std::mutex g_io_lock;
extern Paths *g_paths;

// 调试捕获日志上限（main 从 config 设置）；0=不限：JSONL 追加写、不逐出、图片不删
extern int g_debug_log_max;

json load_json_array(const std::string &path);          // 缺文件/损坏返回 []
// 调试日志合并读取：jsonl（新→旧）+ 旧 debug_log.json 数组，统一「最新在前」。
// 内部不加锁（文件读取本身原子；append 在 g_io_lock 下整文件重写）
json load_debug_logs();
void save_json_atomic(const std::string &path, const json &v, int indent = 2);
void append_detection_log(const json &entry);           // 头插，截断 1000
void append_debug_log(const json &entry);               // 默认追加 jsonl 不限条数；上限>0 时头插截断并删被逐条目图片
void prepend_alert(const Paths &p, const json &alert);  // 与 MonitoringManager.handle_alert 一致
