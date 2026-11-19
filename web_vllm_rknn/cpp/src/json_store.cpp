#include "json_store.h"
#include "util.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

std::mutex g_io_lock;
Paths *g_paths = nullptr;
int g_debug_log_max = 0;  // 0=不限条数

Paths Paths::from_data_dir(const std::string &dir) {
    Paths p;
    p.data_dir = dir;
    p.devices_file = dir + "/devices.json";
    p.rules_file = dir + "/rules.json";
    p.tasks_file = dir + "/tasks.json";
    p.log_file = dir + "/detection_log.json";
    p.alerts_dir = dir + "/alerts_images";
    p.alerts_history_dir = dir + "/alerts_history";
    p.debug_dir = dir + "/debug_images";
    p.debug_log_file = dir + "/debug_log.json";
    p.debug_log_jsonl = dir + "/debug_log.jsonl";
    fs::create_directories(p.alerts_dir);
    return p;
}

bool valid_alert_date(const std::string &date) {
    if (date.size() != 10) return false;
    for (size_t i = 0; i < 10; i++) {
        char c = date[i];
        if (i == 4 || i == 7) { if (c != '-') return false; }
        else if (c < '0' || c > '9') return false;
    }
    return true;
}

std::string Paths::alerts_file_for(const std::string &date) const {
    fs::create_directories(alerts_history_dir);
    return alerts_history_dir + "/alerts_" + (date.empty() ? today_str() : date) + ".json";
}

std::string Paths::today_alerts_file() const {
    return alerts_file_for("");
}

json load_json_array(const std::string &path) {
    std::ifstream in(path);
    if (!in) return json::array();
    json j = json::parse(in, nullptr, false);
    if (j.is_discarded() || !j.is_array()) return json::array();
    return j;
}

void save_json_atomic(const std::string &path, const json &v, int indent) {
    // dump 时 ensure_ascii=false 对齐 json.dump(ensure_ascii=False, indent=2)
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        out << v.dump(indent);
    }
    fs::rename(tmp, path);  // os.replace 等价（同目录原子）
}

void append_detection_log(const json &entry) {
    std::lock_guard<std::mutex> lk(g_io_lock);
    const std::string &log_file = g_paths->log_file;
    json logs = load_json_array(log_file);
    logs.insert(logs.begin(), entry);
    if (logs.size() > 1000) logs.get_ref<json::array_t&>().resize(1000);
    save_json_atomic(log_file, logs);
}

// 调试日志合并读取，统一「最新在前」：
//   1) 新格式 debug_log.jsonl（一行一条 JSON，追加序=旧→新，读出后反转）
//   2) 旧格式 debug_log.json（升级前遗留的数组，头插=新→旧），拼在 jsonl 之后
// 单行损坏（断电截断等）跳过。调用方无锁：jsonl 行级追加 + rename 原子重写，读侧不互斥
json load_debug_logs() {
    json out = json::array();
    {
        std::ifstream in(g_paths->debug_log_jsonl);
        std::string line;
        json arr = json::array();
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            json e = json::parse(line, nullptr, false);
            if (!e.is_discarded()) arr.push_back(std::move(e));
        }
        for (size_t i = arr.size(); i > 0; i--) out.push_back(std::move(arr[i - 1]));
    }
    json legacy = load_json_array(g_paths->debug_log_file);
    for (auto & e : legacy) out.push_back(std::move(e));
    return out;
}

// 调试捕获日志。默认（g_debug_log_max=0）：追加进 jsonl，不限条数、不删图——
// 全量保留，页面翻页翻看。上限>0 时回退旧策略：头插截断、被逐条目图片一并删除。
// 上限模式下顺带把 jsonl+旧 json 压实进 jsonl 并删除遗留 json（只发生一次）。
void append_debug_log(const json & entry) {
    std::lock_guard<std::mutex> lk(g_io_lock);
    if (g_debug_log_max <= 0) {
        std::ofstream out(g_paths->debug_log_jsonl, std::ios::app);
        out << entry.dump(-1, ' ', false) << '\n';  // ensure_ascii=false：中文原样，行更短
        return;
    }
    json logs = load_debug_logs();
    logs.insert(logs.begin(), entry);
    auto & arr = logs.get_ref<json::array_t &>();
    while (arr.size() > (size_t)g_debug_log_max) {
        const json & old = arr.back();
        for (const char * key : {"image", "original_image"}) {
            if (old.contains(key) && old[key].is_string()) {
                std::string rel = old[key].get<std::string>();
                if (!rel.empty() && rel.find("..") == std::string::npos && rel[0] != '/')
                    fs::remove(g_paths->debug_dir + "/" + rel);
            }
        }
        arr.pop_back();
    }
    // 压实写回 jsonl（一行一条），旧 json 已并入、删除防重复计数
    {
        std::ofstream out(g_paths->debug_log_jsonl + ".tmp", std::ios::trunc);
        for (auto it = arr.rbegin(); it != arr.rend(); ++it)  // jsonl 顺序=旧→新
            out << it->dump(-1, ' ', false) << '\n';
    }
    fs::rename(g_paths->debug_log_jsonl + ".tmp", g_paths->debug_log_jsonl);
    std::error_code ec;
    fs::remove(g_paths->debug_log_file, ec);
}

void prepend_alert(const Paths &p, const json &alert) {
    std::lock_guard<std::mutex> lk(g_io_lock);
    std::string alerts_file = p.today_alerts_file();
    json alerts = load_json_array(alerts_file);
    alerts.insert(alerts.begin(), alert);
    save_json_atomic(alerts_file, alerts);
}
