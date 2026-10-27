#include "json_store.h"
#include "util.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

std::mutex g_io_lock;
Paths *g_paths = nullptr;

Paths Paths::from_data_dir(const std::string &dir) {
    Paths p;
    p.data_dir = dir;
    p.devices_file = dir + "/devices.json";
    p.rules_file = dir + "/rules.json";
    p.tasks_file = dir + "/tasks.json";
    p.log_file = dir + "/detection_log.json";
    p.alerts_dir = dir + "/alerts_images";
    p.alerts_history_dir = dir + "/alerts_history";
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

void prepend_alert(const Paths &p, const json &alert) {
    std::lock_guard<std::mutex> lk(g_io_lock);
    std::string alerts_file = p.today_alerts_file();
    json alerts = load_json_array(alerts_file);
    alerts.insert(alerts.begin(), alert);
    save_json_atomic(alerts_file, alerts);
}
