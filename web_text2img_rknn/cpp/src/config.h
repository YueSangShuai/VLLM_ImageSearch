#pragma once
// 全局配置：环境变量驱动，命名沿用 web_vllm_* 系列（run_web_rk.sh 注入）。
#include <string>

std::string getenv_str(const char *name, const std::string &fallback = "");
int64_t getenv_int(const char *name, int64_t fallback);
double getenv_double(const char *name, double fallback);
bool getenv_bool(const char *name, bool fallback);

struct Config {
    std::string exe_dir;        // 二进制所在目录
    std::string root;           // 项目根（output/ 上一级）
    std::string data_dir;       // 向量缓存/manifest 所在（data/）
    std::string static_dir;
    std::string models_dir;
    int port = 5005;
    std::string url_prefix;     // 反向代理前缀（可空）

    // MARS 双塔（RKNN, rk3588）
    std::string image_rknn;
    std::string text_rknn;
    std::string vocab_path;     // bert-base-uncased vocab.txt
    int text_max_len = 128;     // 与训练/导出一致（purn_model 切 128）

    // 图库
    std::string gallery_dir;    // 被搜索目录（递归 jpg/png）
    int default_topk = 10;

    static Config load();
};
