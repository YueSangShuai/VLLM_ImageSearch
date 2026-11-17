#include "config.h"

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

std::string getenv_str(const char *name, const std::string &fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    return v;
}

int64_t getenv_int(const char *name, int64_t fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    try { return std::stoll(v); } catch (...) { return fallback; }
}

double getenv_double(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}

bool getenv_bool(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    std::string s(v);
    for (auto &c : s) c = (char)tolower(c);
    return !(s == "0" || s == "false" || s == "no");
}

Config Config::load() {
    Config c;
    {
        char buf[4096];
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            c.exe_dir = std::filesystem::path(buf).parent_path().string();
        } else {
            c.exe_dir = ".";
        }
    }
    namespace fs = std::filesystem;
    c.root = getenv_str("WEB_VLLM_ROOT", fs::path(c.exe_dir).parent_path().string());
    c.data_dir = getenv_str("WEB_VLLM_DATA_DIR", c.root + "/data");
    c.static_dir = getenv_str("WEB_VLLM_STATIC_DIR", c.exe_dir + "/static");
    c.port = (int)getenv_int("WEB_VLLM_PORT", 5005);
    c.url_prefix = getenv_str("URL_PREFIX", "");

    c.models_dir = getenv_str("WEB_VLLM_MODELS_DIR", c.root + "/models");
    c.image_rknn = getenv_str("MARS_IMAGE_RKNN", c.models_dir + "/mars_image_fp16.rknn");
    c.text_rknn = getenv_str("MARS_TEXT_RKNN", c.models_dir + "/mars_text_fp16.rknn");
    c.vocab_path = getenv_str("MARS_VOCAB", c.models_dir + "/vocab.txt");
    c.text_max_len = (int)getenv_int("MARS_TEXT_MAX_LEN", 128);

    c.gallery_dir = getenv_str("GALLERY_DIR", c.root + "/gallery");
    c.default_topk = (int)getenv_int("DEFAULT_TOPK", 10);
    return c;
}
