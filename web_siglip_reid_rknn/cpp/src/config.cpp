#include "config.h"

#include <cstdlib>
#include <climits>
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

static std::string detect_exe_dir() {  // 避免与成员 Config::exe_dir 同名冲突
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path().string();
}

Config Config::load() {
    Config c;
    c.exe_dir = detect_exe_dir();

    namespace fs = std::filesystem;
    // 项目根目录：output_* 的上一级，可用 WEB_VLLM_ROOT 覆盖（env 名沿用原框架）
    std::string root = getenv_str("WEB_VLLM_ROOT",
        fs::path(c.exe_dir).parent_path().string());
    c.data_dir = getenv_str("WEB_VLLM_DATA_DIR", root + "/data");
    c.static_dir = getenv_str("WEB_VLLM_STATIC_DIR", c.exe_dir + "/static");
    c.port = (int)getenv_int("WEB_VLLM_PORT", 5004);

    std::string models_dir = getenv_str("WEB_VLLM_MODELS_DIR", root + "/models");
    if (models_dir.empty()) models_dir = c.exe_dir + "/models";

    // SigLIP2-reid 双塔：视觉塔 RKNN（NPU），文本塔 ONNX + Gemma sp（板端 CPU）
    c.siglip_vision_rknn = getenv_str("SIGLIP_VISION_RKNN",
        models_dir + "/siglip_vision_fp16.rknn");
    c.siglip_text_onnx = getenv_str("SIGLIP_TEXT_ONNX",
        models_dir + "/onnx/siglip_text_int8.onnx");   // int8 动态量化：269MB（fp32 1077MB，
    // 板端 available 仅 ~2.8GB 放不下；规则向量 golden cos≥0.992，0.15 分值尺度下漂移可忽略）
    c.sp_model_path = getenv_str("SIGLIP_SP_MODEL",
        models_dir + "/tokenizer.model");
    c.text_max_length = (int)getenv_int("SIGLIP_TEXT_MAX_LEN", 64);

    // 规则未写 threshold 时的回退（SigLIP 尺度：正 ~0.15-0.22 / 负 ~0）
    c.default_threshold = getenv_double("SIGLIP_DEFAULT_THRESHOLD", 0.15);

    // 调试捕获开关（web /debug 页）；默认关闭
    c.debug_capture = getenv_bool("DEBUG_CAPTURE", false);
    c.yolo_onnx_path = getenv_str("DETECTOR_ONNX_PATH", models_dir + "/yolov8n_640_rk3588.rknn");

    return c;
}
