#include "config.h"

#include <cstdlib>
#include <climits>
#include <filesystem>
#include <regex>
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

std::string strip_image_placeholder(const std::string &prompt) {
    // 对齐 processor.py strip_image_placeholder：
    //   re.sub(r"[ \t]*<\s*image\s*>[ \t]*", "\n", prompt, IGNORECASE) 后 re.sub(r"[ \t]+", " ")
    static const std::regex img_re("[ \t]*<[ \t\r\n]*image[ \t\r\n]*>[ \t]*",
                                   std::regex::icase);
    std::string s = std::regex_replace(prompt, img_re, "\n");
    static const std::regex ws_re("[ \t]+");
    s = std::regex_replace(s, ws_re, " ");
    // strip()
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
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
    // 项目根目录：deploy/ 的上一级（web_vllm_cpp/），可用 WEB_VLLM_ROOT 覆盖
    std::string root = getenv_str("WEB_VLLM_ROOT",
        fs::path(c.exe_dir).parent_path().string());
    // 数据默认用本项目自带的 data/ 目录（不再依赖 Python 版 web_vllm/）
    c.data_dir = getenv_str("WEB_VLLM_DATA_DIR", root + "/data");
    c.static_dir = getenv_str("WEB_VLLM_STATIC_DIR", c.exe_dir + "/static");
    c.port = (int)getenv_int("WEB_VLLM_PORT", 5003);
    // URL_PREFIX：反向代理前缀（如 /x-search）。非空时后端接受带/不带前缀两种路径
    // （pre-routing 剥前缀），页面里的绝对 URL 统一重写为带前缀；off/空 = 关闭
    c.url_prefix = getenv_str("URL_PREFIX", "");
    if (c.url_prefix == "off") c.url_prefix.clear();
    while (!c.url_prefix.empty() && c.url_prefix.back() == '/') c.url_prefix.pop_back();
    if (!c.url_prefix.empty() && c.url_prefix.front() != '/') c.url_prefix = "/" + c.url_prefix;

    std::string models_dir = getenv_str("WEB_VLLM_MODELS_DIR", root + "/models");
    if (models_dir.empty()) models_dir = c.exe_dir + "/models";

    // VLM：同名环境变量复用，值换成 RKNN/RKLLM 产物（板上路径）
    // vlm_model_path -> 视觉塔 .rknn；mmproj_path -> 语言塔 .rkllm
    // 默认=未微调基座 Qwen3-VL-2B-Instruct（2026-10-29 起；实际运行以 run_web_rk.sh 导出为准）
    c.vlm_model_path = getenv_str("VLM_MODEL_NAME",
        "/home/pi/output_rk3588/models/zkteco-person-describe-ft-vl-2b-rk3588.rknn");
    c.mmproj_path = getenv_str("VLM_MMPROJ_PATH",
        "/home/pi/output_rk3588/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm");
    c.n_ctx = (int)getenv_int("VLM_GGUF_N_CTX", 4096);

    std::string mode = getenv_str("VLM_MATCH_MODE", "combined");
    for (auto &ch : mode) ch = (char)tolower(ch);
    if (mode != "description" && mode != "combined") {
        printf("未知的VLM_MATCH_MODE: %s，回退到 combined\n", mode.c_str());
        mode = "combined";
    }
    c.match_mode = mode;

    // 代码级默认与 processor.py DEFAULT_DESCRIPTION_PROMPT 一致（英文）；
    // 实际部署由 run_web_rk.sh 注入中文提示词
    std::string desc_prompt = getenv_str("VLM_DESCRIPTION_PROMPT",
        "Describe the pedestrian in the image in detail, including gender, age range, "
        "view direction, clothing, colors, carried objects, and other visible attributes.");
    c.description_prompt = strip_image_placeholder(desc_prompt);

    c.vlm_max_new_tokens = (int)getenv_int("VLM_MAX_NEW_TOKENS", 120);
    c.vlm_temperature = getenv_double("VLM_GENERATION_TEMPERATURE", 0.0);
    c.vlm_repetition_penalty = getenv_double("VLM_REPETITION_PENALTY", 1.2);

    // 属性线性头（qwen_classifer/export_head.py 导出）；VLM_ATTR_HEAD_PATH=off 显式禁用
    // （getenv_str 把空串当未设置，禁用哨兵值用 "off"）
    c.attr_head_path = getenv_str("VLM_ATTR_HEAD_PATH", models_dir + "/attr_head.bin");
    if (c.attr_head_path == "off") c.attr_head_path.clear();

    c.bge_onnx_path = getenv_str("EMBEDDING_ONNX_PATH", models_dir + "/onnx/bge-small-zh.onnx");
    c.bge_vocab_path = getenv_str("EMBEDDING_VOCAB_PATH",
        "/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5/vocab.txt");
    // BGE 官方检索用法：query 侧加指令前缀，doc 侧不加（非对称）。置空回到对称。
    c.embed_query_prefix = getenv_str("BGE_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：");

    // 调试捕获开关（web /debug 页）；默认关闭。上限 VLM_DEBUG_LOG_MAX 默认 0=不限
    c.debug_capture = getenv_bool("VLM_DEBUG_CAPTURE", false);
    c.debug_log_max = (int)getenv_int("VLM_DEBUG_LOG_MAX", 0);
    c.yolo_onnx_path = getenv_str("DETECTOR_ONNX_PATH", models_dir + "/onnx/yolov8n-dyn.onnx");

    return c;
}
