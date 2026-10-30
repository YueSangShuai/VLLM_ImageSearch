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

    std::string models_dir = getenv_str("WEB_VLLM_MODELS_DIR", root + "/models");
    if (models_dir.empty()) models_dir = c.exe_dir + "/models";

    // VLM：同名环境变量复用，值换成 RKNN/RKLLM 产物（板上路径）
    // vlm_model_path -> 视觉塔 .rknn；mmproj_path -> 语言塔 .rkllm
    c.vlm_model_path = getenv_str("VLM_MODEL_NAME",
        "/home/pi/output_rk3588/models/qwen3vl-ft_vision_rk3588.rknn");
    c.mmproj_path = getenv_str("VLM_MMPROJ_PATH",
        "/home/pi/output_rk3588/models/qwen3vl-ft-merged_w8a8_rk3588.rkllm");
    c.device = getenv_str("VLM_DEVICE", "cuda:0");
    int ngl = (int)getenv_int("VLM_GGUF_N_GPU_LAYERS", -1);
    if (c.device.rfind("cpu", 0) == 0) ngl = 0;
    c.n_gpu_layers = ngl;
    c.n_ctx = (int)getenv_int("VLM_GGUF_N_CTX", 4096);

    std::string mode = getenv_str("VLM_MATCH_MODE", "combined");
    for (auto &ch : mode) ch = (char)tolower(ch);
    if (mode != "structured" && mode != "description" && mode != "combined" &&
        mode != "rule_direct" && mode != "reranker") {
        printf("未知的VLM_MATCH_MODE: %s，回退到 structured\n", mode.c_str());
        mode = "structured";
    }
    c.match_mode = mode;

    // Reranker（Qwen3-VL-Reranker-2B 板端打分）
    c.reranker_vision_path = getenv_str("RERANKER_VISION_PATH",
        models_dir + "/rk3588/qwen3vl-reranker_vision_rk3588.rknn");
    c.reranker_llm_path = getenv_str("RERANKER_RKLLM_PATH",
        models_dir + "/rk3588/qwen3vl-reranker-2b_w8a8_rk3588.rkllm");
    // instruction 改动 = 换分数分布，必须与 scripts/host_ref_rerank.py 默认值保持一致
    c.reranker_instruction = getenv_str("RERANKER_INSTRUCTION",
        "Given a pedestrian description, retrieve surveillance images of pedestrians matching it.");
    c.reranker_mode = getenv_str("RERANKER_MODE", "logits");
    c.reranker_weight_bin = getenv_str("RERANKER_WEIGHT_BIN", "");
    c.reranker_yes_id = (long)getenv_int("RERANKER_YES_ID", 9693);
    c.reranker_no_id = (long)getenv_int("RERANKER_NO_ID", 2152);
    c.reranker_max_pos_texts = (int)getenv_int("RERANKER_MAX_POS_TEXTS", 2);
    c.reranker_use_negative = getenv_bool("RERANKER_NEG", false);

    // 代码级默认与 processor.py DEFAULT_DESCRIPTION_PROMPT 一致（英文）；
    // 实际部署由 use_vllm_web.sh / run_web.sh 注入中文提示词
    std::string desc_prompt = getenv_str("VLM_DESCRIPTION_PROMPT",
        "Describe the pedestrian in the image in detail, including gender, age range, "
        "view direction, clothing, colors, carried objects, and other visible attributes.");
    std::string judge_prompt = getenv_str("VLM_RULE_JUDGE_PROMPT", "");  // 空 = matcher 用 DEFAULT_RULE_JUDGE_PROMPT
    c.rule_judge_max_new_tokens = (int)getenv_int("VLM_RULE_JUDGE_MAX_NEW_TOKENS", 128);
    std::string struct_prompt = getenv_str("VLM_STRUCTURED_PROMPT", "");
    if (struct_prompt.empty()) {
        // 与 processor.py DEFAULT_STRUCTURED_PROMPT 一致（PEDESTRIAN_ATTRIBUTES 见 matcher.cpp）
        struct_prompt = ""; // 由 matcher 用内置列表构造
    }
    c.description_prompt = strip_image_placeholder(desc_prompt);
    c.structured_prompt = struct_prompt.empty() ? "" : strip_image_placeholder(struct_prompt);
    // {rules} 占位符缺失时回退默认（与 processor.py __init__ 一致）
    c.rule_judge_prompt = judge_prompt.empty() ? "" : strip_image_placeholder(judge_prompt);

    c.structured_max_new_tokens = (int)getenv_int("VLM_STRUCTURED_MAX_NEW_TOKENS", 128);
    c.vlm_max_new_tokens = (int)getenv_int("VLM_MAX_NEW_TOKENS", 120);
    c.vlm_temperature = getenv_double("VLM_GENERATION_TEMPERATURE", 0.0);
    c.vlm_repetition_penalty = getenv_double("VLM_REPETITION_PENALTY", 1.2);
    c.structured_repetition_penalty =
        getenv_double("VLM_STRUCTURED_REPETITION_PENALTY", c.vlm_repetition_penalty > 1.2
                        ? c.vlm_repetition_penalty : 1.4);

    c.rule_description_enabled = getenv_bool("VLM_RULE_DESCRIPTION_ENABLED", true);
    c.rule_description_verify = getenv_bool("VLM_RULE_DESCRIPTION_VERIFY", true);
    c.rule_description_max_new_tokens = (int)getenv_int("VLM_RULE_DESCRIPTION_MAX_NEW_TOKENS", 60);
    c.rule_prompt_max_texts = (int)getenv_int("VLM_RULE_PROMPT_MAX_TEXTS", 3);

    c.bge_onnx_path = getenv_str("EMBEDDING_ONNX_PATH", models_dir + "/onnx/bge-small-zh.onnx");
    c.bge_vocab_path = getenv_str("EMBEDDING_VOCAB_PATH",
        "/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5/vocab.txt");
    // BGE 官方检索用法：query 侧加指令前缀，doc 侧不加（非对称）。置空回到对称。
    c.embed_query_prefix = getenv_str("BGE_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：");
    c.yolo_onnx_path = getenv_str("DETECTOR_ONNX_PATH", models_dir + "/onnx/yolov8n-dyn.onnx");

    return c;
}
