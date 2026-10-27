#pragma once
// 全局配置：环境变量解析，与 use_vllm_web.sh / processor.py 同名变量对齐。
#include <string>
#include <vector>

std::string getenv_str(const char *name, const std::string &fallback = "");
int64_t getenv_int(const char *name, int64_t fallback);
double getenv_double(const char *name, double fallback);
bool getenv_bool(const char *name, bool fallback);

// 去掉提示词里的 <image> 占位符；图片由 VlmDescriber 自行插入（对齐 processor.py strip_image_placeholder）
std::string strip_image_placeholder(const std::string &prompt);

struct Config {
    std::string exe_dir;              // 二进制所在目录，用于解析相对路径
    std::string data_dir;             // 共享 JSON 数据目录（同 Python 版 data/）
    std::string static_dir;           // 前端静态目录
    int port = 5003;

    // VLM
    std::string vlm_model_path;       // GGUF 主模型
    std::string mmproj_path;          // 视觉投影 GGUF
    std::string device;               // "cuda:0"/"cpu"
    int n_gpu_layers = -1;
    int n_ctx = 4096;
    std::string match_mode = "structured";  // structured | description | combined | rule_direct

    std::string description_prompt;   // 已剥掉 <image>
    std::string structured_prompt;    // 已剥掉 <image>；空串 = matcher 用内置属性列表构造
    int structured_max_new_tokens = 128;

    // rule_direct 模式
    std::string rule_judge_prompt;    // 已剥掉 <image>；空串 = matcher 用内置默认
    int rule_judge_max_new_tokens = 128;
    int vlm_max_new_tokens = 120;
    double vlm_temperature = 0.0;
    double vlm_repetition_penalty = 1.2;

    // 规则二次核验
    bool rule_description_enabled = false;
    bool rule_description_verify = true;
    int rule_description_max_new_tokens = 60;
    int rule_prompt_max_texts = 3;

    // 嵌入模型
    std::string bge_onnx_path;
    std::string bge_vocab_path;
    int bge_max_length = 512;
    // 非对称检索：规则文本=query（加前缀），图像描述=doc（不加）。BGE 官方用法。
    // 置空 = 对称编码（旧行为）。加前缀会整体抬升/移动分数，需重校 threshold。
    std::string embed_query_prefix;

    // 检测
    std::string yolo_onnx_path;
    int detector_input = 640;
    float det_conf = 0.5f;
    float det_iou = 0.7f;

    static Config load();
};
