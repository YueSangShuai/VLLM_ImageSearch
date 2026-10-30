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
    std::string match_mode = "structured";  // structured | description | combined | rule_direct | reranker

    // Qwen3-VL-Reranker-2B（match_mode=reranker 时生效）
    std::string reranker_vision_path;   // reranker 视觉塔 .rknn
    std::string reranker_llm_path;      // reranker 语言塔 .rkllm
    std::string reranker_instruction;   // 必须与离线校准/host_ref 完全一致
    std::string reranker_mode = "logits";  // logits | hidden（hidden 需配套权重 bin）
    std::string reranker_weight_bin;    // W_yes-W_no float32 向量，空 = 自动探测
    long reranker_yes_id = 9693;
    long reranker_no_id = 2152;
    int reranker_max_pos_texts = 2;     // 每条规则最多打分的 positive 文本数（NPU 成本）
    bool reranker_use_negative = false; // 负样本同样打分并用 score_diff 拦截

    std::string description_prompt;   // 已剥掉 <image>
    std::string structured_prompt;    // 已剥掉 <image>；空串 = matcher 用内置属性列表构造
    int structured_max_new_tokens = 128;

    // rule_direct 模式
    std::string rule_judge_prompt;    // 已剥掉 <image>；空串 = matcher 用内置默认
    int rule_judge_max_new_tokens = 128;
    int vlm_max_new_tokens = 120;
    double vlm_temperature = 0.0;
    double vlm_repetition_penalty = 1.2;
    // 结构化路径专用：W8A8 下 1.2 压不住属性列表复读（10/28 实测 6 张新告警 3 张全枚举），
    // 1.4 复读消失（代价是出现列表外垃圾 token，matcher 解析时丢弃）；描述路径仍用 1.2。
    double structured_repetition_penalty = 1.4;

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
