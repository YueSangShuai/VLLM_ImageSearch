#pragma once
// 全局配置：环境变量解析，与 use_vllm_web.sh / processor.py 同名变量对齐。
#include <string>
#include <vector>

std::string getenv_str(const char *name, const std::string &fallback = "");
int64_t getenv_int(const char *name, int64_t fallback);
double getenv_double(const char *name, double fallback);
bool getenv_bool(const char *name, bool fallback);

struct Config {
    std::string exe_dir;              // 二进制所在目录，用于解析相对路径
    std::string data_dir;             // 共享 JSON 数据目录（同 Python 版 data/）
    std::string static_dir;           // 前端静态目录
    int port = 5003;

    // SigLIP2 视觉塔（NPU）
    std::string siglip_vision_rknn;

    // SigLIP2 文本塔（板端 CPU, ORT）+ Gemma sentencepiece 分词
    std::string siglip_text_onnx;
    std::string sp_model_path;        // tokenizer.model（google/siglip2-base-patch16-224）
    int text_max_length = 64;         // SigLIP 文本塔固定序列长

    // SigLIP 分值尺度远小于 BGE（正确属性 ~0.15-0.22，无关 ~0，PETA 实测）。
    // 规则未显式给 threshold 时的回退值；逐规则 threshold 字段优先。
    double default_threshold = 0.15;

    // 调试捕获：DEBUG_CAPTURE=1 时，所有检测到的行人（无论是否命中规则）
    // 都存图 + 逐规则打分快照进 data/debug_log.json，web /debug 页查看。默认关闭。
    bool debug_capture = false;

    // 检测
    std::string yolo_onnx_path;
    float det_conf = 0.5f;
    float det_iou = 0.7f;

    static Config load();
};
