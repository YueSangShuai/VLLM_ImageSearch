#pragma once
// RKLLM 后端封装（RK3588 NPU）：图像 + 提示词 -> 生成文本。
// 公共接口与 web_vllm_cpp 的 llama/mtmd 版 VlmDescriber 完全一致，
// matcher/monitor/server 等上游零改动。
// 构造参数按位置兼容旧签名：
//   vision_path -> .rknn 视觉塔；llm_path -> .rkllm 语言塔；
//   device / n_gpu_layers 在 NPU 路线忽略；n_ctx -> max_context_len。
// 行为对齐 rknn-llm v1.3.1 multimodal_model_demo：
//   视觉 448x448 固定分辨率（ROI 直接 bilinear resize），
//   embedding 以 RKLLM_INPUT_MULTIMODAL 喂入，Qwen3-VL special token 占位。
#include "frame.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class VlmDescriber {
public:
    VlmDescriber(const std::string & vision_path, const std::string & llm_path,
                 const std::string & device, int n_gpu_layers, int n_ctx,
                 double temperature, double repetition_penalty);
    ~VlmDescriber();
    VlmDescriber(const VlmDescriber &) = delete;
    VlmDescriber & operator=(const VlmDescriber &) = delete;

    bool ok() const;

    // 单轮生成。prompt 须已剥掉 <image>；内部在文首拼 <image> 占位符。
    // 多线程共用同一实例，内部加锁串行生成。
    // attrs_out 非空且属性头已加载时，顺带输出本次图像的属性（锁内计算，出参归调用方）。
    std::string describe(const ImageRGB & img, const std::string & prompt,
                         int max_new_tokens, double temperature, double repetition_penalty,
                         std::vector<std::string> * attrs_out = nullptr);

    // 属性线性头（qwen_classifer 导出，可选件）。必须在构造后调用（需视觉 embed_size）。
    // path 为空或加载失败：attrs_out 恒为空，预处理保持旧行为（拉伸 resize）。
    void setAttrHead(const std::string & path);
    bool attrHeadReady() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex mu_;  // 对应 python VLMDescriber._lock
};
