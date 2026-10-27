#pragma once
// llama.cpp + libmtmd 封装：图像 + 提示词 -> 生成文本。
// 行为对齐 vlm_search/vlm_describer.py 的 GGUF 后端 + llama-mtmd-cli 单轮生成路径
// （PROGRESS.md 中实图验证用的就是这条链路）。
// 注意：本头文件刻意不引入 llama/common 头（pimpl），避免与 json_store 的符号冲突。
#include "frame.h"
#include <memory>
#include <mutex>
#include <string>

class VlmDescriber {
public:
    // device 以 "cpu" 开头 -> 纯 CPU；否则 GPU（n_gpu_layers 生效）
    VlmDescriber(const std::string & gguf_path, const std::string & mmproj_path,
                 const std::string & device, int n_gpu_layers, int n_ctx,
                 double temperature, double repetition_penalty);
    ~VlmDescriber();
    VlmDescriber(const VlmDescriber &) = delete;
    VlmDescriber & operator=(const VlmDescriber &) = delete;

    bool ok() const;

    // 单轮生成。prompt 须已剥掉 <image>；内部在文首拼 mtmd media marker。
    // 与 Python 端一致：多线程共用同一实例，内部加锁串行生成。
    std::string describe(const ImageRGB & img, const std::string & prompt,
                         int max_new_tokens, double temperature, double repetition_penalty);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex mu_;  // 对应 python VLMDescriber._lock
};
