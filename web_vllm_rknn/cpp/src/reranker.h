#pragma once
// Qwen3-Reranker-0.6B w8a8（RKLLM, NPU 第二实例）：BGE 初筛后的二次核验。
// 与 VlmDescriber 同进程共存（板端实测 2026-11-18：两 rkllm 上下文并存无冲突，
// 单对 prefill ~231ms，边界案例判定与主机 bf16 一致 6/6）。
//
// 输入走 RKLLM_INPUT_PROMPT + rkllm_set_chat_template("","","")置空模板，
// C++ 手工拼装完整 ChatML（复刻模型自带 jinja 模板，纯静态布局）。
// 打分：末位 yes(9693)/no(2152) 两 token logit softmax = P(yes)。
// 失败返回 -1，调用方按放行处理（核验故障不阻断告警链路）。
#include <memory>
#include <mutex>
#include <string>

class Reranker {
public:
    Reranker(const std::string & model_path, double default_threshold);
    ~Reranker();
    Reranker(const Reranker &) = delete;
    Reranker & operator=(const Reranker &) = delete;

    bool ok() const;
    double defaultThreshold() const;

    // query=规则正样本文本 document=VLM 描述 → P(yes)∈[0,1]；<0 = 失败
    double rerank(const std::string & query, const std::string & document);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex mu_;       // 多线程（各设备）共用一个实例，串行化 rkllm_run
    double threshold_;
    bool ok_ = false;
};
