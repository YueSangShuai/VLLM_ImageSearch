#pragma once
// Qwen3-VL-Reranker-2B 板端打分器（RKNN 视觉塔 + RKLLM 语言塔，GET_LOGITS 模式）。
// 输入 (ROI 图, 规则文本) -> sigmoid(logit_yes - logit_no) 相关性分数。
// prompt 与 HF chat_template 渲染对齐（渲染样例 calib/prompt_sample.txt）：
//   im_start/system: Judge whether the Document meets the requirements ... im_end
//   im_start/user: <Instruct>: {instruct}<Query>:{rule_text}<Document>:{image}im_end
//   im_start/assistant  （生成前缀；GET_LOGITS 取末 token logits）
// 兜底：mode="hidden" 时用末 token hidden 与 W_yes-W_no 权重向量（rerank_yes_no_diff.bin）点积，
// 与 HF score_linear 数学等价（config.json tie_word_embeddings=true，logits=hidden@W^T）。
#include "frame.h"
#include <memory>
#include <mutex>
#include <string>

class RerankerScorer {
public:
    RerankerScorer(const std::string & vision_path, const std::string & llm_path,
                   int n_ctx, const std::string & instruction,
                   const std::string & mode,           // "logits" | "hidden"
                   const std::string & weight_bin,     // hidden 模式权重，"" = 自动探测
                   long yes_id, long no_id);
    ~RerankerScorer();
    RerankerScorer(const RerankerScorer &) = delete;
    RerankerScorer & operator=(const RerankerScorer &) = delete;

    bool ok() const;

    // 单条打分；<0 = 失败。同一图像连续多条规则时内部复用视觉 embedding（按 RGB 数据指针+尺寸缓存）。
    double score(const ImageRGB & img, const std::string & text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex mu_;
};
