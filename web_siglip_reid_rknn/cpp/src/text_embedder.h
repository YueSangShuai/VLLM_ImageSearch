#pragma once
// SigLIP2 文本向量：sentencepiece(Gemma) 分词 + ONNX Runtime 文本塔 + L2 归一化。
// 与 transformers google/siglip2-base-patch16-224 tokenizer 对齐：
//   add_bos=False, add_eos=True(</s>=1)，padding=max_length(64) 右侧补 pad=0。
// 输出 768-d，与 SiglipVision 图像向量同空间（cosine 直接可比）。
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

class TextEmbedder {
public:
    TextEmbedder(const std::string &onnx_path, const std::string &sp_model_path,
                 int max_length = 64);

    // 返回归一化后的向量（rows = texts.size(), dim = dim_）
    std::vector<float> encode(const std::vector<std::string> &texts);
    std::vector<float> encode_one(const std::string &text) { return encode({text}); }
    int dim() const { return dim_; }

private:
    // 固定长度 max_length_：ids 尾补 eos 后 pad 0，mask 真实位为 1
    void tokenize(const std::string &text, std::vector<int32_t> &ids,
                  std::vector<int32_t> &mask) const;

    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    int max_length_;
    int dim_ = 768;
    int32_t eos_id_ = 1, pad_id_ = 0;
    std::mutex mu_;
};
