#pragma once
// BGE 文本向量：WordPiece 分词 + ONNX Runtime BERT + CLS pooling + L2 归一化。
// 与 sentence-transformers 的 bge-small-zh-v1.5 管线逐位对齐（CLS pooling，do_lower_case=False）。
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdint>

class TextEmbedder {
public:
    TextEmbedder(const std::string &onnx_path, const std::string &vocab_path,
                 int max_length = 512);

    // 返回归一化后的向量（rows = texts.size(), dim = 512）
    // query_prefix 非空时按 BGE 官方用法给每条文本加检索指令前缀（非对称检索的 query 侧）
    std::vector<float> encode(const std::vector<std::string> &texts,
                              const std::string &query_prefix = "");
    std::vector<float> encode_one(const std::string &text,
                                  const std::string &query_prefix = "") {
        return encode({text}, query_prefix);
    }
    int dim() const { return dim_; }

private:
    struct Encoded {
        std::vector<int64_t> input_ids, attention_mask, token_type_ids;
        size_t seq_len;
    };
    Encoded tokenize(const std::string &text) const;

    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id2token_;
    int32_t cls_id_ = 101, sep_id_ = 102, pad_id_ = 0, unk_id_ = 100;
    int max_length_;
    int dim_ = 512;
    std::mutex mu_;
};
