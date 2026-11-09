#include "text_embedder.h"

#include <cmath>
#include <stdexcept>
#include <cstring>

namespace {
const Ort::MemoryInfo g_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

TextEmbedder::TextEmbedder(const std::string &onnx_path, const std::string &sp_model_path,
                           int max_length)
    : env_(ORT_LOGGING_LEVEL_WARNING, "siglip_text"), max_length_(max_length) {
    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto st = sp_->Load(sp_model_path);   // v0.2.0 尚无 Open()，用 Load
    if (!st.ok())
        throw std::runtime_error("无法打开 sentencepiece 模型: " + sp_model_path + " (" + st.ToString() + ")");
    session_opts_.SetIntraOpNumThreads(2);
    session_opts_.SetInterOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts_);
}

void TextEmbedder::tokenize(const std::string &text, std::vector<int32_t> &ids,
                            std::vector<int32_t> &mask) const {
    // HF SiglipTokenizer(Gemma)：编码 + 尾加 </s>；截断后不补 eos（规则短句 64 位内基本不触发）
    std::vector<int> pieces;
    auto st = sp_->Encode(text, &pieces);
    if (!st.ok()) throw std::runtime_error("sentencepiece 分词失败: " + text);
    ids.assign((size_t)max_length_, pad_id_);
    mask.assign((size_t)max_length_, 0);
    size_t n = std::min<size_t>(pieces.size(), (size_t)max_length_ - 1);  // 给 eos 留位
    for (size_t i = 0; i < n; i++) ids[i] = (int32_t)pieces[i];
    ids[n] = eos_id_;
    size_t real = n + 1;
    for (size_t i = 0; i < real; i++) mask[i] = 1;
}

std::vector<float> TextEmbedder::encode(const std::vector<std::string> &texts) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<float> out;
    out.reserve(texts.size() * dim_);
    for (const auto &t : texts) {
        std::vector<int32_t> ids, mask;
        tokenize(t, ids, mask);

        std::array<int64_t, 2> shape{(int64_t)1, (int64_t)max_length_};
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(
            g_mem, ids.data(), ids.size(), shape.data(), 2));
        inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(
            g_mem, mask.data(), mask.size(), shape.data(), 2));

        const char *in_names[] = {"input_ids", "attention_mask"};
        const char *out_names[] = {"txt_emb"};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 2,
                                     out_names, 1);

        const float *emb = outputs[0].GetTensorData<float>();
        auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        dim_ = (int)dims.back();
        // 图内已 L2 归一化；此处兜底再归一一次（防导出图变化）
        std::vector<float> vec(emb, emb + dim_);
        double norm = 0;
        for (float v : vec) norm += (double)v * v;
        norm = std::sqrt(norm);
        if (norm > 0) for (auto &v : vec) v = (float)(v / norm);
        out.insert(out.end(), vec.begin(), vec.end());
    }
    return out;
}
