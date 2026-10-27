#include "text_embedder.h"
#include "util.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace {

const Ort::MemoryInfo g_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

// HF BertTokenizer 辅助（do_lower_case=False 路径）
bool is_whitespace_cp(uint32_t cp) {
    return cp == 32 || cp == 9 || cp == 10 || cp == 13 || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

bool is_control_cp(uint32_t cp) {
    if (cp == 0) return false;
    if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return false;
    if (cp < 0x20 || cp == 0x7F) return true;
    if (cp >= 0x80 && cp <= 0x9F) return true;
    if (cp == 0xAD) return true;
    if (cp == 0x200B || cp == 0x200C || cp == 0x200D) return true;
    if (cp >= 0x200E && cp <= 0x200F) return true;
    if (cp == 0x202A || cp == 0x202B || cp == 0x202C || cp == 0x202D || cp == 0x202E) return true;
    if (cp >= 0x2060 && cp <= 0x2063) return true;
    if (cp == 0xFEFF) return true;
    return false;
}

bool is_chinese_cp(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// ASCII 标点 + 常见 CJK/全角标点（Unicode Po/Ps/Pe 近似）
bool is_punct_cp(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
        (cp >= 123 && cp <= 126)) return true;
    switch (cp) {
        case 0x2018: case 0x2019: case 0x201C: case 0x201D: case 0x3001: case 0x3002:
        case 0x300A: case 0x300B: case 0x300C: case 0x300D: case 0x300E: case 0x300F:
        case 0x3010: case 0x3011: case 0x3014: case 0x3015: case 0x3016: case 0x3017:
        case 0xFF01: case 0xFF05: case 0xFF07: case 0xFF0C: case 0xFF0E: case 0xFF1A:
        case 0xFF1B: case 0xFF1F: case 0xFF3B: case 0xFF3D: case 0xFF3F: case 0xFF5B:
        case 0xFF5D: case 0xFF61: case 0xFF03: case 0xFF04: case 0xFF06: case 0xFF0A:
        case 0xFF0F: case 0xFF20: case 0xFF3C: case 0xFF3E: case 0xFF40:
            return true;
    }
    return false;
}

// HF str.lower() 的实用近似：ASCII + 全角拉丁（中文不受影响）
uint32_t lower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return cp + 32;
    return cp;
}

}  // namespace

TextEmbedder::TextEmbedder(const std::string &onnx_path, const std::string &vocab_path,
                           int max_length)
    : env_(ORT_LOGGING_LEVEL_WARNING, "bge"), max_length_(max_length) {
    {
        std::ifstream in(vocab_path);
        if (!in) throw std::runtime_error("无法打开 vocab: " + vocab_path);
        std::string line;
        int id = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            vocab_[line] = id;
            id2token_.push_back(line);
            id++;
        }
        cls_id_ = vocab_.count("[CLS]") ? vocab_["[CLS]"] : 101;
        sep_id_ = vocab_.count("[SEP]") ? vocab_["[SEP]"] : 102;
        pad_id_ = vocab_.count("[PAD]") ? vocab_["[PAD]"] : 0;
        unk_id_ = vocab_.count("[UNK]") ? vocab_["[UNK]"] : 100;
    }
    session_opts_.SetIntraOpNumThreads(2);
    session_opts_.SetInterOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts_);
}

TextEmbedder::Encoded TextEmbedder::tokenize(const std::string &text) const {
    // 1. clean_text：控制字符/空白 → 空格
    // 注意：生产端 SentenceTransformer 读 sentence_bert_config.json 的 do_lower_case=true
    //（覆盖 tokenizer_config.json 的 false），实际行为是小写化，黄金样例已证实。
    auto cps = utf8_to_codepoints(text);
    std::vector<uint32_t> cleaned;
    cleaned.reserve(cps.size());
    for (uint32_t cp : cps) {
        if (is_control_cp(cp) || is_whitespace_cp(cp)) cleaned.push_back(32);
        else cleaned.push_back(lower_cp(cp));
    }

    // 2. 分词：中文逐字、标点独立、空白切词（单个累加器，避免漏 flush）
    std::vector<std::string> words;
    std::string buf;
    auto flush_word = [&]() {
        if (!buf.empty()) { words.push_back(buf); buf.clear(); }
    };
    auto cur = codepoints_to_utf8(cleaned);
    auto cps2 = utf8_to_codepoints(cur);
    for (uint32_t cp : cps2) {
        std::string ch = codepoints_to_utf8({cp});
        if (cp == 32) { flush_word(); }
        else if (is_chinese_cp(cp) || is_punct_cp(cp)) {
            flush_word();
            words.push_back(ch);
        } else {
            buf += ch;
        }
    }
    flush_word();

    // 3. WordPiece（不做 lower）
    auto vocab_find = [&](const std::string &s) -> int32_t {
        auto it = vocab_.find(s);
        return it == vocab_.end() ? -1 : it->second;
    };
    std::vector<int64_t> ids;
    for (const auto &w : words) {
        auto wcps = utf8_to_codepoints(w);
        bool unk = false;
        std::vector<int64_t> pieces;
        size_t start = 0;
        while (start < wcps.size()) {
            size_t end = wcps.size();
            int32_t found = -1;
            std::string found_tok;
            while (start < end) {
                auto sub = codepoints_to_utf8(std::vector<uint32_t>(wcps.begin() + start,
                                                                    wcps.begin() + end));
                if (start > 0) sub = "##" + sub;
                int32_t id = vocab_find(sub);
                if (id >= 0) { found = id; found_tok = sub; break; }
                end--;
            }
            if (found < 0) { unk = true; break; }
            pieces.push_back(found);
            start = end;
        }
        if (unk) {
            ids.push_back(unk_id_);
        } else {
            ids.insert(ids.end(), pieces.begin(), pieces.end());
        }
    }

    // 4. [CLS] ... [SEP]，max_length 截断
    if ((int)ids.size() > max_length_ - 2) ids.resize(max_length_ - 2);
    ids.insert(ids.begin(), cls_id_);
    ids.push_back(sep_id_);

    Encoded e;
    e.seq_len = ids.size();
    e.input_ids = ids;
    e.attention_mask.assign(ids.size(), 1);
    e.token_type_ids.assign(ids.size(), 0);
    return e;
}

std::vector<float> TextEmbedder::encode(const std::vector<std::string> &texts,
                                        const std::string &query_prefix) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<float> out;
    out.reserve(texts.size() * dim_);
    for (const auto &t : texts) {
        auto enc = tokenize(query_prefix.empty() ? t : query_prefix + t);
        size_t seq = enc.seq_len;

        std::array<int64_t, 2> shape{(int64_t)1, (int64_t)seq};
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            g_mem, enc.input_ids.data(), enc.input_ids.size(), shape.data(), 2));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            g_mem, enc.attention_mask.data(), enc.attention_mask.size(), shape.data(), 2));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            g_mem, enc.token_type_ids.data(), enc.token_type_ids.size(), shape.data(), 2));

        const char *in_names[] = {"input_ids", "attention_mask", "token_type_ids"};
        const char *out_names[] = {"last_hidden_state"};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3,
                                     out_names, 1);

        // CLS pooling：[0][0][:]
        const float *hs = outputs[0].GetTensorData<float>();
        auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int hid = (int)dims[2];
        dim_ = hid;
        std::vector<float> vec(hs, hs + hid);
        double norm = 0;
        for (float v : vec) norm += (double)v * v;
        norm = std::sqrt(norm);
        if (norm > 0) for (auto &v : vec) v = (float)(v / norm);
        out.insert(out.end(), vec.begin(), vec.end());
    }
    return out;
}
