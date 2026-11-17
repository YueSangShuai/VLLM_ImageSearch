#pragma once
// MARS 文本塔（RKNN rk3588）：英文查询 → 256-d L2 归一化向量。
// 链路复刻训练/评测口径：pre_caption 清洗（dataset/utils.py，50 词截断）
// → BertTokenizer(uncased) WordPiece → 定长 128（pad=0）
// → mars_text_fp16.rknn（前 6 层 BERT + text_proj + L2 归一化在图内）。
// 实测：不清洗直接喂原句会让排序翻转（正 -0.15 < 负 -0.07），清洗必须在此复刻。
#include <rknn_api.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

std::string pre_caption(const std::string &caption);  // dataset/utils.py 的等价复刻

class TextEmbedder {
public:
    TextEmbedder(const std::string &rknn_path, const std::string &vocab_path, int max_len);
    ~TextEmbedder();

    // 返回 L2 归一化后的 256-d 文本向量（内部先做 pre_caption）
    std::vector<float> encode(const std::string &text);
    // 已清洗文本直接编码（golden 对拍用）
    std::vector<float> encode_clean(const std::string &cleaned);
    int dim() const { return dim_; }

    // HF BertTokenizer 口径校验用（golden_check 对比 x86 tokenizers 输出）
    void tokenize(const std::string &cleaned, std::vector<int32_t> &ids,
                  std::vector<float> &attn) const;

private:

    rknn_context ctx_ = 0;
    rknn_tensor_attr in_ids_attr_ = {};
    rknn_tensor_attr in_mask_attr_ = {};
    rknn_tensor_attr out_attr_ = {};
    std::unordered_map<std::string, int32_t> vocab_;
    int32_t cls_id_ = 101, sep_id_ = 102, pad_id_ = 0, unk_id_ = 100;
    int max_len_ = 128;
    int dim_ = 256;
    std::mutex mu_;
};
