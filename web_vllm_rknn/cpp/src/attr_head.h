#pragma once
// 行人属性线性头（qwen_classifer 训练：视觉塔 mean-pooling 2048 维 -> 54 属性 sigmoid）。
// 权重由 qwen_classifer/export_head.py 导出为 attr_head.bin；头是可选件，加载失败仅日志。
#include <string>
#include <vector>

class AttrHead {
public:
    // expect_in_dim 传运行时视觉 embed_size，与 bin 内 in_dim 双校验（防换视觉塔错配）
    bool load(const std::string & path, int expect_in_dim);
    bool ready() const { return ready_; }

    // embed：run_imgenc 交错布局（token i 段 j 在 (i*n_output + j)*embed_size，段 0 = main）。
    // mean-pool 段 0 -> W·x+b -> sigmoid -> 逐属性阈值；attrs_out 填属性名（含 Female，
    // Male 派生由调用方负责，与 extractStructuredAttributes 语义对齐）。
    void run(const float * embed, int n_tokens, int n_output, int embed_size,
             std::vector<std::string> * attrs_out,
             std::vector<float> * probs_out = nullptr) const;

private:
    bool ready_ = false;
    int in_dim_ = 0;
    int n_attrs_ = 0;
    std::vector<float> w_;   // [n_attrs * in_dim] row-major
    std::vector<float> b_;
    std::vector<float> thr_;
};
