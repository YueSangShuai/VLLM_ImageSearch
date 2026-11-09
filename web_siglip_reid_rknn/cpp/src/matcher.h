#pragma once
// SigLIP2 直接匹配：行人裁剪 → 视觉塔图像向量 × 规则文本向量 cosine。
// 替代原「VLM 描述 + BGE」两级结构；符号/属性支线（attr_head 绑定 Qwen 嵌入）已移除。
// 分值尺度：正确属性 ≈0.15–0.22，无关 ≈0（PETA 实测），默认阈值见 Config::default_threshold。
#include "config.h"
#include "frame.h"
#include "json_store.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

class SiglipVision;
class TextEmbedder;

class Matcher {
public:
    Matcher(SiglipVision * vision, TextEmbedder * embedder, const Config & cfg);

    // rules：tasks.json 按 device 合并后的规则数组（monitor 负责合并）
    // 规则文本（positive 或回退 text）经 SigLIP 文本塔预编码
    void updateRules(const json & rules);

    // 返回 matches json 数组；字段为原 semantic 分支的子集（无 description/attributes）
    json match(const ImageRGB & image);

    // ---- 调试捕获（DEBUG_CAPTURE）----
    // match() 期间的快照：{rules: 逐规则得分明细（name/score/threshold/status/hit/
    // positive_list/positive_scores，配了 negative 的附 negative_list/negative_scores/
    // max_negative_score，status 可为 negative_higher）}；未走 match 路径时补一次打分
    json lastDebugSnapshot();
    json scoreForDebug(const ImageRGB & image);

private:
    struct EmbMat {                      // python 的 (N, dim) ndarray
        std::vector<float> v;            // 行优先 N*dim
        size_t n = 0;
        size_t dim = 768;
    };

    std::vector<std::string> normalizeTexts(const json & texts) const;

    // 打分 + 填 dbg_rules（不含冷却触发记账）；match 与 scoreForDebug 共用。调用方持 mu_。
    json scoreImage(const std::vector<float> & image_emb, json & dbg_rules);

    SiglipVision * vision_;
    TextEmbedder * emb_;
    Config cfg_;

    std::vector<json> rules_;
    std::map<std::string, EmbMat> rule_emb_;
    std::map<std::string, EmbMat> neg_emb_;   // 负样本文本向量：pos 须超过 max(neg)+score_diff 才告警
    std::map<std::string, std::vector<std::string>> pos_texts_;
    std::map<std::string, std::vector<std::string>> neg_texts_;
    std::map<std::string, double> last_trigger_;

    json dbg_snapshot_ = json::object();
    std::mutex mu_;   // updateRules(HTTP线程) 与 match(设备线程) 互斥
};
