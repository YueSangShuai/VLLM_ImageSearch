#pragma once
// 对齐 processor.py VLMTextMatcher：VLM 描述/结构化属性 + BGE 向量 / 符号匹配。
// 四种模式：structured | description | combined | rule_direct
#include "config.h"
#include "frame.h"
#include "json_store.h"

#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

class VlmDescriber;
class TextEmbedder;

class Matcher {
public:
    Matcher(VlmDescriber * describer, TextEmbedder * embedder, const Config & cfg);

    // rules：tasks.json 按 device 合并后的规则数组（monitor 负责合并）
    void updateRules(const json & rules);

    // 对应 match()：返回 matches json 数组（字段名与 Python 逐一对应）
    json match(const ImageRGB & image);

private:
    struct EmbMat {                      // python 的 (N, dim) ndarray
        std::vector<float> v;            // 行优先 N*dim
        size_t n = 0;
        size_t dim = 512;
    };

    // ---- 属性表 / 文本工具（对应下划线方法）----
    std::vector<std::string> normalizeTexts(const json & texts) const;
    std::string lookupAttribute(const std::string & token) const;   // "" = 未命中
    std::vector<std::string> extractValidAttributes(const std::string & text) const;
    std::optional<std::set<std::string>> parseRuleAttributes(const std::string & text) const;
    std::string sanitizeRuleText(const std::string & text, size_t max_len) const;
    std::pair<std::string, std::string> splitBilingual(const std::string & text) const;
    std::string cleanVlmDescription(const std::string & text, size_t max_chars) const;
    static bool isNoMatchDescription(const std::string & text);
    std::string extractVerifiedRuleDescription(const std::string & text) const;

    // ---- 业务流程 ----
    std::vector<std::string> extractStructuredAttributes(const ImageRGB & img);
    std::string describeImage(const ImageRGB & img);
    std::string describeRuleMatch(const json & rule, const std::vector<std::string> & pos,
                                  const std::vector<std::string> & neg,
                                  const std::string & fallback, const ImageRGB & img);
    std::string buildRuleDescriptionPrompt(const json & rule, const std::vector<std::string> & pos,
                                           const std::vector<std::string> & neg) const;
    std::string buildRuleJudgePrompt(const std::vector<json> & rules) const;
    std::vector<json> parseMatchedRuleNames(const std::string & output,
                                            const std::vector<json> & candidates) const;
    json matchRuleDirect(const ImageRGB & img, double now);

    VlmDescriber * vlm_;
    TextEmbedder * emb_;
    Config cfg_;

    std::string mode_;
    std::string base_prompt_, structured_prompt_, judge_prompt_, description_prompt_;
    int structured_max_, vlm_max_, judge_max_, rule_desc_max_, rule_prompt_max_texts_;
    double temp_, rep_;
    bool rule_desc_enabled_, rule_desc_verify_;

    std::vector<json> rules_;
    std::map<std::string, EmbMat> rule_emb_, neg_emb_;
    std::map<std::string, std::vector<std::string>> pos_texts_, neg_texts_;
    std::map<std::string, std::optional<std::set<std::string>>> attr_pos_;  // nullopt=需语义匹配
    std::map<std::string, std::set<std::string>> attr_neg_;
    std::map<std::string, double> last_trigger_;

    std::string last_desc_zh_;                 // last_description_zh
    std::vector<std::string> last_attrs_;      // last_attributes

    std::mutex mu_;   // updateRules(HTTP线程) 与 match(设备线程) 互斥
};
