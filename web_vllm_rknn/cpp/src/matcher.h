#pragma once
// 对齐 processor.py VLMTextMatcher：VLM 描述 + BGE 向量 / 属性符号匹配。
// 两种模式：description | combined（combined 用属性线性头）
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

    // 回滚最近一次 match() 中这些规则启动的冷却（朝向事后过滤丢弃的命中用）：
    // match() 不知道过滤，触发即记 last_trigger_；调用方确认丢弃后调本方法还原旧值
    void undoTrigger(const std::vector<std::string> & rule_ids);

    // ---- 调试捕获（VLM_DEBUG_CAPTURE）----
    // match() 期间生成的描述快照：{description, description_zh, attributes,
    // rules(逐规则得分明细：name/type/score/threshold/status/hit，symbolic 附带
    // expected/hit_attrs/missing_attrs，semantic 附带 positive_list/positive_scores)}
    json lastDebugSnapshot();
    // 未走 match 描述路径（无规则等）时补一次描述生成，供调试页展示
    json describeForDebug(const ImageRGB & image);

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

    // ---- 业务流程 ----
    struct DescResult {                  // 一次描述生成的输出；不再走 last_* 成员，
        std::string text;                // 使描述阶段可以完全脱离 mu_ 跑
        std::string zh;
        std::vector<std::string> attrs;
    };
    DescResult describeImage(const ImageRGB & img, const std::string & prompt);

    VlmDescriber * vlm_;
    TextEmbedder * emb_;
    Config cfg_;

    std::string mode_;
    std::string base_prompt_, description_prompt_;
    int vlm_max_;
    double temp_, rep_;

    std::vector<json> rules_;
    std::map<std::string, EmbMat> rule_emb_;
    std::map<std::string, EmbMat> neg_emb_;          // 负样本文本向量：pos 须超过 max(neg)+score_diff 才告警
    std::map<std::string, std::vector<std::string>> pos_texts_;
    std::map<std::string, std::vector<std::string>> neg_texts_;
    std::map<std::string, std::optional<std::set<std::string>>> attr_pos_;  // nullopt=需语义匹配
    std::map<std::string, std::set<std::string>> attr_neg_;   // 符号规则：负样本属性（检出任一 → neg=1.0）
    std::map<std::string, double> last_trigger_;
    // 本次 match() 触发前的 last_trigger_ 旧值（-1=原本无记录），供 undoTrigger 回滚；
    // match() 开头清空。仅设备线程触碰（match/undoTrigger 同线程），锁内读写即可
    std::map<std::string, double> cooldown_backup_;
    std::map<std::string, std::vector<float>> vec_cache_;     // 规则文本→向量缓存：updateRules 只编码新文本

    json dbg_snapshot_ = json::object();       // 调试捕获快照（match 每次调用先清空）
    bool head_on_ = false;                     // 属性线性头可用：combined 不再走 VLM 结构化生成

    std::mutex mu_;   // updateRules(HTTP线程) 与 match(设备线程) 互斥
};
