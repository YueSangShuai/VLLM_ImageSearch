#include "matcher.h"
#include "siglip_vision.h"
#include "text_embedder.h"
#include "util.h"

#include <algorithm>
#include <cmath>

// ==================== 通用工具 ====================

static double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

static std::string getStr(const json & j, const char * key, const std::string & def = "") {
    if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
    return def;
}

static double ruleNumber(const json & r, const char * key, double def) {
    if (!r.contains(key)) return def;
    const json & v = r[key];
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) { try { return std::stod(v.get<std::string>()); } catch (...) { return def; } }
    return def;
}
static bool ruleBool(const json & r, const char * key, bool def) {
    if (!r.contains(key)) return def;
    const json & v = r[key];
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0;
    return def;
}

// ==================== Matcher ====================

Matcher::Matcher(SiglipVision * vision, TextEmbedder * embedder, const Config & cfg)
    : vision_(vision), emb_(embedder), cfg_(cfg) {}

std::vector<std::string> Matcher::normalizeTexts(const json & texts) const {
    std::vector<std::string> out;
    if (texts.is_string()) {
        std::string t = trim(texts.get<std::string>());
        if (!t.empty()) out.push_back(t);
        return out;
    }
    if (texts.is_array()) {
        for (const auto & t : texts) {
            if (!t.is_string()) continue;
            std::string s = trim(t.get<std::string>());
            if (!s.empty()) out.push_back(s);
        }
    }
    return out;
}

void Matcher::updateRules(const json & rules) {
    std::lock_guard<std::mutex> lk(mu_);
    rules_.clear();
    if (rules.is_array()) {
        for (const auto & r : rules) {
            if (r.is_object() && ruleBool(r, "enabled", true)) rules_.push_back(r);
        }
    }
    rule_emb_.clear();
    pos_texts_.clear();
    neg_emb_.clear();
    neg_texts_.clear();

    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        // positive 为空时回退 [text]（英文属性短句，如 "person, black upper clothing"）
        json pos_json = rule.contains("positive") ? rule["positive"] : json();
        std::vector<std::string> positives = normalizeTexts(pos_json);
        if (positives.empty()) positives = normalizeTexts(json::array({getStr(rule, "text")}));

        pos_texts_[id] = positives;

        // SigLIP 图文同空间：规则文本与图像对称编码，无 query 前缀
        EmbMat m;
        if (!positives.empty()) {
            m.v = emb_->encode(positives);
            m.n = positives.size();
            m.dim = emb_->dim();
        }
        rule_emb_[id] = m;

        // 负样本：与正样本同口径编码（同步 web_vllm_rknn 平台 11-04 负样本门），
        // 打分时 pos 须 > max(neg)+score_diff 才告警
        std::vector<std::string> negatives =
            normalizeTexts(rule.contains("negative") ? rule["negative"] : json());
        neg_texts_[id] = negatives;
        EmbMat nm;
        if (!negatives.empty()) {
            nm.v = emb_->encode(negatives);
            nm.n = negatives.size();
            nm.dim = emb_->dim();
        }
        neg_emb_[id] = nm;
    }
}

// ---- 调试捕获 ----

json Matcher::lastDebugSnapshot() {
    std::lock_guard<std::mutex> lk(mu_);
    return dbg_snapshot_;
}

json Matcher::scoreForDebug(const ImageRGB & image) {
    std::lock_guard<std::mutex> lk(mu_);
    dbg_snapshot_ = json::object();
    auto image_emb = vision_->embed(image);
    json dbg_rules = json::array();
    scoreImage(image_emb, dbg_rules);
    dbg_snapshot_["rules"] = dbg_rules;
    return dbg_snapshot_;
}

// 逐规则打分（含未命中/冷却中）。冷却中的规则只记分不记账。调用方持 mu_。
json Matcher::scoreImage(const std::vector<float> & image_emb, json & dbg_rules) {
    json matches = json::array();
    double now = std::chrono::duration<double>(
                     std::chrono::system_clock::now().time_since_epoch()).count();

    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        double interval = ruleNumber(rule, "interval", 60);
        auto lt = last_trigger_.find(id);
        bool in_cooldown = lt != last_trigger_.end() && now - lt->second < interval;

        json d;
        d["rule_name"] = getStr(rule, "name");
        d["threshold"] = round4(ruleNumber(rule, "threshold", cfg_.default_threshold));
        d["score_diff"] = round4(ruleNumber(rule, "score_diff", 0));
        d["hit"] = false;

        auto pe = rule_emb_.find(id);
        if (pe == rule_emb_.end() || pe->second.n == 0) {
            d["status"] = "no_embedding";
            dbg_rules.push_back(d);
            continue;
        }
        const EmbMat & pos = pe->second;
        size_t dim = std::min(pos.dim, image_emb.size());
        double best_score = -1e30;
        size_t best_idx = 0;
        json positive_scores = json::array();
        for (size_t r = 0; r < pos.n; r++) {
            const float * row = pos.v.data() + r * pos.dim;
            float s = 0;
            for (size_t k = 0; k < dim; k++) s += row[k] * image_emb[k];
            positive_scores.push_back(round4(s));
            if (s > best_score) { best_score = s; best_idx = r; }
        }

        // 负样本向量 vs 图像向量：逐条得分 + 最大值（同步 web_vllm_rknn 平台负样本门）
        double max_negative_score = 0.0;
        json negative_scores = json::array();
        auto ne = neg_emb_.find(id);
        if (ne != neg_emb_.end() && ne->second.n > 0) {
            const EmbMat & neg = ne->second;
            size_t ndim = std::min(neg.dim, image_emb.size());
            for (size_t r = 0; r < neg.n; r++) {
                const float * row = neg.v.data() + r * neg.dim;
                float s = 0;
                for (size_t k = 0; k < ndim; k++) s += row[k] * image_emb[k];
                negative_scores.push_back(round4(s));
                if ((double)s > max_negative_score) max_negative_score = (double)s;
            }
        }

        double score_diff = ruleNumber(rule, "score_diff", 0);
        double threshold = ruleNumber(rule, "threshold", cfg_.default_threshold);
        // 负样本门：pos 须超过 max(neg)+score_diff；无负样本时退化为旧行为 pos>score_diff
        bool passed = best_score >= threshold && best_score > max_negative_score + score_diff;

        d["score"] = round4(best_score);
        bool neg_blocked = !passed && best_score >= threshold &&
                           max_negative_score > 0 &&
                           best_score <= max_negative_score + score_diff;
        if (in_cooldown) {
            d["status"] = "cooldown";
            d["would_hit"] = passed;
        } else if (passed) {
            d["status"] = "hit";
            d["hit"] = true;
        } else if (neg_blocked) {
            d["status"] = "negative_higher";
        } else {
            d["status"] = best_score < threshold ? "below_threshold" : "below_score_diff";
        }
        auto pt = pos_texts_.find(id);
        if (pt != pos_texts_.end()) {
            d["positive_list"] = pt->second;
            if (best_idx < pt->second.size()) d["best_positive"] = pt->second[best_idx];
        }
        d["positive_scores"] = positive_scores;
        if (max_negative_score > 0 || !negative_scores.empty()) {
            auto nt = neg_texts_.find(id);
            if (nt != neg_texts_.end()) d["negative_list"] = nt->second;
            d["negative_scores"] = negative_scores;
            d["max_negative_score"] = round4(max_negative_score);
        }
        dbg_rules.push_back(d);

        if (!passed || in_cooldown) continue;   // 冷却中只记分不触发，与原行为一致
        last_trigger_[id] = now;

        json m;
        m["rule_id"] = id;
        m["rule_name"] = getStr(rule, "name");
        m["positive"] = (pt != pos_texts_.end() && best_idx < pt->second.size())
                            ? pt->second[best_idx] : "";
        m["positive_list"] = pt != pos_texts_.end() ? json(pt->second) : json::array();
        m["positive_scores"] = positive_scores;
        auto nt = neg_texts_.find(id);
        m["negative_list"] = nt != neg_texts_.end() ? json(nt->second) : json::array();
        m["negative_scores"] = negative_scores;
        m["negative_higher"] = max_negative_score >= threshold &&
                               max_negative_score >= best_score;
        m["final_score"] = round4(best_score);
        matches.push_back(m);
    }
    return matches;
}

json Matcher::match(const ImageRGB & image) {
    std::lock_guard<std::mutex> lk(mu_);
    json empty = json::array();
    dbg_snapshot_ = json::object();   // 按图作废，防止上一帧串图
    if (rules_.empty() || rule_emb_.empty()) {
        if (cfg_.debug_capture) dbg_snapshot_ = json::object();
        return empty;
    }

    auto image_emb = vision_->embed(image);
    json dbg_rules = json::array();
    json matches = scoreImage(image_emb, dbg_rules);
    dbg_snapshot_["rules"] = dbg_rules;
    return matches;
}
