#include "matcher.h"
#include "attributes.h"
#include "reranker_scorer.h"
#include "text_embedder.h"
#include "util.h"
#include "vlm_describer.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <unordered_map>

// 预定义属性表见 attributes.h（与 bench_models 共用）

// ==================== 通用文本工具 ====================

// Python str.split() 的空白码点集合（近似：ASCII 空白 + 常见 Unicode 空白）
static bool isSpaceCp(uint32_t cp) {
    return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
           (cp >= 0x1C && cp <= 0x1F) || cp == 0x85 || cp == 0xA0 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x3000;
}

// " ".join(s.split())：任意空白折叠为单个空格，并去掉首尾空白
static std::string collapseWs(const std::string & s) {
    auto cps = utf8_to_codepoints(s);
    std::vector<uint32_t> out;
    bool pending = false;
    for (uint32_t cp : cps) {
        if (isSpaceCp(cp)) { if (!out.empty()) pending = true; continue; }
        if (pending) { out.push_back(' '); pending = false; }
        out.push_back(cp);
    }
    return codepoints_to_utf8(out);
}

// "".join(s.split())：去掉所有空白
static std::string stripAllWs(const std::string & s) {
    auto cps = utf8_to_codepoints(s);
    std::vector<uint32_t> out;
    for (uint32_t cp : cps) if (!isSpaceCp(cp)) out.push_back(cp);
    return codepoints_to_utf8(out);
}

// 按空白切词（等价 chunk.split()）
static std::vector<std::string> splitWs(const std::string & s) {
    std::vector<std::string> out;
    for (const auto & part : split_any(s, " \t\r\n")) {
        if (!part.empty()) out.push_back(part);
    }
    return out;
}

static double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

static std::string getStr(const json & j, const char * key, const std::string & def = "") {
    if (j.contains(key) && j[key].is_string()) return j[key].get<std::string>();
    return def;
}

// ==================== Matcher ====================

Matcher::Matcher(VlmDescriber * describer, TextEmbedder * embedder, const Config & cfg,
                 RerankerScorer * reranker)
    : vlm_(describer), emb_(embedder), reranker_(reranker), cfg_(cfg) {
    mode_ = cfg.match_mode;
    rr_max_pos_texts_ = cfg.reranker_max_pos_texts > 0 ? cfg.reranker_max_pos_texts : 1;
    rr_use_neg_ = cfg.reranker_use_negative;
    base_prompt_ = cfg.description_prompt;
    structured_max_ = cfg.structured_max_new_tokens;
    vlm_max_ = cfg.vlm_max_new_tokens;
    judge_max_ = cfg.rule_judge_max_new_tokens;
    rule_desc_max_ = cfg.rule_description_max_new_tokens;
    rule_prompt_max_texts_ = cfg.rule_prompt_max_texts;
    temp_ = cfg.vlm_temperature;
    rep_ = cfg.vlm_repetition_penalty;
    rule_desc_enabled_ = cfg.rule_description_enabled;
    rule_desc_verify_ = cfg.rule_description_verify;

    // 内置结构化提示词 = DEFAULT_STRUCTURED_PROMPT
    if (!cfg.structured_prompt.empty()) {
        structured_prompt_ = cfg.structured_prompt;
    } else {
        structured_prompt_ = defaultStructuredPrompt();
    }

    // 内置规则直判提示词 = DEFAULT_RULE_JUDGE_PROMPT；{rules} 缺失时回退
    static const std::string DEFAULT_RULE_JUDGE =
        "请判断图像中的行人是否符合以下每一条规则。\n"
        "{rules}\n"
        "输出要求：只输出符合的规则名称，多条之间用顿号「、」分隔；"
        "如果没有任何规则符合，只输出：不匹配\n"
        "除规则名称或「不匹配」外，不要输出任何其他文字、编号、标点或解释；"
        "只依据图像中清晰可见的内容判断，不要依据规则名称猜测。";
    judge_prompt_ = cfg.rule_judge_prompt;
    if (judge_prompt_.empty() || !contains(judge_prompt_, "{rules}")) judge_prompt_ = DEFAULT_RULE_JUDGE;

    description_prompt_ = (mode_ == "structured") ? structured_prompt_ : base_prompt_;
}

// ---- 文本工具实现 ----

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

std::string Matcher::lookupAttribute(const std::string & token) const {
    // _lookup_attribute：strip() 后去标点，再查两种写法
    std::string cleaned = trim_chars(trim(token), ".。:：`*\"'");
    if (cleaned.empty()) return "";
    const auto & tbl = attrLookupTable();
    auto it = tbl.find(tolower_ascii(cleaned));
    if (it != tbl.end()) return it->second;
    std::vector<uint32_t> nosep_cps;
    for (uint32_t cp : utf8_to_codepoints(cleaned)) {
        if (cp != '-' && cp != '_' && cp != ' ') nosep_cps.push_back(cp);   // re.sub("[-_ ]","")
    }
    it = tbl.find(tolower_ascii(codepoints_to_utf8(nosep_cps)));
    return it != tbl.end() ? it->second : std::string();
}

std::vector<std::string> Matcher::extractValidAttributes(const std::string & text) const {
    // _extract_valid_attributes：re.split(r"[,，、;；/|。.！!？?\s]+") 后按出现顺序去重
    std::vector<std::string> result;
    std::set<std::string> seen;
    for (const auto & token : split_any(text, ",，、;；/|。.！!？? \t\r\n")) {
        std::string attr = lookupAttribute(token);
        if (!attr.empty() && seen.insert(attr).second) result.push_back(attr);
    }
    return result;
}

std::optional<std::set<std::string>> Matcher::parseRuleAttributes(const std::string & text) const {
    // _parse_rule_attributes：全部 token 都是属性名才返回集合，否则 nullopt（回退语义匹配）
    std::set<std::string> attrs;
    bool found = false;
    for (const auto & chunk : split_any(text, ",，、;；/|")) {
        for (const auto & sub : splitWs(chunk)) {
            std::string cleaned = trim_chars(trim(sub), ".。:：`*\"'");
            if (cleaned.empty()) continue;
            std::string attr = lookupAttribute(cleaned);
            if (attr.empty()) return std::nullopt;
            attrs.insert(attr);
            found = true;
        }
    }
    if (!found) return std::nullopt;
    return attrs;
}

std::string Matcher::sanitizeRuleText(const std::string & text, size_t max_len) const {
    // _sanitize_rule_text：换行转空格 -> 折叠空白 -> 截断
    std::string s = text;
    replace_all(s, "\n", " ");
    return utf8_substr(collapseWs(s), max_len);
}

std::pair<std::string, std::string> Matcher::splitBilingual(const std::string & text) const {
    // _split_bilingual：找「中文：」「中 文 描 述:」等变体
    std::string t = trim(text);
    static const std::regex re("中\\s*文\\s*(?:描\\s*述)?\\s*[：:]");
    std::smatch m;
    if (std::regex_search(t, m, re)) {
        std::string zh = trim_chars(t.substr(m.position() + m.length()), " ：:，,。");
        std::string en = trim_chars(t.substr(0, m.position()), " ，。；;");
        return {en, zh};
    }
    return {t, ""};
}

std::string Matcher::cleanVlmDescription(const std::string & text, size_t max_chars) const {
    // _clean_vlm_description
    std::string s = collapseWs(text);
    static const char * markers[] = {
        "请注意", "以上信息", "免责声明", "如果存在", "请理解", "在提供此类内容时",
        "需要注意", "最后需要注意", "此外，在此情况下",
    };
    for (const char * marker : markers) {
        size_t pos = s.find(marker);
        if (pos != std::string::npos) s = trim(s.substr(0, pos));
    }
    s = trim_chars(s, " ，。；;");
    if (utf8_length(s) <= max_chars) return s;

    std::string clipped = utf8_substr(s, max_chars);
    static const char * seps[] = {"。", "；", "，", ","};
    for (const char * sep : seps) {
        size_t pos = clipped.rfind(sep);
        if (pos == std::string::npos) continue;
        if (utf8_length(clipped.substr(0, pos)) >= max_chars / 2) {
            return trim_chars(clipped.substr(0, pos), " ，。；;");
        }
    }
    return trim_chars(clipped, " ，。；;");
}

bool Matcher::isNoMatchDescription(const std::string & text) {
    // _is_no_match_description
    static const char * markers[] = {
        "不满足", "不符合", "未满足", "无法判断", "看不清", "不能确定", "无法确定",
        "不确定", "不存在", "没有符合", "无符合", "未看到", "没有看到", "没有穿",
        "并未", "不是", "可能符合",
    };
    std::string norm = stripAllWs(text);
    for (const char * m : markers) {
        if (norm.rfind(m, 0) == 0 || norm.find(m) != std::string::npos) return true;
    }
    return false;
}

// 按多字节分隔符切分（"。"/"，" 均为 3 字节 UTF-8）
static std::vector<std::string> split_utf8(const std::string & s, const char * delim) {
    std::vector<std::string> out;
    size_t start = 0, dlen = strlen(delim);
    for (;;) {
        size_t p = s.find(delim, start);
        out.push_back(s.substr(start, p == std::string::npos ? std::string::npos : p - start));
        if (p == std::string::npos) break;
        start = p + dlen;
    }
    return out;
}

// 微调模型惯于输出「由于视角的原因无法判断性别」「整体画面较为模糊」
// 「整体场景给人一种…的氛围」等 hedge/环境句，prompt 压不住（W8A8 指令
// 遵循弱）。这些分句进 BGE 向量会稀释衣着相似度，按分句剔除。
static bool isNoiseClause(const std::string & s) {
    static const char * bad[] = {
        "无法", "难以", "看不清", "看不出", "看不清楚", "视角",
        "模糊", "氛围", "感觉", "图像显示", "画面显示", "画质", "分辨率",
    };
    for (const char * b : bad)
        if (s.find(b) != std::string::npos) return true;
    return false;
}

// 分句匹配：描述按「。」→「，」切成短句，噪声分句剔除，每个分句独立编码；
// 规则文本（query）取各分句 cos 的最大值。离线对比（128 条真实描述）：
// 相比整段拼接单向量，期望规则命中率 34%→48%，正-自身负样本 margin +0.05→+0.13。
// max 会放大最近分句效应（「深色上衣」对「黑色」规则拿高分），
// 因此颜色类规则需用离线标定的更高阈值（黑衣服≈0.60）配合，不能沿用统一阈值。
// 姿态/携带类分句（走路/骑车）也保留参与匹配，规则不相关时分数自然低。
static std::vector<std::string> buildEmbeddingClauses(const std::string & description) {
    std::vector<std::string> out;
    for (const auto & seg : split_utf8(description, "。")) {
        for (const auto & clause : split_utf8(seg, "，")) {
            std::string c = trim_chars(clause, " ，。");
            if (c.empty() || isNoiseClause(c)) continue;
            out.push_back(std::move(c));
        }
    }
    if (out.empty()) out.push_back(description);
    return out;
}

std::string Matcher::extractVerifiedRuleDescription(const std::string & text) const {
    // _extract_verified_rule_description
    std::string s = cleanVlmDescription(text, 120);
    std::string norm = stripAllWs(s);
    if (isNoMatchDescription(s)) return "";
    if (norm.rfind("满足：", 0) == 0 || norm.rfind("满足:", 0) == 0) {
        std::string sep = contains(s, "：") ? "：" : ":";
        size_t pos = s.find(sep);
        return trim_chars(s.substr(pos + sep.size()), " ，。；;");
    }
    if (norm.rfind("满足", 0) == 0) {
        return trim(trim_chars(s, "满足:： ，。；;"));
    }
    return s;
}

// ---- 业务流程 ----

std::vector<std::string> Matcher::extractStructuredAttributes(const ImageRGB & img) {
    // extract_structured_attributes：Female 缺失时派生 Male
    std::string raw = vlm_->describe(img, structured_prompt_, structured_max_, temp_,
                                     cfg_.structured_repetition_penalty);
    auto attributes = extractValidAttributes(raw);
    // 复读守卫：正常行人上衣+下装至多 2~4 个颜色属性；一次输出 ≥8 个颜色 ≈ 在抄 prompt 里的
    // 属性列表（W8A8 低质量帧仍有概率触发，10/28 新告警实测）。判退化则整体丢弃，
    // 规则匹配自动回退描述向量路径。
    size_t colors = 0;
    for (const auto & a : attributes)
        if (a.size() > 1 && (a[0] == 'u' || a[0] == 'l') &&
            a != "ustriped_color" && a != "umulticolor" && a != "lstriped_color" && a != "lmulticolor")
            colors++;
    if (colors >= 8) {
        printf("[matcher] 结构化输出疑似复读(%zu 个颜色属性)，已丢弃: %s\n",
               colors, utf8_substr(raw, 120).c_str());
        return {};
    }
    if (std::find(attributes.begin(), attributes.end(), "Female") == attributes.end()) {
        attributes.push_back("Male");
    }
    return attributes;
}

std::string Matcher::describeImage(const ImageRGB & img) {
    // describe_image
    last_desc_zh_.clear();
    if (mode_ == "structured") {
        return join(extractStructuredAttributes(img), ",");
    }
    if (mode_ == "combined") {
        last_attrs_ = extractStructuredAttributes(img);   // 先属性（符号匹配+展示），再描述（语义匹配）
    }
    std::string description = vlm_->describe(img, description_prompt_, vlm_max_, temp_, rep_);
    auto [en, zh] = splitBilingual(description);
    last_desc_zh_ = zh.empty() ? "" : collapseWs(zh);
    return cleanVlmDescription(en, 1200);
}

std::string Matcher::buildRuleDescriptionPrompt(const json & rule,
                                                const std::vector<std::string> & positives_all,
                                                const std::vector<std::string> & negatives_all) const {
    // _build_rule_description_prompt
    std::string name = sanitizeRuleText(getStr(rule, "name", "当前规则"), 32);
    std::vector<std::string> positives, negatives;
    for (size_t i = 0; i < positives_all.size() && (int)i < rule_prompt_max_texts_; i++)
        positives.push_back(sanitizeRuleText(positives_all[i], 48));
    for (size_t i = 0; i < negatives_all.size() && (int)i < rule_prompt_max_texts_; i++)
        negatives.push_back(sanitizeRuleText(negatives_all[i], 48));

    std::string prompt =
        "请判断图像中的这个行人是否满足下面这条规则。"
        "输出格式只能二选一：\n"
        "满足：<只写一句与这条规则相关的行人描述>\n"
        "不满足\n"
        "规则名称：" + name + "\n";
    if (!positives.empty()) prompt += "目标特征：" + join(positives, "、") + "\n";
    if (!negatives.empty()) prompt += "排除/区分特征：" + join(negatives, "、") + "\n";
    prompt +=
        "输出要求：只写一句中文事实描述，重点说明与该规则相关的可见特征；"
        "可以包含性别估计、衣着颜色款式、头盔/帽子、携带物、动作或交通工具等与规则有关的信息；"
        "不要输出无关外观，不要照抄规则原文，不要输出匹配分数，不要输出“符合规则/不符合规则”等结论；"
        "不要因为规则名称或目标特征而编造图像中不存在的内容；"
        "不要输出请注意、免责声明、推理过程或额外解释。";
    return prompt;
}

std::string Matcher::describeRuleMatch(const json & rule, const std::vector<std::string> & pos,
                                       const std::vector<std::string> & neg,
                                       const std::string & fallback, const ImageRGB & img) {
    // describe_rule_match
    if (!rule_desc_enabled_) return fallback;
    std::string prompt = buildRuleDescriptionPrompt(rule, pos, neg);
    std::string description =
        extractVerifiedRuleDescription(vlm_->describe(img, prompt, rule_desc_max_, temp_, rep_));
    if (rule_desc_verify_ && description.empty()) return "";
    return description.empty() ? fallback : description;
}

std::string Matcher::buildRuleJudgePrompt(const std::vector<json> & rules) const {
    // _build_rule_judge_prompt
    std::vector<std::string> lines;
    for (size_t i = 0; i < rules.size(); i++) {
        const json & rule = rules[i];
        std::string name = sanitizeRuleText(getStr(rule, "name"), 32);
        std::vector<std::string> positives;
        auto it = pos_texts_.find(getStr(rule, "id"));
        if (it != pos_texts_.end()) {
            for (size_t k = 0; k < it->second.size() && (int)k < rule_prompt_max_texts_; k++)
                positives.push_back(sanitizeRuleText(it->second[k], 48));
        }
        std::string line = std::to_string(i + 1) + ". " + name;
        if (!positives.empty()) line += "（目标特征：" + join(positives, "、") + "）";
        lines.push_back(line);
    }
    std::string out = judge_prompt_;
    // replace("{rules}", ...) 只替换一次
    size_t pos = out.find("{rules}");
    if (pos != std::string::npos) out.replace(pos, 7, join(lines, "\n"));
    return out;
}

std::vector<json> Matcher::parseMatchedRuleNames(const std::string & output,
                                                 const std::vector<json> & candidates) const {
    // _parse_matched_rule_names：长名优先回认，避免短名是长名子串时重复命中
    std::string remaining = stripAllWs(output);
    if (remaining.empty()) return {};

    std::vector<size_t> order(candidates.size());
    std::vector<std::string> names(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
        order[i] = i;
        names[i] = stripAllWs(sanitizeRuleText(getStr(candidates[i], "name"), 32));
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return utf8_length(names[a]) > utf8_length(names[b]);
    });

    std::vector<size_t> hits;
    for (size_t idx : order) {
        if (names[idx].empty()) continue;
        size_t pos = remaining.find(names[idx]);
        if (pos != std::string::npos) {
            remaining.erase(pos, names[idx].size());
            hits.push_back(idx);
        }
    }
    std::sort(hits.begin(), hits.end());   // matched.sort(key=candidates.index)
    std::vector<json> matched;
    for (size_t idx : hits) matched.push_back(candidates[idx]);
    return matched;
}

// 规则公共字段读取
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

void Matcher::updateRules(const json & rules) {
    std::lock_guard<std::mutex> lk(mu_);
    // update_rules：只保留 enabled 规则；预计算正/负向量与属性解析
    rules_.clear();
    if (rules.is_array()) {
        for (const auto & r : rules) {
            if (r.is_object() && ruleBool(r, "enabled", true)) rules_.push_back(r);
        }
    }
    rule_emb_.clear(); neg_emb_.clear();
    pos_texts_.clear(); neg_texts_.clear();
    attr_pos_.clear(); attr_neg_.clear();

    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        // positive 为空时回退 [text]（python: rule.get('positive') or [rule.get('text','')]）
        json pos_json = rule.contains("positive") ? rule["positive"] : json();
        std::vector<std::string> positives = normalizeTexts(pos_json);
        if (positives.empty()) positives = normalizeTexts(json::array({getStr(rule, "text")}));
        std::vector<std::string> negatives =
            normalizeTexts(rule.contains("negative") ? rule["negative"] : json::array());

        pos_texts_[id] = positives;
        neg_texts_[id] = negatives;
        if (mode_ == "rule_direct" || mode_ == "reranker") continue;   // 直判/重排模式无需向量

        auto embed = [&](const std::vector<std::string> & texts) {
            EmbMat m;
            if (!texts.empty()) {
                // 非对称检索：规则文本是 query，加 BGE 指令前缀；图像描述侧不加
                m.v = emb_->encode(texts, cfg_.embed_query_prefix);
                m.n = texts.size();
            }
            return m;
        };
        rule_emb_[id] = embed(positives);
        neg_emb_[id] = embed(negatives);

        if (mode_ == "structured" || mode_ == "combined") {
            std::optional<std::set<std::string>> merged = std::set<std::string>();
            bool ok = !positives.empty();
            for (const auto & t : positives) {
                auto parsed = parseRuleAttributes(t);
                if (!parsed) { ok = false; break; }
                merged->insert(parsed->begin(), parsed->end());
            }
            attr_pos_[id] = ok ? merged : std::optional<std::set<std::string>>{};

            std::set<std::string> neg_attrs;
            for (const auto & t : negatives) {
                auto parsed = parseRuleAttributes(t);
                if (parsed) neg_attrs.insert(parsed->begin(), parsed->end());
            }
            attr_neg_[id] = neg_attrs;
        }
    }

    description_prompt_ = (mode_ == "structured") ? structured_prompt_ : base_prompt_;
}

json Matcher::matchRuleDirect(const ImageRGB & img, double now) {
    // match_rule_direct
    std::vector<json> eligible;
    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        double interval = ruleNumber(rule, "interval", 60);
        auto it = last_trigger_.find(id);
        if (it != last_trigger_.end() && now - it->second < interval) continue;
        eligible.push_back(rule);
    }
    json matches = json::array();
    if (eligible.empty()) return matches;

    std::string prompt = buildRuleJudgePrompt(eligible);
    std::string output = vlm_->describe(img, prompt, judge_max_, temp_, rep_);
    std::string judge_output = collapseWs(output);
    auto matched = parseMatchedRuleNames(output, eligible);
    if (matched.empty()) {
        if (!judge_output.empty()) printf("规则直判未命中（模型输出: %s）\n", judge_output.c_str());
        return matches;
    }

    for (const auto & rule : matched) {
        std::string id = getStr(rule, "id");
        last_trigger_[id] = now;
        std::vector<std::string> pos = pos_texts_.count(id) ? pos_texts_[id] : std::vector<std::string>{};
        std::vector<std::string> neg = neg_texts_.count(id) ? neg_texts_[id] : std::vector<std::string>{};
        std::string rule_text = trim(getStr(rule, "text"));
        std::string name = getStr(rule, "name");

        json m;
        m["rule_id"] = id;
        m["rule_name"] = name;
        m["positive"] = !pos.empty() ? pos[0] : (rule_text.empty() ? name : rule_text);
        m["positive_list"] = pos;
        m["negative_list"] = neg;
        m["positive_scores"] = json::array();
        m["negative_scores"] = json::array();
        m["final_score"] = 1.0;
        m["negative_higher"] = false;
        m["description"] = rule_text.empty() ? (!pos.empty() ? pos[0] : name) : rule_text;
        m["base_description"] = judge_output;
        m["description_zh"] = "";
        m["attributes"] = json::array();
        matches.push_back(m);
    }
    return matches;
}

json Matcher::matchReranker(const ImageRGB & img, double now) {
    // match_mode=reranker：ROI 图 × 规则文本 直接过 Qwen3-VL-Reranker 打分。
    // final_score = max(positive 文本分)；threshold/score_diff/interval 语义与向量模式一致。
    json matches = json::array();
    if (!reranker_ || !reranker_->ok() || rules_.empty()) return matches;

    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        double interval = ruleNumber(rule, "interval", 60);
        auto lt = last_trigger_.find(id);
        if (lt != last_trigger_.end() && now - lt->second < interval) continue;

        std::vector<std::string> pos = pos_texts_.count(id) ? pos_texts_[id] : std::vector<std::string>{};
        if (pos.empty()) continue;
        if ((int)pos.size() > rr_max_pos_texts_) pos.resize(rr_max_pos_texts_);

        double best_positive_score = 0.0;
        std::string best_positive_text;
        json positive_scores = json::array();
        for (const auto & t : pos) {
            double s = reranker_->score(img, t);
            if (s < 0) {   // 单条失败：整帧放弃（多为 NPU/内存异常，下帧再来）
                fprintf(stderr, "reranker 打分失败: rule=%s text=%s\n", getStr(rule, "name").c_str(), t.c_str());
                return matches;
            }
            positive_scores.push_back(round4(s));
            if (positive_scores.size() == 1 || s > best_positive_score) {
                best_positive_score = s;
                best_positive_text = t;
            }
        }

        double max_negative_score = 0.0;
        json negative_scores = json::array();
        if (rr_use_neg_) {
            std::vector<std::string> neg = neg_texts_.count(id) ? neg_texts_[id] : std::vector<std::string>{};
            if ((int)neg.size() > rr_max_pos_texts_) neg.resize(rr_max_pos_texts_);
            for (const auto & t : neg) {
                double s = reranker_->score(img, t);
                if (s < 0) return matches;
                negative_scores.push_back(round4(s));
                if (s > max_negative_score) max_negative_score = s;
            }
        }

        double score_diff = ruleNumber(rule, "score_diff", 0);
        double threshold = ruleNumber(rule, "threshold", 0.5);
        if (!(best_positive_score >= threshold && best_positive_score > max_negative_score + score_diff))
            continue;

        last_trigger_[id] = now;
        bool negative_higher = (max_negative_score >= threshold && max_negative_score >= best_positive_score);

        json m;
        m["rule_id"] = id;
        m["rule_name"] = getStr(rule, "name");
        m["positive"] = best_positive_text;
        m["positive_list"] = pos;
        m["negative_list"] = neg_texts_.count(id) ? neg_texts_[id] : std::vector<std::string>{};
        m["positive_scores"] = positive_scores;
        m["negative_scores"] = negative_scores;
        m["final_score"] = round4(best_positive_score);
        m["negative_higher"] = negative_higher;
        m["description"] = best_positive_text;   // reranker 无描述产出
        m["base_description"] = "";
        m["description_zh"] = "";
        m["attributes"] = json::array();
        matches.push_back(m);
    }
    return matches;
}

json Matcher::match(const ImageRGB & image) {
    std::lock_guard<std::mutex> lk(mu_);
    // match：符号匹配（属性规则）+ BGE 向量匹配（自然语言规则）
    json empty = json::array();
    if (rules_.empty()) return empty;

    double now = std::chrono::duration<double>(
                     std::chrono::system_clock::now().time_since_epoch()).count();
    if (mode_ == "rule_direct") return matchRuleDirect(image, now);
    if (mode_ == "reranker") return matchReranker(image, now);
    if (rule_emb_.empty()) return empty;

    std::string description = describeImage(image);
    if (description.empty()) {
        printf("VLM描述为空，跳过本次匹配\n");
        return empty;
    }

    std::set<std::string> detected;
    if (mode_ == "structured") {
        for (const auto & a : split_any(description, ",")) if (!a.empty()) detected.insert(a);
    } else if (mode_ == "combined") {
        detected.insert(last_attrs_.begin(), last_attrs_.end());
    }

    // 结构化/混合模式：全部规则都能符号匹配时跳过图像描述向量计算
    bool needs_embedding = true;
    if (mode_ == "structured" || mode_ == "combined") {
        needs_embedding = false;
        for (const auto & rule : rules_) {
            auto it = attr_pos_.find(getStr(rule, "id"));
            if (it == attr_pos_.end() || !it->second.has_value()) { needs_embedding = true; break; }
        }
    }
    // 分句矩阵：行 = 分句向量（BGE doc 侧，不加前缀）；规则文本对分句取 max
    std::vector<float> image_emb;
    size_t image_rows = 0;
    if (needs_embedding) {
        auto clauses = buildEmbeddingClauses(description);
        image_emb = emb_->encode(clauses);
        image_rows = clauses.size();
    }

    json matches = json::array();
    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        double interval = ruleNumber(rule, "interval", 60);
        auto lt = last_trigger_.find(id);
        if (lt != last_trigger_.end() && now - lt->second < interval) continue;

        double best_positive_score = 0.0;
        double max_negative_score = 0.0;
        std::string best_positive_text;
        json positive_scores = json::array(), negative_scores = json::array();

        auto ap = attr_pos_.find(id);
        bool symbolic = ap != attr_pos_.end() && ap->second.has_value() && !ap->second->empty();
        if (symbolic) {
            // 符号匹配：得分为正样本属性覆盖率；任一负样本属性命中记 1.0
            std::vector<std::string> pos_attrs(ap->second->begin(), ap->second->end());  // set 已有序
            std::vector<std::string> present;
            for (const auto & a : pos_attrs) {
                bool has = detected.count(a) > 0;
                positive_scores.push_back(has ? 1.0 : 0.0);
                if (has) present.push_back(a);
            }
            best_positive_score = (double)present.size() / (double)pos_attrs.size();
            best_positive_text = present.empty() ? pos_attrs[0] : join(present, ",");
            auto an = attr_neg_.find(id);
            if (an != attr_neg_.end()) {
                for (const auto & a : an->second) {
                    if (detected.count(a)) { max_negative_score = 1.0; break; }
                }
            }
        } else {
            auto pe = rule_emb_.find(id);
            if (pe == rule_emb_.end() || pe->second.n == 0) continue;
            const EmbMat & pos = pe->second;
            size_t best_idx = 0;
            float best_raw = -1e30f;
            for (size_t r = 0; r < pos.n; r++) {
                const float * row = pos.v.data() + r * pos.dim;
                float s = image_rows ? -1e30f : 0.0f;
                for (size_t c = 0; c < image_rows; c++) {   // 对每个分句取 max
                    const float * cv = image_emb.data() + c * pos.dim;
                    float sc = 0;
                    for (size_t k = 0; k < pos.dim; k++) sc += row[k] * cv[k];
                    if (sc > s) s = sc;
                }
                positive_scores.push_back(round4(s));
                if (s > best_raw) { best_raw = s; best_idx = r; }
            }
            best_positive_score = best_raw;
            auto pt = pos_texts_.find(id);
            if (pt != pos_texts_.end() && best_idx < pt->second.size())
                best_positive_text = pt->second[best_idx];

            auto ne = neg_emb_.find(id);
            if (ne != neg_emb_.end() && ne->second.n > 0) {
                const EmbMat & negm = ne->second;
                float max_raw = -1e30f;
                for (size_t r = 0; r < negm.n; r++) {
                    const float * row = negm.v.data() + r * negm.dim;
                    float s = image_rows ? -1e30f : 0.0f;
                    for (size_t c = 0; c < image_rows; c++) {   // 同样对分句取 max
                        const float * cv = image_emb.data() + c * negm.dim;
                        float sc = 0;
                        for (size_t k = 0; k < negm.dim; k++) sc += row[k] * cv[k];
                        if (sc > s) s = sc;
                    }
                    negative_scores.push_back(round4(s));
                    if (s > max_raw) max_raw = s;
                }
                max_negative_score = max_raw;
            }
        }

        double score_diff = ruleNumber(rule, "score_diff", 0);
        double threshold = ruleNumber(rule, "threshold", 0.5);
        if (!(best_positive_score >= threshold && best_positive_score > max_negative_score + score_diff))
            continue;

        last_trigger_[id] = now;
        std::vector<std::string> pos = pos_texts_.count(id) ? pos_texts_[id] : std::vector<std::string>{};
        std::vector<std::string> neg = neg_texts_.count(id) ? neg_texts_[id] : std::vector<std::string>{};
        bool negative_higher = (max_negative_score >= threshold && max_negative_score >= best_positive_score);

        std::string rule_description = describeRuleMatch(rule, pos, neg, description, image);
        if (rule_desc_enabled_ && rule_desc_verify_ && rule_description.empty()) continue;

        json m;
        m["rule_id"] = id;
        m["rule_name"] = getStr(rule, "name");
        m["positive"] = best_positive_text;
        m["positive_list"] = pos;
        m["negative_list"] = neg;
        m["positive_scores"] = positive_scores;
        m["negative_scores"] = negative_scores;
        m["final_score"] = round4(best_positive_score);
        m["negative_higher"] = negative_higher;
        m["description"] = rule_description;
        m["base_description"] = description;
        m["description_zh"] = last_desc_zh_;
        if (!detected.empty()) {
            if (mode_ == "structured") {
                json arr = json::array();   // description.split(",")
                for (const auto & a : split_any(description, ",")) if (!a.empty()) arr.push_back(a);
                m["attributes"] = arr;
            } else {
                m["attributes"] = last_attrs_;
            }
        } else {
            m["attributes"] = json::array();
        }
        matches.push_back(m);
    }
    return matches;
}
