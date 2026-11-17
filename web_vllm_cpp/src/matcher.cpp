#include "matcher.h"
#include "text_embedder.h"
#include "util.h"
#include "vlm_describer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>
#include <unordered_map>

// ==================== 预定义属性表（processor.py PEDESTRIAN_ATTRIBUTES） ====================

static const std::vector<std::string> PEDESTRIAN_ATTRIBUTES = {
    "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Side", "Back",
    "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
    "ublack", "ugray", "ublue", "ugreen", "uwhite", "upurple", "ured", "ubrown",
    "uyellow", "upink", "uorange", "ubeige", "ustriped_color", "umulticolor",
    "lwhite", "lpink", "lred", "lgreen", "lyellow", "lpurple", "lbrown", "lblack",
    "lorange", "lblue", "lgray", "lbeige", "lstriped_color", "lmulticolor",
};

// 小写 / 去分隔符两种写法都能命中；Male 由 Female 缺失时派生
static const std::unordered_map<std::string, std::string> & attrLookup() {
    static const std::unordered_map<std::string, std::string> tbl = [] {
        std::unordered_map<std::string, std::string> m;
        for (const auto & attr : PEDESTRIAN_ATTRIBUTES) {
            m[tolower_ascii(attr)] = attr;
            std::string nosep;
            for (char c : attr) if (c != '-' && c != '_' && c != ' ') nosep += c;
            m[tolower_ascii(nosep)] = attr;
        }
        m["male"] = "Male";
        return m;
    }();
    return tbl;
}

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

Matcher::Matcher(VlmDescriber * describer, TextEmbedder * embedder, const Config & cfg)
    : vlm_(describer), emb_(embedder), cfg_(cfg) {
    mode_ = cfg.match_mode;
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
        structured_prompt_ =
            "请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），"
            "只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。\n"
            "预定义属性列表：" + join(PEDESTRIAN_ATTRIBUTES, ",");
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
    const auto & tbl = attrLookup();
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
// 「整体场景给人一种…的氛围」等 hedge/环境句，prompt 压不住。这些分句进 BGE
// 向量会稀释衣着相似度，按分句剔除。
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
// 规则文本（query）取各分句 cos 的最大值。注意：rknn 线上 2026-11-04 已改回
// 「规则短句(分句) vs 描述整句」单向量口径，本函数与 web_vllm_rknn 同步保留备用
// （勿删，历史口径参考；启用需重新标定规则阈值，分数分布整体上移 0.07~0.11）。
[[maybe_unused]] static std::vector<std::string> buildEmbeddingClauses(const std::string & description) {
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

// 只有含衣着/携带关键词的句子进 BGE 匹配。长描述的整体相似度被话题重叠
// （行人/男性/行走/环境）主导，颜色↔衣物的绑定关系在单向量里被稀释，
// 实测「白上衣黑短裤」对「穿黑色上衣的男人」也能拿 0.56 分（阈值之上）。
// 摘出“穿了什么”短句后与规则短文本对称比较，同/非同色拉开明显
// （实测 FP margin -0.001 / TP margin +0.104）。没摘到任何句则回退全文，
// 避免丢姿态/携带类规则的匹配依据。
static std::string buildEmbeddingText(const std::string & description) {
    static const char * kws[] = {
        "衣", "裤", "裙", "袖", "穿", "戴", "帽", "鞋", "袜", "眼镜",
        "外套", "大衣", "风衣", "夹克", "衬衫", "西装", "背包", "背着", "拎", "提",
    };
    std::string kept;
    size_t start = 0;
    for (;;) {
        size_t pos = description.find("。", start);
        std::string seg = description.substr(
            start, pos == std::string::npos ? std::string::npos : pos - start);
        for (const char * kw : kws) {
            if (seg.find(kw) != std::string::npos) {
                if (!kept.empty()) kept += "。";
                kept += seg;
                break;
            }
        }
        if (pos == std::string::npos) break;
        start = pos + 3;   // strlen("。")
    }
    kept = trim_chars(kept, " 。");
    return kept.empty() ? description : kept;
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
    std::string raw = vlm_->describe(img, structured_prompt_, structured_max_, temp_, rep_);
    auto attributes = extractValidAttributes(raw);
    if (std::find(attributes.begin(), attributes.end(), "Female") == attributes.end()) {
        attributes.push_back("Male");
    }
    return attributes;
}

Matcher::DescResult Matcher::describeImage(const ImageRGB & img, const std::string & prompt) {
    // describe_image：结果走局部返回而非共享成员——调用方（match）不持 mu_ 也能安全生成
    DescResult r;
    if (mode_ == "structured") {
        r.attrs = extractStructuredAttributes(img);
        r.text = join(r.attrs, ",");
        return r;
    }
    if (mode_ == "combined") {
        r.attrs = extractStructuredAttributes(img);   // 先属性（符号匹配+展示），再描述（语义匹配）
    }
    std::string description = vlm_->describe(img, prompt, vlm_max_, temp_, rep_);
    auto [en, zh] = splitBilingual(description);
    r.zh = zh.empty() ? "" : collapseWs(zh);
    r.text = cleanVlmDescription(en, 1200);
    return r;
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
    // update_rules：只保留 enabled 规则；预计算正/负向量与属性解析。
    // 卡顿治理两条（同 web_vllm_rknn）：
    //   ① vec_cache_ 文本→向量缓存：只有新文本才进 BGE（调阈值/改名/开关 = 零编码，微秒级）；
    //   ② 编码在 mu_ 锁外进行——match() 的 VLM 描述可达数秒，旧版整表重编码挤在锁内，
    //      HTTP 保存规则要等“当前帧描述完 + 全部编码”=整站等。
    struct RuleTexts {
        std::string id;
        std::vector<std::string> pos, neg;
    };
    std::vector<json> new_rules;
    std::vector<RuleTexts> rts;
    std::set<std::string> want;
    if (rules.is_array()) {
        for (const auto & rule : rules) {
            if (!(rule.is_object() && ruleBool(rule, "enabled", true))) continue;
            RuleTexts rt;
            rt.id = getStr(rule, "id");
            // positive 为空时回退 [text]（python: rule.get('positive') or [rule.get('text','')]）
            json pos_json = rule.contains("positive") ? rule["positive"] : json();
            rt.pos = normalizeTexts(pos_json);
            if (rt.pos.empty()) rt.pos = normalizeTexts(json::array({getStr(rule, "text")}));
            rt.neg = normalizeTexts(rule.contains("negative") ? rule["negative"] : json());
            for (const auto & t : rt.pos) want.insert(t);
            for (const auto & t : rt.neg) want.insert(t);
            new_rules.push_back(rule);
            rts.push_back(std::move(rt));
        }
    }

    // 锁内比对缓存，只收集缺失文本（直判模式无需向量）
    std::vector<std::string> todo;
    if (mode_ != "rule_direct") {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto & t : want)
            if (!vec_cache_.count(t)) todo.push_back(t);
    }

    // 锁外编码缺失文本（TextEmbedder::encode 自带锁，与在跑的描述/打分互不干涉）
    if (!todo.empty()) {
        std::vector<float> vecs = emb_->encode(todo);
        size_t dim = vecs.empty() ? 0 : vecs.size() / todo.size();
        std::lock_guard<std::mutex> lk(mu_);
        if (dim) {
            for (size_t i = 0; i < todo.size() && (i + 1) * dim <= vecs.size(); i++)
                vec_cache_[todo[i]] = std::vector<float>(vecs.begin() + i * dim,
                                                         vecs.begin() + (i + 1) * dim);
        }
    }

    // 锁内只做查表拼装换表（微秒级）
    {
        std::lock_guard<std::mutex> lk(mu_);
        rules_ = std::move(new_rules);
        rule_emb_.clear(); neg_emb_.clear();
        pos_texts_.clear(); neg_texts_.clear();
        attr_pos_.clear(); attr_neg_.clear();

        for (const auto & rt : rts) {
            const std::string & id = rt.id;
            pos_texts_[id] = rt.pos;
            neg_texts_[id] = rt.neg;
            if (mode_ == "rule_direct") continue;   // 直判模式无需向量

            EmbMat m;
            for (const auto & t : rt.pos) {
                auto it = vec_cache_.find(t);
                if (it == vec_cache_.end()) continue;
                m.v.insert(m.v.end(), it->second.begin(), it->second.end());
                m.n++;
            }
            rule_emb_[id] = std::move(m);

            // 负样本：与正样本同口径，打分时 pos 须 > max(neg)+score_diff 才告警
            EmbMat nm;
            for (const auto & t : rt.neg) {
                auto it = vec_cache_.find(t);
                if (it == vec_cache_.end()) continue;
                nm.v.insert(nm.v.end(), it->second.begin(), it->second.end());
                nm.n++;
            }
            neg_emb_[id] = std::move(nm);

            if (mode_ == "structured" || mode_ == "combined") {
                std::optional<std::set<std::string>> merged = std::set<std::string>();
                bool ok = !rt.pos.empty();
                for (const auto & t : rt.pos) {
                    auto parsed = parseRuleAttributes(t);
                    if (!parsed) { ok = false; break; }
                    merged->insert(parsed->begin(), parsed->end());
                }
                attr_pos_[id] = ok ? merged : std::optional<std::set<std::string>>{};

                std::set<std::string> neg_attrs;
                for (const auto & t : rt.neg) {
                    auto parsed = parseRuleAttributes(t);
                    if (parsed) neg_attrs.insert(parsed->begin(), parsed->end());
                }
                attr_neg_[id] = neg_attrs;
            }
        }

        description_prompt_ = (mode_ == "structured") ? structured_prompt_ : base_prompt_;
    }
}

// ---- 调试捕获 ----

json Matcher::lastDebugSnapshot() {
    std::lock_guard<std::mutex> lk(mu_);
    return dbg_snapshot_;
}

json Matcher::describeForDebug(const ImageRGB & image) {
    // VLM 生成（秒级）在 mu_ 锁外：旧版持锁跑完整描述，该设备的 updateRules
    // 要排队等描述结束；refreshRules 又连锁调 updateRules，会拖住
    // /api/monitoring/status、start/stop 等接口，表现为整站假死。
    std::string prompt;
    {
        std::lock_guard<std::mutex> lk(mu_);
        prompt = description_prompt_;
    }
    DescResult dr = describeImage(image, prompt);
    json snap;
    snap["description"] = dr.text;
    snap["description_zh"] = dr.zh;
    json attrs = json::array();
    for (const auto & a : dr.attrs) attrs.push_back(a);
    snap["attributes"] = attrs;
    return snap;
}

json Matcher::matchRuleDirect(const ImageRGB & img, double now) {
    // match_rule_direct：eligible+prompt 锁内组好，规则直判 VLM（秒级）挪到锁外，
    // 解析/记冷却/写调试快照回到锁内——与 match() 同款锁纪律
    std::vector<json> eligible;
    std::string prompt;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto & rule : rules_) {
            std::string id = getStr(rule, "id");
            double interval = ruleNumber(rule, "interval", 60);
            auto it = last_trigger_.find(id);
            if (it != last_trigger_.end() && now - it->second < interval) continue;
            eligible.push_back(rule);
        }
        if (eligible.empty()) return json::array();
        prompt = buildRuleJudgePrompt(eligible);
    }

    std::string output = vlm_->describe(img, prompt, judge_max_, temp_, rep_);
    json matches = json::array();
    std::lock_guard<std::mutex> lk(mu_);
    std::string judge_output = collapseWs(output);
    auto matched = parseMatchedRuleNames(output, eligible);
    if (matched.empty()) {
        if (!judge_output.empty()) printf("规则直判未命中（模型输出: %s）\n", judge_output.c_str());
        // 调试快照：直判模式没有描述，记录直判输出供 /debug 页查看
        dbg_snapshot_["description"] = judge_output;
        dbg_snapshot_["description_zh"] = "";
        dbg_snapshot_["attributes"] = json::array();
        return matches;
    }

    // 逐规则明细：命中/未命中都进调试页
    {
        json dbg_rules = json::array();
        for (const auto & rule : eligible) {
            bool hit = false;
            for (const auto & m : matched)
                if (getStr(m, "id") == getStr(rule, "id")) { hit = true; break; }
            json d;
            d["rule_name"] = getStr(rule, "name");
            d["type"] = "judge";
            d["score"] = hit ? 1.0 : 0.0;
            d["threshold"] = 0.5;
            d["status"] = hit ? "hit" : "not_judged";
            d["hit"] = hit;
            dbg_rules.push_back(d);
        }
        dbg_snapshot_["rules"] = dbg_rules;
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

json Matcher::match(const ImageRGB & image) {
    // match：符号匹配（属性规则）+ BGE 向量匹配（自然语言规则）
    json empty = json::array();
    std::string prompt;
    bool rules_empty, direct;
    {
        std::lock_guard<std::mutex> lk(mu_);
        dbg_snapshot_ = json::object();   // 调试快照按图作废，防止上一帧串图
        rules_empty = rules_.empty();
        direct = (mode_ == "rule_direct");
        if (!direct && !rules_empty && rule_emb_.empty()) return empty;
        prompt = description_prompt_;
    }
    if (rules_empty) {
        // 调试捕获开启且无规则：仍生成一次描述供 /debug 展示（生成在 mu_ 锁外）
        if (cfg_.debug_capture) {
            json snap = describeForDebug(image);
            std::lock_guard<std::mutex> lk(mu_);
            dbg_snapshot_ = std::move(snap);
        }
        return empty;
    }

    double now = std::chrono::duration<double>(
                     std::chrono::system_clock::now().time_since_epoch()).count();
    if (direct) return matchRuleDirect(image, now);

    // VLM 描述（秒级）不持 matcher mu_：刷新规则时的 updateRules 不必排队等描述
    // （这正是「更新规则整个平台卡住」的根因）；多设备并发生成由 VlmDescriber 内部锁兜底
    DescResult dr = describeImage(image, prompt);

    // 调试快照：本帧描述/中文描述/属性（未命中规则的检测也能在 /debug 页看到）
    {
        std::lock_guard<std::mutex> lk(mu_);
        dbg_snapshot_["description"] = dr.text;
        dbg_snapshot_["description_zh"] = dr.zh;
        json attrs = json::array();
        for (const auto & a : dr.attrs) attrs.push_back(a);
        dbg_snapshot_["attributes"] = attrs;
    }

    std::string description = dr.text;
    if (description.empty()) {
        printf("VLM描述为空，跳过本次匹配\n");
        return empty;
    }

    std::set<std::string> detected;
    if (mode_ == "structured") {
        for (const auto & a : split_any(description, ",")) if (!a.empty()) detected.insert(a);
    } else if (mode_ == "combined") {
        detected.insert(dr.attrs.begin(), dr.attrs.end());
    }

    // 结构化/混合模式：全部规则都能符号匹配时跳过图像描述向量计算
    bool needs_embedding = true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (mode_ == "structured" || mode_ == "combined") {
            needs_embedding = false;
            for (const auto & rule : rules_) {
                auto it = attr_pos_.find(getStr(rule, "id"));
                if (it == attr_pos_.end() || !it->second.has_value()) { needs_embedding = true; break; }
            }
        }
    }
    std::vector<float> image_emb;
    if (needs_embedding)
        image_emb = emb_->encode_one(buildEmbeddingText(description));   // 锁外，BGE 自带锁

    json matches = json::array();
    // 调试捕获：逐规则得分明细（未命中/冷却中的规则也算分、也记录），match 末尾挂到快照
    json dbg_rules = json::array();
    // 规则二次核验的 VLM 生成也挪到锁外：先收集命中项，生成完再回填/回滚
    struct DescJob { size_t match_idx; size_t dbg_idx; json rule;
                     std::vector<std::string> pos, neg; };
    std::vector<DescJob> desc_jobs;
    {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto & rule : rules_) {
        std::string id = getStr(rule, "id");
        double interval = ruleNumber(rule, "interval", 60);
        auto lt = last_trigger_.find(id);
        bool in_cooldown = lt != last_trigger_.end() && now - lt->second < interval;

        // 调试明细：本规则的静态信息先填好，得分算完再补 score/status
        json d;
        d["rule_name"] = getStr(rule, "name");
        d["threshold"] = round4(ruleNumber(rule, "threshold", 0.5));
        d["score_diff"] = round4(ruleNumber(rule, "score_diff", 0));
        d["hit"] = false;

        double best_positive_score = 0.0;
        double max_negative_score = 0.0;
        std::string best_positive_text;
        json positive_scores = json::array(), negative_scores = json::array();

        auto ap = attr_pos_.find(id);
        bool symbolic = ap != attr_pos_.end() && ap->second.has_value() && !ap->second->empty();
        d["type"] = symbolic ? "symbolic" : "semantic";
        if (symbolic) {
            // 符号匹配：得分为正样本属性覆盖率；任一负样本属性命中记 1.0
            std::vector<std::string> pos_attrs(ap->second->begin(), ap->second->end());  // set 已有序
            std::vector<std::string> present, missing;
            for (const auto & a : pos_attrs) {
                bool has = detected.count(a) > 0;
                positive_scores.push_back(has ? 1.0 : 0.0);
                (has ? present : missing).push_back(a);
            }
            best_positive_score = (double)present.size() / (double)pos_attrs.size();
            best_positive_text = present.empty() ? pos_attrs[0] : join(present, ",");
            auto an = attr_neg_.find(id);
            if (an != attr_neg_.end()) {
                for (const auto & a : an->second) {
                    if (detected.count(a)) { max_negative_score = 1.0; break; }
                }
            }
            d["expected"] = pos_attrs;    // 规则要求的属性
            d["hit_attrs"] = present;     // 本帧检出的
            d["missing_attrs"] = missing; // 未命中的属性（调试页高亮）
        } else {
            auto pe = rule_emb_.find(id);
            if (pe == rule_emb_.end() || pe->second.n == 0) {
                d["status"] = "no_embedding";   // 规则无正样本向量，不打分
                dbg_rules.push_back(d);
                continue;
            }
            const EmbMat & pos = pe->second;
            size_t best_idx = 0;
            float best_raw = -1e30f;
            for (size_t r = 0; r < pos.n; r++) {
                float s = 0;
                const float * row = pos.v.data() + r * pos.dim;
                for (size_t k = 0; k < pos.dim; k++) s += row[k] * image_emb[k];
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
                    float s = 0;
                    const float * row = negm.v.data() + r * negm.dim;
                    for (size_t k = 0; k < negm.dim; k++) s += row[k] * image_emb[k];
                    negative_scores.push_back(round4(s));
                    if (s > max_raw) max_raw = s;
                }
                max_negative_score = max_raw;
            }
        }

        double score_diff = ruleNumber(rule, "score_diff", 0);
        double threshold = ruleNumber(rule, "threshold", 0.5);
        bool passed = best_positive_score >= threshold &&
                      best_positive_score > max_negative_score + score_diff;

        // 调试明细入表：状态区分 命中/低于阈值/被负样本压制/低于score_diff/冷却中
        d["score"] = round4(best_positive_score);
        bool neg_blocked = !passed && best_positive_score >= threshold &&
                           max_negative_score > 0 &&
                           best_positive_score <= max_negative_score + score_diff;
        if (in_cooldown) {
            d["status"] = "cooldown";
            d["would_hit"] = passed;
        } else if (passed) {
            d["status"] = "hit";
            d["hit"] = true;
        } else if (neg_blocked) {
            d["status"] = "negative_higher";
        } else {
            d["status"] = best_positive_score < threshold ? "below_threshold" : "below_score_diff";
        }
        if (!symbolic) {
            auto pt = pos_texts_.find(id);
            if (pt != pos_texts_.end()) d["positive_list"] = pt->second;
            d["positive_scores"] = positive_scores;
        }
        if (max_negative_score > 0 || !negative_scores.empty()) {
            auto nt = neg_texts_.find(id);
            if (nt != neg_texts_.end()) d["negative_list"] = nt->second;
            d["negative_scores"] = negative_scores;
            d["max_negative_score"] = round4(max_negative_score);
        }
        size_t dbg_idx = dbg_rules.size();
        dbg_rules.push_back(d);
        if (!passed || in_cooldown) continue;   // 冷却中只记分不触发

        last_trigger_[id] = now;
        std::vector<std::string> pos = pos_texts_.count(id) ? pos_texts_[id] : std::vector<std::string>{};
        std::vector<std::string> neg = neg_texts_.count(id) ? neg_texts_[id] : std::vector<std::string>{};
        bool negative_higher = (max_negative_score >= threshold && max_negative_score >= best_positive_score);

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
        m["description"] = description;         // 二次核验命中的条目稍后锁外生成规则描述回填
        m["base_description"] = description;
        m["description_zh"] = dr.zh;
        if (!detected.empty()) {
            if (mode_ == "structured") {
                json arr = json::array();   // description.split(",")
                for (const auto & a : split_any(description, ",")) if (!a.empty()) arr.push_back(a);
                m["attributes"] = arr;
            } else {
                m["attributes"] = dr.attrs;
            }
        } else {
            m["attributes"] = json::array();
        }
        size_t match_idx = matches.size();
        matches.push_back(m);
        if (rule_desc_enabled_) desc_jobs.push_back({match_idx, dbg_idx, rule, pos, neg});
    }
    dbg_snapshot_["rules"] = dbg_rules;   // 逐规则得分明细（含未命中），随快照进 /debug
    }

    // 规则二次核验的 VLM 生成（秒级）不持 mu_：批量生成规则描述期间 updateRules、
    // 监控接口不必排队。核验不通过的条目从 matches 摘除（与旧行为一致：冷却已记不回收）
    if (!desc_jobs.empty()) {
        std::vector<std::string> descs;
        descs.reserve(desc_jobs.size());
        for (auto & job : desc_jobs)
            descs.push_back(describeRuleMatch(job.rule, job.pos, job.neg, description, image));
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<size_t> drop;
        for (size_t i = 0; i < desc_jobs.size(); i++) {
            if (rule_desc_verify_ && descs[i].empty()) {
                drop.push_back(desc_jobs[i].match_idx);
                if (desc_jobs[i].dbg_idx < dbg_rules.size())
                    dbg_rules[desc_jobs[i].dbg_idx]["status"] = "desc_rejected";
                continue;
            }
            matches[desc_jobs[i].match_idx]["description"] = descs[i];
        }
        dbg_snapshot_["rules"] = dbg_rules;   // 回填 desc_rejected 标记
        std::sort(drop.begin(), drop.end());
        for (auto it = drop.rbegin(); it != drop.rend(); ++it)
            matches.erase(matches.begin() + *it);
    }
    return matches;
}
