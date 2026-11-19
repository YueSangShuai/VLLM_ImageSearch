#include "matcher.h"
#include "attributes.h"
#include "reranker.h"
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

Matcher::Matcher(VlmDescriber * describer, TextEmbedder * embedder, Reranker * reranker,
                 const Config & cfg)
    : vlm_(describer), emb_(embedder), reranker_(reranker), cfg_(cfg) {
    mode_ = cfg.match_mode;
    base_prompt_ = cfg.description_prompt;
    vlm_max_ = cfg.vlm_max_new_tokens;
    temp_ = cfg.vlm_temperature;
    rep_ = cfg.vlm_repetition_penalty;
    head_on_ = vlm_ && vlm_->attrHeadReady();
    description_prompt_ = base_prompt_;
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
// 2026-11-04 线上改回 规则短句(分句) vs 描述整句 单向量，本函数暂留备用（勿删，历史口径参考）。
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

// ---- 业务流程 ----

Matcher::DescResult Matcher::describeImage(const ImageRGB & img, const std::string & prompt) {
    // describe_image：一次生成出描述；属性线性头随视觉同一次前向顺带产出（头未启用则属性为空）
    // 结果走局部返回而非共享成员：调用方（match）不持 mu_ 也能安全生成
    DescResult r;
    std::string description = vlm_->describe(img, prompt, vlm_max_, temp_, rep_,
                                             head_on_ ? &r.attrs : nullptr);
    if (head_on_ && !description.empty() && !r.attrs.empty() &&
        std::find(r.attrs.begin(), r.attrs.end(), "Female") == r.attrs.end()) {
        r.attrs.push_back("Male");   // Female 缺失派生 Male（规则符号表用）
    }
    auto [en, zh] = splitBilingual(description);
    r.zh = zh.empty() ? "" : collapseWs(zh);
    r.text = cleanVlmDescription(en, 1200);
    return r;
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
    // 卡顿治理两条（2026-11-05）：
    //   ① vec_cache_ 文本→向量缓存：只有新文本才进 BGE（调阈值/改名/开关 = 零编码，微秒级）；
    //   ② 编码在 mu_ 锁外进行——match() 持 mu_ 跨整个 VLM 描述（可达数秒），旧版整表
    //      重编码挤在锁内，HTTP 保存规则要等“当前帧描述完 + 全部编码”=整站等。
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

    // 锁内比对缓存，只收集缺失文本
    std::vector<std::string> todo;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto & t : want)
            if (!vec_cache_.count(t)) todo.push_back(t);
    }

    // 锁外编码缺失文本（非对称检索：规则文本是 query，加 BGE 指令前缀；描述侧不加）。
    // TextEmbedder::encode 自带锁，与在跑的描述/打分互不干涉。
    if (!todo.empty()) {
        size_t dim = emb_->dim();
        std::vector<float> vecs = emb_->encode(todo, cfg_.embed_query_prefix);
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < todo.size() && (i + 1) * dim <= vecs.size(); i++)
            vec_cache_[todo[i]] = std::vector<float>(vecs.begin() + i * dim,
                                                     vecs.begin() + (i + 1) * dim);
    }

    // 锁内只做查表拼装换表（微秒级）
    {
        std::lock_guard<std::mutex> lk(mu_);
        rules_ = std::move(new_rules);
        rule_emb_.clear();
        pos_texts_.clear();
        attr_pos_.clear();
        neg_emb_.clear();
        neg_texts_.clear();
        attr_neg_.clear();

        for (const auto & rt : rts) {
            const std::string & id = rt.id;
            pos_texts_[id] = rt.pos;

            EmbMat m;
            for (const auto & t : rt.pos) {
                auto it = vec_cache_.find(t);
                if (it == vec_cache_.end()) continue;
                m.v.insert(m.v.end(), it->second.begin(), it->second.end());
                m.n++;
            }
            rule_emb_[id] = std::move(m);

            // 负样本：与正样本同口径（query 前缀），打分时 pos 须 > max(neg)+score_diff 才告警
            neg_texts_[id] = rt.neg;
            EmbMat nm;
            for (const auto & t : rt.neg) {
                auto it = vec_cache_.find(t);
                if (it == vec_cache_.end()) continue;
                nm.v.insert(nm.v.end(), it->second.begin(), it->second.end());
                nm.n++;
            }
            neg_emb_[id] = std::move(nm);

            if (mode_ == "combined") {
                std::optional<std::set<std::string>> merged = std::set<std::string>();
                bool ok = !rt.pos.empty();
                for (const auto & t : rt.pos) {
                    auto parsed = parseRuleAttributes(t);
                    if (!parsed) { ok = false; break; }
                    merged->insert(parsed->begin(), parsed->end());
                }
                attr_pos_[id] = ok ? merged : std::optional<std::set<std::string>>{};

                // 符号规则负样本：能解析出的属性并集；本帧检出任一 → max_negative=1.0 压掉告警
                std::set<std::string> neg_attrs;
                for (const auto & t : rt.neg) {
                    auto parsed = parseRuleAttributes(t);
                    if (parsed) neg_attrs.insert(parsed->begin(), parsed->end());
                }
                attr_neg_[id] = neg_attrs;
            }
        }

        description_prompt_ = base_prompt_;
    }
}

// ---- 调试捕获 ----

json Matcher::lastDebugSnapshot() {
    std::lock_guard<std::mutex> lk(mu_);
    return dbg_snapshot_;
}

json Matcher::describeForDebug(const ImageRGB & image) {
    // VLM 生成（秒级）在 mu_ 锁外：旧版持锁跑完整描述，该设备的 updateRules
    // 要排队等 ~16s；refreshRules 又持 Monitor::mu_ 调 updateRules，会连锁拖住
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

json Matcher::match(const ImageRGB & image) {
    // match：符号匹配（属性规则）+ BGE 向量匹配（自然语言规则）
    json empty = json::array();
    std::string prompt;
    bool rules_empty;
    {
        std::lock_guard<std::mutex> lk(mu_);
        dbg_snapshot_ = json::object();   // 调试快照按图作废，防止上一帧串图
        cooldown_backup_.clear();         // 回滚备份只对本次 match 的触发有效
        rules_empty = rules_.empty();
        if (!rules_empty && rule_emb_.empty()) return empty;
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

    // VLM 描述（秒级）不持 matcher mu_：刷新规则时的 updateRules 不必排队等描述
    // （这正是「更新规则整个平台卡住」的根因）；多设备并发生成由 VlmDescriber 内部锁兜底
    DescResult dr = describeImage(image, prompt);

    std::lock_guard<std::mutex> lk(mu_);
    std::string description = dr.text;

    std::set<std::string> detected;
    if (mode_ == "combined") {
        detected.insert(dr.attrs.begin(), dr.attrs.end());
    }

    // 调试快照：本帧描述/中文描述/属性（未命中规则的检测也能在 /debug 页看到）
    {
        dbg_snapshot_ = json::object();
        dbg_snapshot_["description"] = description;
        dbg_snapshot_["description_zh"] = dr.zh;
        json attrs = json::array();
        for (const auto & a : dr.attrs) attrs.push_back(a);
        dbg_snapshot_["attributes"] = attrs;
    }

    if (description.empty()) {
        printf("VLM描述为空，跳过本次匹配\n");
        return empty;
    }

    // combined：全部规则都能符号匹配（属性规则）时跳过图像描述向量计算
    bool needs_embedding = true;
    if (mode_ == "combined") {
        needs_embedding = false;
        for (const auto & rule : rules_) {
            auto it = attr_pos_.find(getStr(rule, "id"));
            if (it == attr_pos_.end() || !it->second.has_value()) { needs_embedding = true; break; }
        }
    }
    // 整句对整句：整段描述（cleanVlmDescription 后）作为单行向量（BGE doc 侧，不加前缀）。
    // 2026-11-04 应需求从「分句+取 max」改回整句口径，注意分数整体下移 0.07~0.11，
    // 分句口径下标定的规则阈值（黑衣服≈0.60）相应偏高，必要时下调。
    std::vector<float> image_emb;
    size_t image_rows = 0;
    if (needs_embedding) {
        image_emb = emb_->encode({description});
        image_rows = 1;
    }

    json matches = json::array();
    // 调试捕获：逐规则得分明细（未命中/冷却中的规则也算分、也记录），match 末尾挂到快照
    json dbg_rules = json::array();
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
        std::string best_positive_text;
        json positive_scores = json::array();
        double max_negative_score = 0.0;
        json negative_scores = json::array();

        auto ap = attr_pos_.find(id);
        bool symbolic = ap != attr_pos_.end() && ap->second.has_value() && !ap->second->empty();
        d["type"] = symbolic ? "symbolic" : "semantic";
        if (symbolic) {
            // 符号匹配：得分为正样本属性覆盖率
            std::vector<std::string> pos_attrs(ap->second->begin(), ap->second->end());  // set 已有序
            std::vector<std::string> present, missing;
            for (const auto & a : pos_attrs) {
                bool has = detected.count(a) > 0;
                positive_scores.push_back(has ? 1.0 : 0.0);
                (has ? present : missing).push_back(a);
            }
            best_positive_score = (double)present.size() / (double)pos_attrs.size();
            best_positive_text = present.empty() ? pos_attrs[0] : join(present, ",");
            // 负样本属性：检出任一 → 1.0（python: rule_attribute_negatives & detected）
            auto an = attr_neg_.find(id);
            if (an != attr_neg_.end()) {
                for (const auto & a : an->second)
                    if (detected.count(a)) { max_negative_score = 1.0; break; }
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

            // 负样本向量 vs 描述：逐条得分 + 最大值（python: negative_embeddings @ image_embedding）
            auto ne = neg_emb_.find(id);
            if (ne != neg_emb_.end() && ne->second.n > 0 && image_rows > 0) {
                const EmbMat & neg = ne->second;
                for (size_t r = 0; r < neg.n; r++) {
                    const float * row = neg.v.data() + r * neg.dim;
                    float s = -1e30f;
                    for (size_t c = 0; c < image_rows; c++) {
                        const float * cv = image_emb.data() + c * neg.dim;
                        float sc = 0;
                        for (size_t k = 0; k < neg.dim; k++) sc += row[k] * cv[k];
                        if (sc > s) s = sc;
                    }
                    negative_scores.push_back(round4(s));
                    if ((double)s > max_negative_score) max_negative_score = (double)s;
                }
            }
        }

        double score_diff = ruleNumber(rule, "score_diff", 0);
        double threshold = ruleNumber(rule, "threshold", 0.5);
        // 负样本门（python 口径）：pos 须超过 max(neg)+score_diff；无负样本时退化为旧行为 pos>score_diff
        bool passed = best_positive_score >= threshold &&
                      best_positive_score > max_negative_score + score_diff;

        // ===== 二次核验（Qwen3-Reranker）：BGE 达标的语义规则再核一道 =====
        // 启用条件：Reranker 可用 && 规则字段 rerank_threshold>0（未设/0=关闭，存量规则零变化）
        // && 语义规则（符号规则不走 BGE、也不核验）。冷却中照跑（would_hit 须反映核验结果）但不触发。
        // query=最高分正样本文本，document=本帧 VLM 描述。P<0（rkllm 失败）放行：
        // 核验故障不阻断告警链路。耗时 ~231ms/次，只有 BGE 过关的规则才会调用，成本有界。
        double rerank_thr = ruleNumber(rule, "rerank_threshold", 0);
        bool rerank_checked = false;
        double rerank_p = -1.0;
        if (rerank_thr > 0) d["rerank_threshold"] = round4(rerank_thr);
        if (reranker_ && reranker_->ok() && rerank_thr > 0 && !symbolic && passed) {
            rerank_p = reranker_->rerank(best_positive_text, description);
            rerank_checked = true;
            d["rerank_score"] = rerank_p >= 0 ? json(round4(rerank_p)) : json();
            if (rerank_p >= 0 && rerank_p < rerank_thr) passed = false;
        }
        bool rerank_blocked = rerank_checked && rerank_p >= 0 && rerank_p < rerank_thr;

        // 调试明细入表：状态区分 命中/低于阈值/被负样本压制/低于score_diff/核验未过/冷却中
        d["score"] = round4(best_positive_score);
        bool neg_blocked = !passed && !rerank_blocked && best_positive_score >= threshold &&
                           max_negative_score > 0 &&
                           best_positive_score <= max_negative_score + score_diff;
        if (in_cooldown) {
            d["status"] = "cooldown";
            d["would_hit"] = passed;
        } else if (passed) {
            d["status"] = "hit";
            d["hit"] = true;
        } else if (rerank_blocked) {
            d["status"] = "rerank_rejected";
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
        dbg_rules.push_back(d);
        if (!passed || in_cooldown) continue;   // 冷却中只记分不触发，与旧行为一致

        // 先备份旧冷却时刻再覆盖：朝向过滤在 match 之后才丢弃命中，
        // 被丢弃的规则靠 undoTrigger 还原到这里（丢弃的命中不该启动冷却）
        auto prev_lt = last_trigger_.find(id);
        cooldown_backup_[id] = prev_lt == last_trigger_.end() ? -1.0 : prev_lt->second;
        last_trigger_[id] = now;
        std::vector<std::string> pos = pos_texts_.count(id) ? pos_texts_[id] : std::vector<std::string>{};

        json m;
        m["rule_id"] = id;
        m["rule_name"] = getStr(rule, "name");
        m["positive"] = best_positive_text;
        m["positive_list"] = pos;
        m["positive_scores"] = positive_scores;
        auto nt = neg_texts_.find(id);
        m["negative_list"] = nt != neg_texts_.end() ? json(nt->second) : json::array();
        m["negative_scores"] = negative_scores;
        m["negative_higher"] = max_negative_score >= threshold &&
                               max_negative_score >= best_positive_score;
        m["final_score"] = round4(best_positive_score);
        if (rerank_checked) {   // 命中的规则也带上核验分，告警/检测日志可追溯
            m["rerank_score"] = rerank_p >= 0 ? json(round4(rerank_p)) : json();
            m["rerank_threshold"] = round4(rerank_thr);
        }
        m["description"] = description;
        m["base_description"] = description;
        m["description_zh"] = dr.zh;
        if (!detected.empty() || (head_on_ && !dr.attrs.empty())) {
            m["attributes"] = dr.attrs;   // combined（符号匹配）或 description+头（仅展示）
        } else {
            m["attributes"] = json::array();
        }
        matches.push_back(m);
    }
    dbg_snapshot_["rules"] = dbg_rules;   // 逐规则得分明细（含未命中），随快照进 /debug
    return matches;
}

void Matcher::undoTrigger(const std::vector<std::string> & rule_ids) {
    if (rule_ids.empty()) return;
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto & id : rule_ids) {
        auto bk = cooldown_backup_.find(id);
        if (bk == cooldown_backup_.end()) continue;   // 本次 match 未触发该规则，无可回滚
        if (bk->second < 0) last_trigger_.erase(id);  // 触发前本无冷却 → 直接清除
        else last_trigger_[id] = bk->second;          // 还原为触发前的旧冷却时刻
        cooldown_backup_.erase(bk);
    }
}
