#include "text_embedder.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

const char *WS = " \t\n\r\f\v";

bool is_whitespace_cp(uint32_t cp) {
    return cp == 32 || cp == 9 || cp == 10 || cp == 13 || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

bool is_control_cp(uint32_t cp) {
    if (cp == 0) return true;  // HF: \x00 控制字符（非 \t\n\r）
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

bool is_punct_cp(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
        (cp >= 123 && cp <= 126)) return true;
    if (cp >= 0x2010 && cp <= 0x2017) return true;
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201C || cp == 0x201D) return true;
    if (cp == 0x3001 || cp == 0x3002 || cp == 0xFF0C || cp == 0xFF0E || cp == 0xFF01 ||
        cp == 0xFF1F || cp == 0xFF1A || cp == 0xFF1B || cp == 0xFF03 || cp == 0xFF05 ||
        cp == 0xFF04 || cp == 0xFF06 || cp == 0xFF0A || cp == 0xFF20 || cp == 0xFF3B ||
        cp == 0xFF3D || cp == 0xFF5B || cp == 0xFF5D) return true;
    return false;
}

bool is_chinese_cp(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF);
}

uint32_t lower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return cp + 32;
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32;  // Latin-1 大写
    return cp;
}

// strip_accents（do_lower_case=True 路径）：常用变音符 → 基字母（英文语料够用）
uint32_t strip_accent_cp(uint32_t cp) {
    static const std::pair<uint32_t, uint32_t> table[] = {
        {0xC0, 'A'}, {0xC1, 'A'}, {0xC2, 'A'}, {0xC3, 'A'}, {0xC4, 'A'}, {0xC5, 'A'},
        {0xC6, 'AE'}, {0xC7, 'C'}, {0xC8, 'E'}, {0xC9, 'E'}, {0xCA, 'E'}, {0xCB, 'E'},
        {0xCC, 'I'}, {0xCD, 'I'}, {0xCE, 'I'}, {0xCF, 'I'}, {0xD0, 'D'}, {0xD1, 'N'},
        {0xD2, 'O'}, {0xD3, 'O'}, {0xD4, 'O'}, {0xD5, 'O'}, {0xD6, 'O'}, {0xD8, 'O'},
        {0xD9, 'U'}, {0xDA, 'U'}, {0xDB, 'U'}, {0xDC, 'U'}, {0xDD, 'Y'}, {0xDE, 'T'},
        {0xE0, 'a'}, {0xE1, 'a'}, {0xE2, 'a'}, {0xE3, 'a'}, {0xE4, 'a'}, {0xE5, 'a'},
        {0xE6, 'ae'}, {0xE7, 'c'}, {0xE8, 'e'}, {0xE9, 'e'}, {0xEA, 'e'}, {0xEB, 'e'},
        {0xEC, 'i'}, {0xED, 'i'}, {0xEE, 'i'}, {0xEF, 'i'}, {0xF1, 'n'}, {0xF2, 'o'},
        {0xF3, 'o'}, {0xF4, 'o'}, {0xF5, 'o'}, {0xF6, 'o'}, {0xF8, 'o'}, {0xF9, 'u'},
        {0xFA, 'u'}, {0xFB, 'u'}, {0xFC, 'u'}, {0xFD, 'y'}, {0xFF, 'y'},
        {0x100, 'A'}, {0x101, 'a'}, {0x102, 'A'}, {0x103, 'a'}, {0x104, 'A'}, {0x105, 'a'},
        {0x112, 'E'}, {0x113, 'e'}, {0x114, 'E'}, {0x115, 'e'}, {0x116, 'E'}, {0x117, 'e'},
        {0x118, 'E'}, {0x119, 'e'}, {0x14C, 'O'}, {0x14D, 'o'}, {0x14E, 'O'}, {0x14F, 'o'},
        {0x150, 'O'}, {0x151, 'o'}, {0x168, 'U'}, {0x169, 'u'}, {0x16A, 'U'}, {0x16B, 'u'},
        {0x16C, 'U'}, {0x16D, 'u'}, {0x16E, 'U'}, {0x16F, 'u'}, {0x170, 'U'}, {0x171, 'u'},
        {0x172, 'U'}, {0x173, 'u'}, {0x174, 'W'}, {0x175, 'w'}, {0x176, 'Y'}, {0x177, 'y'},
        {0x178, 'Y'}, {0x179, 'Z'}, {0x17A, 'z'}, {0x17B, 'Z'}, {0x17C, 'z'}, {0x17D, 'Z'},
        {0x17E, 'z'}, {0x0152, 'OE'}, {0x0153, 'oe'}, {0x0178, 'Y'}, {0x0192, 'f'},
        {0x02C6, '^'}, {0x02DC, '~'}, {0x2013, '-'}, {0x2014, '-'}, {0x2018, '\''},
        {0x2019, '\''}, {0x201A, ','}, {0x201C, '"'}, {0x201D, '"'}, {0x201E, '"'},
        {0x2020, '+'}, {0x2022, '*'}, {0x2026, '.'}, {0x2039, '<'}, {0x203A, '>'},
        {0x20AC, 'e'}, {0x2122, 'T'},
    };
    for (auto &kv : table)
        if (kv.first == cp) return kv.second;
    return cp;
}

std::string collapse_spaces(const std::string &s) {
    // re.sub(r"\s{2,}", ' ')
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        char c = s[i];
        if (strchr(WS, c)) {
            size_t j = i;
            while (j < s.size() && strchr(WS, s[j])) j++;
            out += ' ';
            i = j;
        } else {
            out += c;
            i++;
        }
    }
    return out;
}

}  // namespace

// dataset/utils.py pre_caption(caption, max_words=50) 等价复刻：
// lower → 删 [,.'!?\"()*#:;~] → '-' '/' 换空格 → 多空白压一个 → strip → 50 词截断
std::string pre_caption(const std::string &caption) {
    std::string s;
    s.reserve(caption.size());
    for (char ch : caption) {
        unsigned char c = (unsigned char)ch;
        uint32_t base = lower_cp(c);
        if (base >= 0x80) { s += ch; continue; }  // UTF-8 多字节原样
        char lc = (char)base;
        if (strchr(",.'!?\"()*#:;~", lc)) continue;
        if (lc == '-' || lc == '/') lc = ' ';
        s += lc;
    }
    s = collapse_spaces(s);
    // rstrip('\n') + strip(' ')（collapse 后换行已成空格，与 python 顺序差异可忽略）
    size_t b = s.find_first_not_of(" \n");
    size_t e = s.find_last_not_of(" \n");
    s = (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
    // 50 词截断
    int words = 0;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t sp = s.find(' ', pos);
        if (sp == std::string::npos) sp = s.size();
        if (++words > 50) { s = s.substr(0, pos); if (!s.empty() && s.back() == ' ') s.pop_back(); break; }
        pos = sp + 1;
    }
    return s;
}

TextEmbedder::TextEmbedder(const std::string &rknn_path, const std::string &vocab_path,
                           int max_len)
    : max_len_(max_len) {
    {
        std::ifstream in(vocab_path);
        if (!in) throw std::runtime_error("无法打开 vocab: " + vocab_path);
        std::string line;
        int id = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            vocab_[line] = id++;
        }
        auto get = [&](const char *t, int32_t dft) {
            auto it = vocab_.find(t);
            return it == vocab_.end() ? dft : it->second;
        };
        cls_id_ = get("[CLS]", 101);
        sep_id_ = get("[SEP]", 102);
        pad_id_ = get("[PAD]", 0);
        unk_id_ = get("[UNK]", 100);
    }
    int ret = rknn_init(&ctx_, (void *)rknn_path.c_str(), 0, 0, NULL);
    if (ret < 0) throw std::runtime_error("rknn_init MARS 文本塔失败 ret=" + std::to_string(ret));
    const char *cores = getenv("RKNN_MARS_CORE_NUM");
    int core = (cores && atoi(cores) > 0) ? atoi(cores) : 1;
    ret = rknn_set_core_mask(ctx_, core == 3 ? RKNN_NPU_CORE_0_1_2
                                 : core == 2 ? RKNN_NPU_CORE_0_1
                                             : RKNN_NPU_CORE_0);
    if (ret < 0) throw std::runtime_error("rknn_set_core_mask MARS 文本塔失败");
    rknn_input_output_num io_num = {};
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_input != 2)
        throw std::runtime_error("文本塔期望 2 输入，实际 " + std::to_string((unsigned)io_num.n_input));
    in_ids_attr_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_ids_attr_, sizeof(in_ids_attr_));
    in_mask_attr_.index = 1;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_mask_attr_, sizeof(in_mask_attr_));
    out_attr_.index = 0;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr_, sizeof(out_attr_));
    dim_ = (int)out_attr_.n_elems;
    fprintf(stderr, "MARS text rknn: ids type=%d mask type=%d out dim=%d\n",
            (int)in_ids_attr_.type, (int)in_mask_attr_.type, dim_);
}

TextEmbedder::~TextEmbedder() {
    if (ctx_) rknn_destroy(ctx_);
}

// HF BertTokenizer(uncased)：clean → lower/strip_accents → 标点/CJK 隔离 → WordPiece
// → [CLS]..[SEP]，定长 max_len（pad=0），attn = ids!=pad
void TextEmbedder::tokenize(const std::string &cleaned, std::vector<int32_t> &ids,
                            std::vector<float> &attn) const {
    // 1. clean_text + 2. basic tokenize 合成一遍扫描
    auto cps = utf8_to_codepoints(cleaned);
    std::vector<uint32_t> norm;
    norm.reserve(cps.size());
    for (uint32_t cp : cps) {
        if (cp == 0) continue;                      // 控制字符删除
        if (is_whitespace_cp(cp)) { norm.push_back(32); continue; }
        if (is_control_cp(cp)) continue;
        uint32_t lc = lower_cp(cp);
        lc = strip_accent_cp(lc);
        if (is_punct_cp(lc) || is_chinese_cp(lc)) { // 独立成 token
            norm.push_back(32);
            norm.push_back(lc);
            norm.push_back(32);
        } else {
            norm.push_back(lc);
        }
    }
    // 3. 空白切词 → WordPiece
    std::string text = codepoints_to_utf8(norm);
    ids.clear();
    std::string word;
    auto do_word = [&](const std::string &w) {
        if (w.empty()) return;
        auto wcps = utf8_to_codepoints(w);
        if (wcps.size() > 100) { ids.push_back(unk_id_); return; }
        bool unk = false;
        std::vector<int32_t> pieces;
        size_t start = 0;
        while (start < wcps.size()) {
            size_t end = wcps.size();
            int32_t found = -1;
            while (start < end) {
                std::string sub = codepoints_to_utf8(
                    std::vector<uint32_t>(wcps.begin() + start, wcps.begin() + end));
                if (start > 0) sub = "##" + sub;
                auto it = vocab_.find(sub);
                if (it != vocab_.end()) { found = it->second; break; }
                end--;
            }
            if (found < 0) { unk = true; break; }
            pieces.push_back(found);
            start = end;
        }
        if (unk) ids.push_back(unk_id_);
        else ids.insert(ids.end(), pieces.begin(), pieces.end());
    };
    for (char ch : text + " ") {
        if (strchr(WS, ch)) { do_word(word); word.clear(); }
        else word += ch;
    }

    // 4. 特殊符 + 定长 pad
    if ((int)ids.size() > max_len_ - 2) ids.resize(max_len_ - 2);
    ids.insert(ids.begin(), cls_id_);
    ids.push_back(sep_id_);
    ids.resize(max_len_, pad_id_);
    attn.assign(max_len_, 0.0f);
    for (int i = 0; i < max_len_; i++) attn[i] = (ids[i] != pad_id_) ? 1.0f : 0.0f;
}

std::vector<float> TextEmbedder::encode(const std::string &text) {
    return encode_clean(pre_caption(text));
}

std::vector<float> TextEmbedder::encode_clean(const std::string &cleaned) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int32_t> ids;
    std::vector<float> attn;
    tokenize(cleaned, ids, attn);

    rknn_input ins[2] = {};
    ins[0].index = 0;
    ins[0].type = (rknn_tensor_type)in_ids_attr_.type;   // INT32（导出时 ids 为 int32）
    ins[0].fmt = RKNN_TENSOR_UNDEFINED;
    ins[0].size = (uint32_t)(max_len_ * sizeof(int32_t));
    ins[0].pass_through = 1;
    ins[0].buf = ids.data();
    ins[1].index = 1;
    ins[1].type = RKNN_TENSOR_FLOAT32;
    ins[1].fmt = RKNN_TENSOR_UNDEFINED;
    ins[1].size = (uint32_t)(max_len_ * sizeof(float));
    ins[1].pass_through = 1;
    ins[1].buf = attn.data();
    int ret = rknn_inputs_set(ctx_, 2, ins);
    if (ret < 0) throw std::runtime_error("text rknn_inputs_set 失败 ret=" + std::to_string(ret));
    ret = rknn_run(ctx_, nullptr);
    if (ret < 0) throw std::runtime_error("text rknn_run 失败 ret=" + std::to_string(ret));

    rknn_output out = {};
    out.index = 0;
    out.want_float = 1;
    ret = rknn_outputs_get(ctx_, 1, &out, NULL);
    if (ret < 0) throw std::runtime_error("text rknn_outputs_get 失败 ret=" + std::to_string(ret));
    size_t n = out_attr_.n_elems;
    std::vector<float> vec(n);
    memcpy(vec.data(), out.buf, n * sizeof(float));
    rknn_outputs_release(ctx_, 1, &out);

    double norm = 0;
    for (float v : vec) norm += (double)v * v;
    norm = std::sqrt(norm);
    if (norm > 0) for (auto &v : vec) v = (float)(v / norm);
    dim_ = (int)n;
    return vec;
}
