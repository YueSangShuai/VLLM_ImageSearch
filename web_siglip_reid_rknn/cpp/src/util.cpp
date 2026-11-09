#include "util.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <ctime>

std::vector<uint32_t> utf8_to_codepoints(const std::string &s) {
    std::vector<uint32_t> cps;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6 && i + 1 < n) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE && i + 2 < n) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E && i + 3 < n) { cp = c & 0x07; len = 4; }
        else { cp = 0xFFFD; len = 1; }
        bool bad = false;
        for (int k = 1; k < len; k++) {
            size_t idx = i + k;
            if (idx >= n || ((unsigned char)s[idx] >> 6) != 0x2) { bad = true; break; }
            cp = (cp << 6) | ((unsigned char)s[idx] & 0x3F);
        }
        if (bad) { cps.push_back(0xFFFD); i += 1; continue; }
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

std::string codepoints_to_utf8(const std::vector<uint32_t> &cps) {
    std::string out;
    for (uint32_t cp : cps) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

static bool is_ascii_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && (unsigned char)s[b] <= ' ') b++;
    while (e > b && (unsigned char)s[e - 1] <= ' ') e--;
    // Python str.strip() 还会去掉 U+3000 等空白；对中文场景主要需要全角空格
    auto cps = utf8_to_codepoints(s.substr(b, e - b));
    auto is_space_cp = [](uint32_t cp) {
        return cp == 0x20 || cp == 0x3000 || cp == 0xA0 || cp == 0x09 ||
               cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
               cp == 0x1C || cp == 0x1D || cp == 0x1E || cp == 0x1F;
    };
    size_t cb = 0, ce = cps.size();
    while (cb < ce && is_space_cp(cps[cb])) cb++;
    while (ce > cb && is_space_cp(cps[ce - 1])) ce--;
    return codepoints_to_utf8(std::vector<uint32_t>(cps.begin() + cb, cps.begin() + ce));
}

std::string trim_chars(const std::string &s, const std::string &chars) {
    // 按码点去掉首尾 chars 集合中的字符（chars 为 UTF-8 字符列表）
    auto target = utf8_to_codepoints(chars);
    auto cps = utf8_to_codepoints(s);
    auto in = [&](uint32_t cp) { return std::find(target.begin(), target.end(), cp) != target.end(); };
    size_t b = 0, e = cps.size();
    while (b < e && in(cps[b])) b++;
    while (e > b && in(cps[e - 1])) e--;
    return codepoints_to_utf8(std::vector<uint32_t>(cps.begin() + b, cps.begin() + e));
}

std::vector<std::string> split_any(const std::string &s, const std::string &seps) {
    // 按任意分隔码点切分；连续分隔符自然产生空串（与 python re.split 一致）
    auto seps_cp = utf8_to_codepoints(seps);
    auto cps = utf8_to_codepoints(s);
    std::vector<std::string> out;
    std::vector<uint32_t> cur;
    for (uint32_t cp : cps) {
        if (std::find(seps_cp.begin(), seps_cp.end(), cp) != seps_cp.end()) {
            out.push_back(codepoints_to_utf8(cur));
            cur.clear();
        } else {
            cur.push_back(cp);
        }
    }
    out.push_back(codepoints_to_utf8(cur));
    return out;
}

std::string join(const std::vector<std::string> &parts, const std::string &sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

bool contains(const std::string &hay, const std::string &needle) {
    return needle.empty() || hay.find(needle) != std::string::npos;
}

void replace_all(std::string &hay, const std::string &from, const std::string &to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = hay.find(from, pos)) != std::string::npos) {
        hay.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string tolower_ascii(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

std::string utf8_substr(const std::string &s, size_t max_chars) {
    auto cps = utf8_to_codepoints(s);
    if (cps.size() <= max_chars) return s;
    return codepoints_to_utf8(std::vector<uint32_t>(cps.begin(), cps.begin() + max_chars));
}

size_t utf8_length(const std::string &s) { return utf8_to_codepoints(s).size(); }

std::string uuid4() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd() ^ (uint64_t)time(nullptr));
    std::uniform_int_distribution<uint64_t> dist;
    uint8_t b[16];
    uint64_t x = dist(gen), y = dist(gen);
    memcpy(b, &x, 8); memcpy(b + 8, &y, 8);
    b[6] = (uint8_t)((b[6] & 0x0F) | 0x40);
    b[8] = (uint8_t)((b[8] & 0x3F) | 0x80);
    char buf[37];
    snprintf(buf, sizeof(buf),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return buf;
}

std::string now_isoformat() {
    auto now = std::chrono::system_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  now.time_since_epoch()).count() % 1000000;
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local;
    localtime_r(&tt, &tm_local);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_local);
    std::string out(buf);
    if (us != 0) {
        char frac[16];
        snprintf(frac, sizeof(frac), ".%06lld", (long long)us);
        out += frac;
    }
    return out;
}

std::string today_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local;
    localtime_r(&tt, &tm_local);
    char buf[16];
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_local);
    return buf;
}

std::string sanitize_name(std::string s) {
    replace_all(s, "/", "_");
    replace_all(s, "\\", "_");
    return s;
}

std::string format_double(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}
