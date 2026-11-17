#pragma once
// 字符串/时间/UUID 工具：Python 端按码点操作，这里逐一对齐。
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

// UTF-8 -> 码点数组
std::vector<uint32_t> utf8_to_codepoints(const std::string &s);
std::string codepoints_to_utf8(const std::vector<uint32_t> &cps);

std::string trim(const std::string &s);
std::string trim_chars(const std::string &s, const std::string &chars);
std::vector<std::string> split_any(const std::string &s, const std::string &seps);
std::string join(const std::vector<std::string> &parts, const std::string &sep);
bool contains(const std::string &hay, const std::string &needle);
void replace_all(std::string &hay, const std::string &from, const std::string &to);
std::string tolower_ascii(std::string s);

// 按码点截断/取子串（对齐 Python len()/切片）
std::string utf8_substr(const std::string &s, size_t max_chars);
size_t utf8_length(const std::string &s);

std::string uuid4();
// Python datetime.now().isoformat()（microsecond!=0 时输出 6 位小数）
std::string now_isoformat();
std::string today_str();               // %Y-%m-%d
std::string sanitize_name(std::string s);  // '/'、'\\' -> '_'

std::string format_double(double v);   // 与 python json 数值输出风格接近即可
