#pragma once
// 预定义行人属性表：与 processor.py PEDESTRIAN_ATTRIBUTES 对齐。
// matcher.cpp（线上匹配）与 bench_models.cpp（--structured 离线测量）共用，勿再各自复制。
#include "util.h"

#include <string>
#include <unordered_map>
#include <vector>

inline const std::vector<std::string> & pedestrianAttributes() {
    static const std::vector<std::string> attrs = {
        "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Side", "Back",
        "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
        "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
        "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
        "ublack", "ugray", "ublue", "ugreen", "uwhite", "upurple", "ured", "ubrown",
        "uyellow", "upink", "uorange", "ubeige", "ustriped_color", "umulticolor",
        "lwhite", "lpink", "lred", "lgreen", "lyellow", "lpurple", "lbrown", "lblack",
        "lorange", "lblue", "lgray", "lbeige", "lstriped_color", "lmulticolor",
    };
    return attrs;
}

// 小写 / 去分隔符两种写法都能命中；Male 由 Female 缺失时派生
inline const std::unordered_map<std::string, std::string> & attrLookupTable() {
    static const std::unordered_map<std::string, std::string> tbl = [] {
        std::unordered_map<std::string, std::string> m;
        for (const auto & attr : pedestrianAttributes()) {
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

// cfg.structured_prompt 为空时的兜底提示词（与 matcher 默认一致）。
// 注意：属性列表必须保留——它锚定输出词表，去掉后模型会输出自由词（10/28 实测）。
inline std::string defaultStructuredPrompt() {
    std::string p = "请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），"
                    "只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。\n"
                    "预定义属性列表：";
    const auto & attrs = pedestrianAttributes();
    for (size_t i = 0; i < attrs.size(); i++) {
        if (i) p += ",";
        p += attrs[i];
    }
    return p;
}
