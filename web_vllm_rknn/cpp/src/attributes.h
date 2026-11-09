#pragma once
// 预定义行人属性表：与 processor.py PEDESTRIAN_ATTRIBUTES 对齐。
// matcher.cpp（符号匹配）与 attr_head.cpp（属性线性头属性名校验）共用，勿再各自复制。
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
