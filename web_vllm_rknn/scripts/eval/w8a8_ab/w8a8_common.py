# -*- coding: utf-8 -*-
"""W8A8 vs bf16 A/B 共用：属性表、与板端 bench_models reportStructured 同规的分词/合法性。

分隔符与 C++ split_any(out, ",，、;；/|。.！!？? \t\r\n") 一致；
合法判定 = 小写原串命中 或 去掉 -_空格 后命中（同 reportStructured 的 nosep 逻辑）。
"""
import sys

sys.path.insert(0, "/data/yuesang/LLM/FG-CLIP/qwen_classifer")
from common import ATTRS  # 54 属性，与 attributes.h 同源

SEP_CHARS = set(",，、;；/|。.！!？? \t\r\n")

_ATTR_SET = {a.lower() for a in ATTRS}
_ATTR_NOSEP = {a.lower().replace("-", "").replace("_", "").replace(" ", ""): a for a in ATTRS}


def tokenize(text):
    out, toks, cur = [], [], ""
    for ch in text:
        if ch in SEP_CHARS:
            toks.append(cur)
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    return [t.strip() for t in toks if t.strip()]


def normalize(token):
    """-> 规范属性名 或 None（非法 token）。"""
    k = token.lower()
    if k in _ATTR_SET:
        for a in ATTRS:
            if a.lower() == k:
                return a
    k2 = k.replace("-", "").replace("_", "").replace(" ", "")
    return _ATTR_NOSEP.get(k2)


def parse_attr_text(text):
    """-> (valid: 规范名列表(去重保序), invalid: 原始非法 token 列表)"""
    valid, seen, invalid = [], set(), []
    for t in tokenize(text):
        norm = normalize(t)
        if norm is None:
            invalid.append(t)
        elif norm not in seen:
            seen.add(norm)
            valid.append(norm)
    return valid, invalid


def structured_prompt():
    """与 attributes.h defaultStructuredPrompt() 逐字一致。"""
    return ("请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），"
            "只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。\n"
            "预定义属性列表：" + ",".join(ATTRS))


COLOR_ATTRS = {a for a in ATTRS if a.startswith(("u", "l"))}  # u*/l* 颜色与花纹类


def is_enumeration(valid_attrs, min_colors=8):
    """服务 matcher 同款复读守卫：≥8 个颜色属性判为抄列表。"""
    return sum(1 for a in valid_attrs if a in COLOR_ATTRS) >= min_colors
