# 检查板端 tasks.json 规则结构（BGE query 文本在哪个字段）
import json

d = json.load(open('/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/scripts/eval/w8a8_ab/board_rules/tasks.json'))
rules = [r for t in d for r in t.get("rules", [])]
print("rules:", len(rules))
print("keys:", sorted(rules[0].keys()))
for r in rules:
    print("----", r.get("name"), "| threshold:", r.get("threshold"),
          "| keys_with_text:", [k for k, v in r.items() if isinstance(v, str) and len(v) > 3])
    for k in ("rule", "feature", "features", "description", "query", "text",
              "content", "condition", "positive", "prompt"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            print("   ", k, "=", v[:100])
        elif isinstance(v, list) and v:
            print("   ", k, "(list) =", str(v)[:120])
