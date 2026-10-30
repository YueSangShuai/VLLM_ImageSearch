#!/usr/bin/env python
# 一致性：主机 bf16 参考分 vs 板端量化分。看排序（同规则正例是否压过跨规则负例 + 秩相关），
# 不苛求绝对分（w8a8 + 448 固定分辨率必有偏移）。用法：compare_scores.py host.json board.json
import json, sys, collections

def spearman(pairs):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    a = ranks([p[0] for p in pairs]); b = ranks([p[1] for p in pairs])
    n = len(pairs)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")

host = json.load(open(sys.argv[1]))["entries"]
board = {(e["image"], e["query_text"], e["kind"]): e for e in
         json.load(open(sys.argv[2]))["entries"]}

pairs, by_rule = [], collections.defaultdict(lambda: {"pos": [], "cross": []})
skipped = 0
for h in host:
    b = board.get((h["image"], h["query_text"], h["kind"]))
    if b is None or b.get("board_score") is None:
        skipped += 1
        continue
    pairs.append((h["score"], b["board_score"]))
    by_rule[h["rule_name"]][h["kind"]].append(b["board_score"])

print(f"matched={len(pairs)} skipped={skipped}")
print(f"global spearman = {spearman(pairs):.3f}")
ok = 0
for rule, d in sorted(by_rule.items()):
    if d["pos"] and d["cross"]:
        mp, mc = sum(d["pos"]) / len(d["pos"]), sum(d["cross"]) / len(d["cross"])
        sep = mp > mc
        ok += sep
        print(f"  {rule}: pos mean={mp:.3f} (n={len(d['pos'])})  cross mean={mc:.3f}  {'OK' if sep else 'REVERSED'}")
    else:
        print(f"  {rule}: pos mean={sum(d['pos'])/max(1,len(d['pos'])):.3f} (n={len(d['pos'])})  [无对照]")
print(f"规则级分离 {ok}/{len([d for d in by_rule.values() if d['pos'] and d['cross']])}")
all_b = [p[1] for p in pairs]
qs = sorted(all_b)
print(f"board score p10/p50/p90: {qs[len(qs)//10]:.3f} / {qs[len(qs)//2]:.3f} / {qs[len(qs)*9//10]:.3f}")
