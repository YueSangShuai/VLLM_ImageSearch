#!/usr/bin/env python3
"""板端服务 vs x86 基线 topk 对齐检查（平局容差：相邻分差 <0.005 允许重排）。
用法: python3 scripts/eval/compare_board.py --baseline /tmp/baseline.json \
        --url http://10.8.18.118:5005/api/search"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="/tmp/baseline.json")
    ap.add_argument("--url", default="http://10.8.18.118:5005/api/search")
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    n_ok = n_fail = 0
    max_diff = 0.0
    for q, ref in base.items():
        body = json.dumps({"text": q, "topk": len(ref)}).encode()
        req = urllib.request.Request(args.url, data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=60))
        got = r["results"]
        if len(got) != len(ref):
            print(f"[len] {q[:50]}: board={len(got)} baseline={len(ref)}")
            n_fail += 1
            continue
        scores = [h["score"] for h in got]
        bad = []
        for i, (f, s) in enumerate(ref):
            if got[i]["file"] != f:
                # 相邻位分差极小 → 视为 fp16 平局重排
                neighbors = scores[max(i - 1, 0):i + 2]
                if max(neighbors) - min(neighbors) >= 0.005:
                    bad.append((i, f, got[i]["file"], got[i]["score"]))
            max_diff = max(max_diff, abs(got[i]["score"] - s))
        if bad:
            n_fail += 1
            print(f"[rank] {q[:60]}: {bad[:3]}")
        else:
            n_ok += 1
    print(f"queries ok={n_ok} fail={n_fail}  max |score diff| vs torch = {max_diff:.4f}")
    print("RESULT:", "PASS" if n_fail == 0 else "CHECK-ABOVE")


if __name__ == "__main__":
    main()
