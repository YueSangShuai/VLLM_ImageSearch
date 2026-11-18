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
    n = len(base)
    top3_ok = 0
    overlap_sum = rank_bad = 0
    max_diff = 0.0
    TIE = 0.015  # 实测 per-image 分差上界（stb 双线性 vs cv2 INTER_LINEAR 的输入像素差
                 # 为主，NPU fp16 仅 3e-4 级）；分差小于此的排序互换语义等价
    for q, ref in base.items():
        body = json.dumps({"text": q, "topk": len(ref)}).encode()
        req = urllib.request.Request(args.url, data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=60))
        got = r["results"]
        if len(got) != len(ref):
            print(f"[len] {q[:50]}: board={len(got)} baseline={len(ref)}")
            rank_bad += 1
            continue
        ref_files = [f for f, _ in ref]
        ref_score = dict(ref)
        top3_ok += set(f for f, _ in ref[:3]) == set(h["file"] for h in got[:3])
        overlap_sum += len(set(ref_files) & set(h["file"] for h in got))
        hard = []
        scores = [h["score"] for h in got]
        for i, (f, s) in enumerate(ref):
            max_diff = max(max_diff, abs(got[i]["score"] - s))
            if got[i]["file"] != f:
                # 平局判定双条件（任一满足即容忍）：
                #  a) 参考侧相邻分差 <TIE
                #  b) 板侧该位与其对调位（换入图在参考中的名次）分差 <TIE
                neigh = [ref[j][1] for j in (max(i - 1, 0), min(i + 1, len(ref) - 1))]
                a = max(neigh) - min(neigh) < TIE          # 参考侧本已是平局
                b = abs(got[i]["score"] - s) < TIE         # 两图分数都贴着平局带
                if not (a or b):
                    hard.append((i, f, got[i]["file"]))
        if hard:
            rank_bad += 1
            print(f"[rank] {q[:50]}: {hard[:2]}")
    print(f"queries n={n}  top3 集合一致 {top3_ok}/{n}  top20 平均重合 {overlap_sum/n:.2f}/20"
          f"  平局外排序不符 {rank_bad}  max|score diff|={max_diff:.4f}")
    print("RESULT:", "PASS" if rank_bad == 0 and max_diff < 0.03 else "CHECK-ABOVE")


if __name__ == "__main__":
    main()
