#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两条文本 → 两个向量化网络的匹配分数。

复现 web_vllm_rknn 线上口径（与 eval_colbert_ab.py / C++ matcher 一致）：
  - bge-small-zh-v1.5      : 单向量 cos；query 侧加 BGE 中文检索指令前缀，doc 侧不加
  - GTE-ModernColBERT-v1   : late-interaction MaxSim；query 加 "[Q] "，doc 加 "[D] "

用法：改下面 QUERIES / DOCS 两个列表，直接运行：
  python3 text_match_score.py
输出所有 查询×候选 组合在两个网络下的分数。
注意：这是整句对整句的分数；板子线上是 规则×描述分句 取 max，数值会更高。
"""
import json
import os

import numpy as np

# ==================== 在这里改输入 ====================
QUERIES = [
    "一名男性穿黑色上衣",
    "一名男性穿蓝色上衣",
    "一名男性穿黄色上衣",
    
]
DOCS = [
    "该行人的背影显示其为一名男性，穿着黄色上衣和灰色裤子，头戴一顶黄色安全帽，脚穿黑色鞋子。他正面向前方行走，身体姿态自然，未携带明显物品。由于图像模糊且视角有限，无法准确判断年龄或具体衣物细节（如领口、袖子等），但可见其整体着装为黄灰配色的工装风格服装"
]
# ======================================================

MODELS = "/data/yuesang/LLM/FG-CLIP/models"
GTE_DIR = os.path.join(MODELS, "GTE-ModernColBERT-v1")
BGE_NAME = os.path.join(MODELS, "bge-small-zh-v1.5")
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class BgeScorer:
    name = "bge-small-zh cos"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(BGE_NAME, device="cpu")

    def score_matrix(self, queries, docs) -> np.ndarray:
        """返回 [len(queries), len(docs)] 的 cos 分数矩阵。"""
        Q = self.m.encode([BGE_QUERY_PREFIX + q for q in queries], normalize_embeddings=True,
                          convert_to_numpy=True)
        D = self.m.encode(list(docs), normalize_embeddings=True, convert_to_numpy=True)
        return Q @ D.T


class GteMcScorer:
    name = "gte-moderncolbert maxsim"

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(GTE_DIR, "tokenizer.json"))
        onnx = os.environ.get("GTE_ONNX", os.path.join(GTE_DIR, "model.onnx"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()
                         if i.name in ("input_ids", "attention_mask")]
        self.out_name = self.sess.get_outputs()[0].name
        cfg = json.load(open(os.path.join(GTE_DIR, "onnx_config.json")))
        self.qid, self.did, self.pad = (cfg["query_prefix_id"], cfg["document_prefix_id"],
                                        cfg["pad_token_id"])
        self.cls, self.sep = 50281, 50282
        self.projection = self._probe_projection()

    def _probe_projection(self):
        outs = self._run([self.tok.encode("[Q] 测试test").ids])
        if outs.shape[-1] != 768:
            return None  # ONNX 已含投影头
        from safetensors import safe_open
        with safe_open(os.path.join(GTE_DIR, "model.safetensors"), framework="numpy") as f:
            for k in ("embedding.weight", "projection.weight", "colbert_linear.out_proj.weight"):
                if k in f.keys():
                    return f.get_tensor(k).astype(np.float32)
        raise RuntimeError("ONNX 输出 768 维 hidden 且未找到投影头")

    def _run(self, id_lists):
        n = len(id_lists)
        max_len = max(len(x) for x in id_lists)
        ids_b = np.full((n, max_len), self.pad, dtype=np.int64)
        for i, x in enumerate(id_lists):
            ids_b[i, :len(x)] = x
        feeds = {}
        if "input_ids" in self.in_names:
            feeds["input_ids"] = ids_b
        if "attention_mask" in self.in_names:
            feeds["attention_mask"] = (ids_b != self.pad).astype(np.int64)
        return self.sess.run([self.out_name], feeds)[0].astype(np.float32)

    def _tokenize(self, text, prefix):
        ids = self.tok.encode(prefix + text).ids
        keep = np.array([i for i, t in enumerate(ids)
                         if t not in (self.cls, self.sep, self.pad, self.qid, self.did)],
                        dtype=np.int64)
        return ids, keep

    def _encode(self, texts, prefix):
        tok_out = [self._tokenize(t, prefix) for t in texts]
        hs = self._run([ids for ids, _ in tok_out])
        if self.projection is not None:
            hs = hs @ self.projection.T
        hs = hs / np.maximum(np.linalg.norm(hs, axis=-1, keepdims=True), 1e-9)
        return [hs[i][keep] if len(keep) else hs[i, :1]
                for i, (_, keep) in enumerate(tok_out)]

    def score_matrix(self, queries, docs) -> np.ndarray:
        """返回 [len(queries), len(docs)] 的 MaxSim 分数矩阵。"""
        qs = self._encode(list(queries), "[Q] ")   # 每条 [m_i, dim]
        ds = self._encode(list(docs), "[D] ")      # 每条 [n_j, dim]
        S = np.zeros((len(qs), len(ds)), dtype=np.float32)
        for i, q in enumerate(qs):
            for j, d in enumerate(ds):
                S[i, j] = np.max(d @ q.T, axis=0).mean()
        return S


def main():
    scorers = [BgeScorer(), GteMcScorer()]
    mats = {sc.name: sc.score_matrix(QUERIES, DOCS) for sc in scorers}

    print(f"{len(QUERIES)} 条查询 × {len(DOCS)} 条候选\n")
    for i, q in enumerate(QUERIES):
        for j, d in enumerate(DOCS):
            row = "  ".join(f"[{sc.name}] {mats[sc.name][i, j]:.4f}" for sc in scorers)
            print(f"Q: {q}\nD: {d}\n  {row}\n")


if __name__ == "__main__":
    main()
