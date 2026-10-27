# -*- coding: utf-8 -*-
"""TextEmbedder：BGE 文本向量封装（重建版）。"""

import threading


class TextEmbedder:
    """sentence-transformers 文本编码器，encode() 返回 (N, dim) float32 数组。"""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        if max_length:
            self.model.max_seq_length = max_length
        self._lock = threading.Lock()

    def encode(self, texts, normalize: bool = True, batch_size: int = 32):
        if isinstance(texts, str):
            texts = [texts]
        with self._lock:
            return self.model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
