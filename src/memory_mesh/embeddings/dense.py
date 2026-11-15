from __future__ import annotations

import hashlib
import math
import re
from typing import List


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


class DenseSemanticModel:
    """
    Lightweight, deterministic hashed subword vector embedding encoder.
    Generates normalized fixed-dimensional vectors for semantic cosine similarity
    without requiring 2GB PyTorch or external model weights.
    """

    def __init__(self, dimension: int = 128, ngram_range: tuple[int, int] = (3, 5)):
        self.dimension = dimension
        self.ngram_range = ngram_range

    def _hash_to_index(self, token: str) -> tuple[int, float]:
        # Hash token to dimension index and sign (+1 or -1 for random projection)
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dimension
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        return idx, sign

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        clean_text = f"_{text.lower().strip()}_"
        words = re.findall(r"\w+", clean_text)

        # Word-level features
        for w in words:
            idx, sign = self._hash_to_index(w)
            vec[idx] += 1.5 * sign

        # Character n-grams (robust to typos, morphology, and inflection)
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(clean_text) - n + 1):
                gram = clean_text[i : i + n]
                idx, sign = self._hash_to_index(gram)
                vec[idx] += 1.0 * sign

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [round(x / norm, 5) for x in vec]
        return vec
