"""
Bộ mã hóa (embedder) cho hai luồng tìm kiếm:

* ImageEmbedder  : ảnh  -> vector, dùng cho tìm ảnh tương đồng (image -> image).
* CaptionEmbedder: văn bản -> vector, dùng cho cả caption trong kho lẫn câu truy vấn (text -> caption).

Backend "clip" dùng CLIP ViT-B/32 cho ảnh và sentence-transformers cho caption.
Backend "lite" chỉ dùng numpy (histogram màu + hashing từ khóa) để chạy test/demo khi không tải được model.
Vector luôn được chuẩn hóa L2 nên cosine = tích vô hướng.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import List, Sequence

import numpy as np
from loguru import logger
from PIL import Image

from app.config import settings

_TOKEN_RE = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "with", "by", "for", "from", "is", "are",
    "its", "it", "this", "that", "into", "under", "over", "near", "beside", "behind", "front", "visible",
    "image", "photo", "picture", "showing", "shows", "some", "several", "two", "three", "one", "has", "have",
    "which", "while", "their", "there", "as", "be", "out", "up", "down", "along", "between", "other",
}


def tokenize(text: str) -> List[str]:
    """Tách từ đơn giản (chữ thường, bỏ stopword, đưa số nhiều tiếng Anh về số ít)."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# ---------------------------------------------------------------- lite backend
class LiteImageEmbedder:
    """Histogram màu 8x8x8 (512 chiều) + căn bậc hai. Đủ để tìm ảnh trùng/gần trùng trong demo."""

    name = "lite-color-hist"
    dim = 512

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        vecs = []
        for img in images:
            arr = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.int32) // 32
            idx = arr[..., 0] * 64 + arr[..., 1] * 8 + arr[..., 2]
            hist = np.bincount(idx.ravel(), minlength=512).astype(np.float32)
            vecs.append(np.sqrt(hist))
        return _l2(np.stack(vecs))


class LiteCaptionEmbedder:
    """Hashing bag-of-words + bigram (512 chiều), trọng số log(1+tf)."""

    name = "lite-hashing"
    dim = 512

    def _bucket(self, term: str) -> int:
        return int(hashlib.md5(term.encode("utf-8")).hexdigest()[:8], 16) % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            terms = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for term in terms:
                out[i, self._bucket(term)] += 1.0
        out = np.log1p(out)
        return _l2(out)


# ---------------------------------------------------------------- real backend
class ClipImageEmbedder:
    def __init__(self, model_name: str):
        import clip
        import torch

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Đang nạp CLIP {model_name} trên {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.name = f"clip-{model_name}"
        self.dim = int(self.model.visual.output_dim)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch).float()
        return _l2(feats.cpu().numpy())


class SentenceCaptionEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Đang nạp mô hình nhúng caption {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)


@lru_cache(maxsize=1)
def get_image_embedder():
    if settings.embedder_backend == "lite":
        return LiteImageEmbedder()
    return ClipImageEmbedder(settings.clip_model)


@lru_cache(maxsize=1)
def get_caption_embedder():
    if settings.embedder_backend == "lite":
        return LiteCaptionEmbedder()
    return SentenceCaptionEmbedder(settings.text_model)
