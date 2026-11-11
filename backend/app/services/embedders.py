"""
Bộ mã hóa (embedder) cho hai luồng tìm kiếm:

* ImageEmbedder   : ảnh -> vector, dùng cho tìm ảnh tương đồng (image -> image).
* ClipTextEmbedder: câu truy vấn -> vector CLIP, cùng không gian với vector ảnh (text -> image).

Backend "clip" dùng CLIP ViT-B/32. Nếu đặt CLIP_FINETUNED_PATH, trọng số fine-tune (scripts/finetune_clip.py)
được nạp chồng lên CLIP gốc; ảnh và câu truy vấn dùng chung mô hình đó.
Backend "lite" chỉ dùng numpy (histogram màu) để chạy test/demo khi không tải được model; khi đó tìm bằng mô tả
chỉ còn nhánh từ khóa.
Vector luôn được chuẩn hóa L2 nên cosine = tích vô hướng.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import numpy as np
from loguru import logger
from PIL import Image, ImageOps

from app.config import PROJECT_ROOT, settings

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


def to_rgb(img: Image.Image) -> Image.Image:
    """Chuẩn hóa ảnh giống lúc fine-tune (scripts/image_preprocess.open_rgb): xoay đúng chiều EXIF,
    ghép nền trắng cho ảnh trong suốt, chuyển RGB."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


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
            arr = np.asarray(to_rgb(img).resize((96, 96)), dtype=np.int32) // 32
            idx = arr[..., 0] * 64 + arr[..., 1] * 8 + arr[..., 2]
            hist = np.bincount(idx.ravel(), minlength=512).astype(np.float32)
            vecs.append(np.sqrt(hist))
        return _l2(np.stack(vecs))


# ---------------------------------------------------------------- real backend
@lru_cache(maxsize=1)
def _load_clip():
    """Nạp CLIP một lần, dùng chung cho nhánh ảnh và nhánh văn bản."""
    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Đang nạp CLIP {settings.clip_model} trên {device}...")
    model, preprocess = clip.load(settings.clip_model, device=device)
    name, meta = f"clip-{settings.clip_model}", {}
    if settings.clip_finetuned_path:
        path = Path(settings.clip_finetuned_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise RuntimeError(f"Không tìm thấy checkpoint CLIP fine-tune: {path}")
        ck = torch.load(path, map_location="cpu")
        if ck.get("base_model") != settings.clip_model:
            raise RuntimeError(f"Checkpoint fine-tune từ {ck.get('base_model')}, khác CLIP_MODEL={settings.clip_model}")
        res = model.load_state_dict(ck["state_dict"], strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"Checkpoint không khớp mô hình CLIP: {res.unexpected_keys[:5]}")
        name, meta = f"{name}-ft", ck.get("meta") or {}
        logger.info(f"Đã nạp CLIP fine-tune {path.name} (epoch {meta.get('epoch')})")
    model.eval()
    return model, preprocess, device, name, meta


class ClipImageEmbedder:
    def __init__(self):
        import torch

        self._torch = torch
        self.model, self.preprocess, self.device, self.name, _ = _load_clip()
        self.dim = int(self.model.visual.output_dim)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self.preprocess(to_rgb(im)) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch).float()
        return _l2(feats.cpu().numpy())


class ClipTextEmbedder:
    """Mã hóa câu truy vấn bằng text encoder của CLIP để so trực tiếp với vector "image"."""

    def __init__(self):
        import clip
        import torch

        self._clip, self._torch = clip, torch
        self.model, _, self.device, base, _ = _load_clip()
        self.name = f"{base}-text"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        tokens = self._clip.tokenize(list(texts), truncate=True).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens).float()
        return _l2(feats.cpu().numpy())


@lru_cache(maxsize=1)
def get_image_embedder():
    if settings.embedder_backend == "lite":
        return LiteImageEmbedder()
    return ClipImageEmbedder()


@lru_cache(maxsize=1)
def get_clip_text_embedder():
    """None khi không dùng được nhánh CLIP text -> image (backend lite hoặc đã tắt)."""
    if settings.embedder_backend == "lite" or not settings.clip_text_search:
        return None
    return ClipTextEmbedder()
