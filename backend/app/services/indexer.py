"""
Logic lập chỉ mục dùng chung cho API tải lên và script index_dataset.py:
ảnh + caption -> (vector ảnh, vector caption, payload) -> Qdrant.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from app.database.vector_store import get_store
from app.services.embedders import get_caption_embedder, get_image_embedder

_PAGEID_RE = re.compile(r"_\d{5,}$")
_NOISE_RE = re.compile(r"\(\d{6,}\)|\s-\s\d{6,}|\.(png|jpe?g|webp|tiff?)$", re.IGNORECASE)


def clean_title(text: str) -> str:
    text = _NOISE_RE.sub("", (text or "").replace("_", " "))
    return re.sub(r"\s+", " ", text).strip(" ,-")


def title_from_filename(name: str) -> str:
    stem = _PAGEID_RE.sub("", Path(name).stem)
    return clean_title(stem) or Path(name).stem


def clean_caption(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def build_payload(*, key: str, file_name: str, caption: str, source: str,
                  meta: Optional[Dict[str, Any]] = None, batch_id: Optional[str] = None,
                  batch_name: Optional[str] = None, categories: Optional[List[str]] = None,
                  description: Optional[str] = None, objects: Optional[List[Dict[str, Any]]] = None,
                  width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
    meta = meta or {}
    objects = objects or []
    labels = sorted({o.get("label") for o in objects if o.get("label")})
    return {
        "key": key,
        "file_name": file_name,
        "caption": caption,
        "title": (clean_title(meta.get("monument_name") or "") or batch_name or title_from_filename(file_name))[:300],
        "description": (description or meta.get("description") or "")[:4000],
        "categories": categories if categories is not None else (meta.get("categories") or [])[:30],
        "depicts": meta.get("depicts") or [],
        "date": meta.get("date") or "",
        "author": meta.get("artist") or meta.get("author") or "",
        "license": meta.get("license") or "",
        "source_url": meta.get("source_url") or "",
        "latitude": meta.get("latitude"),
        "longitude": meta.get("longitude"),
        "source": source,
        "batch_id": batch_id,
        "batch_name": batch_name,
        "objects": objects,
        "object_labels": labels,
        "width": width,
        "height": height,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
    }


def ensure_collection(recreate: bool = False) -> None:
    get_store().ensure_collection(get_image_embedder().dim, get_caption_embedder().dim, recreate=recreate)


def index_items(images: Sequence[Image.Image], payloads: Sequence[Dict[str, Any]]) -> int:
    """Mã hóa một lô ảnh + caption và ghi vào Qdrant."""
    if not payloads:
        return 0
    img_vecs = get_image_embedder().encode(images)
    cap_vecs = get_caption_embedder().encode([p["caption"] for p in payloads])
    items = [
        {"key": p["key"], "image_vector": iv, "caption_vector": cv, "payload": p}
        for p, iv, cv in zip(payloads, img_vecs, cap_vecs)
    ]
    return get_store().upsert(items)
