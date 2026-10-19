"""
Kho vector Qdrant cho ảnh kiến trúc.

Mỗi điểm (point) là một ảnh với 2 vector có tên:
* "image"   : embedding ảnh (CLIP)               -> tìm bằng ảnh
* "caption" : embedding caption (sentence model) -> tìm bằng mô tả

Payload chứa caption gốc (có chỉ mục full-text) và metadata công trình.
"""
from __future__ import annotations

import math
import uuid
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    PointVectors,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from app.config import settings
from app.services.embedders import tokenize

POINT_NAMESPACE = uuid.UUID("6f1c2a8e-0b9d-4d0a-9a51-4b2f2d1e7c11")


def point_id_for(key: str) -> str:
    """ID ổn định theo key ảnh: chạy lại indexer sẽ ghi đè thay vì nhân bản."""
    return str(uuid.uuid5(POINT_NAMESPACE, key))


class VectorStore:
    def __init__(self):
        mode = settings.qdrant_mode
        if mode == "memory":
            self.client = QdrantClient(location=":memory:")
        elif mode == "local":
            self.client = QdrantClient(path=settings.qdrant_path)
        else:
            self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)
        self.mode = mode
        self.collection = settings.qdrant_collection
        self._stats_cache: Optional[Dict[str, Any]] = None
        logger.info(f"Qdrant mode={mode}, collection={self.collection}")

    # ------------------------------------------------------------ schema
    def ensure_collection(self, image_dim: int, caption_dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            params = self.client.get_collection(self.collection).config.params.vectors
            dims = {k: v.size for k, v in params.items()} if isinstance(params, dict) else {}
            want = {"image": image_dim, "caption": caption_dim}
            if dims != want:
                raise RuntimeError(
                    f"Collection '{self.collection}' có kích thước vector {dims}, khác cấu hình hiện tại {want}. "
                    "Hãy chạy lại: python scripts/index_dataset.py --recreate"
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                "image": VectorParams(size=image_dim, distance=Distance.COSINE),
                "caption": VectorParams(size=caption_dim, distance=Distance.COSINE),
            },
        )
        if self.mode == "server":
            self.client.create_payload_index(
                self.collection,
                "caption",
                field_schema=TextIndexParams(
                    type="text", tokenizer=TokenizerType.PREFIX, min_token_len=2, max_token_len=30, lowercase=True
                ),
            )
            for field in ("object_labels", "batch_id", "key"):
                self.client.create_payload_index(self.collection, field, field_schema=PayloadSchemaType.KEYWORD)
        logger.info(f"Đã tạo collection '{self.collection}' (image={image_dim}, caption={caption_dim})")

    # ------------------------------------------------------------ write
    def upsert(self, items: Sequence[Dict[str, Any]]) -> int:
        """items: [{"key", "image_vector", "caption_vector", "payload"}]"""
        points = [
            PointStruct(
                id=point_id_for(it["key"]),
                vector={"image": list(map(float, it["image_vector"])), "caption": list(map(float, it["caption_vector"]))},
                payload={**it["payload"], "key": it["key"]},
            )
            for it in items
        ]
        if points:
            self.client.upsert(self.collection, points=points, wait=True)
            self._stats_cache = None
        return len(points)

    def update_image_vectors(self, items: Sequence[Dict[str, Any]], image_model: str) -> int:
        """items: [{"key", "image_vector"}]. Thay vector "image" (vd. sau khi đổi sang CLIP fine-tune), giữ nguyên phần còn lại."""
        if not items:
            return 0
        ids = [point_id_for(it["key"]) for it in items]
        self.client.update_vectors(
            self.collection,
            points=[PointVectors(id=i, vector={"image": list(map(float, it["image_vector"]))}) for i, it in zip(ids, items)],
            wait=True,
        )
        self.client.set_payload(self.collection, payload={"image_model": image_model}, points=ids, wait=True)
        return len(items)

    def delete_batch(self, batch_id: str) -> List[str]:
        flt = Filter(must=[FieldCondition(key="batch_id", match=MatchValue(value=batch_id))])
        keys = [p.payload.get("key") for p in self._scroll(flt, with_payload=["key"])]
        self.client.delete(self.collection, points_selector=FilterSelector(filter=flt), wait=True)
        self._stats_cache = None
        return keys

    def existing_keys(self, keys: Iterable[str]) -> set:
        ids = [point_id_for(k) for k in keys]
        found = self.client.retrieve(self.collection, ids=ids, with_payload=["key"], with_vectors=False)
        return {p.payload.get("key") for p in found}

    # ------------------------------------------------------------ read
    @staticmethod
    def build_filter(object_labels: Optional[List[str]] = None, batch_id: Optional[str] = None, exclude_keys: Optional[List[str]] = None) -> Optional[Filter]:
        must, must_not = [], []
        if object_labels:
            must.append(FieldCondition(key="object_labels", match=MatchAny(any=object_labels)))
        if batch_id:
            must.append(FieldCondition(key="batch_id", match=MatchValue(value=batch_id)))
        if exclude_keys:
            must_not.append(FieldCondition(key="key", match=MatchAny(any=exclude_keys)))
        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)

    def _scroll(self, flt: Optional[Filter], limit: Optional[int] = None, with_payload=True):
        out, offset = [], None
        while True:
            page, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset,
                with_payload=with_payload, with_vectors=False,
            )
            out.extend(page)
            if offset is None or (limit and len(out) >= limit):
                return out[:limit] if limit else out

    def search_vector(self, using: str, vector: Sequence[float], limit: int,
                      flt: Optional[Filter] = None) -> List[Dict[str, Any]]:
        res = self.client.query_points(
            self.collection, query=list(map(float, vector)), using=using,
            query_filter=flt, limit=limit, with_payload=True,
        ).points
        return [{"id": str(p.id), "score": float(p.score), "payload": p.payload} for p in res]

    def get_vector(self, key: str, using: str = "image") -> Optional[List[float]]:
        pts = self.client.retrieve(self.collection, ids=[point_id_for(key)], with_vectors=[using], with_payload=False)
        if not pts:
            return None
        vec = pts[0].vector
        return vec.get(using) if isinstance(vec, dict) else vec

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        pts = self.client.retrieve(self.collection, ids=[point_id_for(key)], with_payload=True)
        return {"id": str(pts[0].id), "payload": pts[0].payload} if pts else None

    def keyword_search(self, query: str, limit: int, flt: Optional[Filter] = None,
                       candidates: int = 600) -> List[Dict[str, Any]]:
        """Tìm caption chứa từ khóa của truy vấn, xếp hạng bằng BM25."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        conds = [FieldCondition(key="caption", match=MatchText(text=t)) for t in terms]
        base_must = list(flt.must or []) if flt else []
        base_not = list(flt.must_not or []) if flt else []
        cand_filter = Filter(should=conds, must=base_must or None, must_not=base_not or None)
        cands = self._scroll(cand_filter, limit=candidates)
        if not cands:
            return []

        # Số tài liệu chứa từng từ để tính IDF
        n_total = max(self.count(), 1)
        df = {}
        for t in terms:
            df[t] = self.client.count(
                self.collection,
                count_filter=Filter(must=[FieldCondition(key="caption", match=MatchText(text=t))]),
                exact=True,
            ).count or 1
        idf = {t: math.log(1 + (n_total - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}

        k1, b, avg_len = 1.2, 0.75, 28.0
        scored = []
        for p in cands:
            toks = tokenize(p.payload.get("caption", ""))
            tf = Counter(toks)
            matched = [t for t in terms if tf.get(t)]
            if not matched:
                continue  # khớp tiền tố nhưng không khớp nguyên từ
            s = sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avg_len)) for t in matched)
            scored.append({"id": str(p.id), "score": s, "payload": p.payload, "matched_terms": matched})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def count(self, flt: Optional[Filter] = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def stats(self) -> Dict[str, Any]:
        if self._stats_cache is not None:
            return self._stats_cache
        labels, batches, total = Counter(), {}, 0
        for p in self._scroll(None, with_payload=["object_labels", "batch_id", "batch_name"]):
            pl = p.payload or {}
            total += 1
            labels.update(pl.get("object_labels") or [])
            if pl.get("batch_id"):
                b = batches.setdefault(pl["batch_id"], {"batch_id": pl["batch_id"], "name": pl.get("batch_name"), "count": 0})
                b["count"] += 1
        self._stats_cache = {
            "total_images": total,
            "object_labels": [{"label": k, "count": v} for k, v in labels.most_common(40)],
            "batches": list(batches.values()),
        }
        return self._stats_cache


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_store() -> None:
    """Dùng trong test."""
    global _store
    _store = None
