"""
Tìm kiếm theo 2 luồng:

* Bằng hình ảnh  -> mã hóa ảnh truy vấn bằng CLIP, so sánh với vector "image" -> ảnh tương đồng.
* Bằng mô tả     -> kết hợp
      (1) ngữ nghĩa: nhúng câu truy vấn, so với vector "caption"
      (2) từ khóa  : BM25 trên nội dung caption (chỉ mục full-text)
      (3) hình ảnh : mã hóa câu truy vấn bằng text encoder CLIP (đã fine-tune), so trực tiếp với vector "image"
                     -> tìm được cả chi tiết có trong ảnh nhưng caption không nhắc tới
  rồi trộn bằng Reciprocal Rank Fusion.
"""
import base64
import binascii
import io
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.database.vector_store import get_store
from app.models.schemas import ImageSearchRequest, SearchHit, SearchResponse, TextSearchRequest
from app.services.embedders import get_caption_embedder, get_clip_text_embedder, get_image_embedder, tokenize
from app.services.storage import media_url

router = APIRouter()


def _to_hit(rank: int, payload: Dict[str, Any], **scores) -> SearchHit:
    p = payload or {}
    return SearchHit(
        rank=rank,
        key=p.get("key", ""),
        image_url=media_url(p.get("key", "")),
        title=p.get("title") or p.get("file_name", ""),
        caption=p.get("caption", ""),
        file_name=p.get("file_name", ""),
        description=p.get("description") or "",
        categories=p.get("categories") or [],
        depicts=p.get("depicts") or [],
        date=str(p.get("date") or ""),
        author=p.get("author") or "",
        license=p.get("license") or "",
        source_url=p.get("source_url") or "",
        latitude=p.get("latitude"),
        longitude=p.get("longitude"),
        batch_id=p.get("batch_id"),
        batch_name=p.get("batch_name"),
        objects=p.get("objects") or [],
        object_labels=p.get("object_labels") or [],
        **scores,
    )


def _decode_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải ảnh hợp lệ (hỗ trợ JPG, PNG, WEBP...).")


def _filter(req) -> Any:
    labels = [l.strip().lower() for l in (req.object_labels or []) if l and l.strip()]
    return get_store().build_filter(object_labels=labels or None, batch_id=req.batch_id)


def _image_search(img: Image.Image, limit: int, flt) -> List[SearchHit]:
    vec = get_image_embedder().encode([img])[0]
    return _vector_hits(vec, limit, flt)


def _vector_hits(vec, limit: int, flt) -> List[SearchHit]:
    raw = get_store().search_vector("image", vec, limit=limit, flt=flt)
    return [
        _to_hit(i + 1, r["payload"], score=r["score"], similarity=r["score"], match_sources=["image"])
        for i, r in enumerate(raw)
    ]


# ------------------------------------------------------------------ text -> caption
@router.post("/search/text", response_model=SearchResponse)
async def search_by_text(req: TextSearchRequest):
    t0 = time.time()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Hãy nhập mô tả cần tìm.")
    store = get_store()
    flt = _filter(req)
    pool = min(max(req.limit * 2, 100), settings.max_limit * 2)

    semantic: List[Dict[str, Any]] = []
    keyword: List[Dict[str, Any]] = []
    visual: List[Dict[str, Any]] = []
    clip_text = get_clip_text_embedder()
    if req.mode == "visual" and clip_text is None:
        raise HTTPException(status_code=400, detail="Chế độ 'visual' cần backend CLIP (EMBEDDER_BACKEND=clip).")
    if req.mode in ("hybrid", "visual") and clip_text is not None:
        visual = store.search_vector("image", clip_text.encode([query])[0], limit=pool, flt=flt)
    if req.mode in ("hybrid", "semantic"):
        qvec = get_caption_embedder().encode([query])[0]
        semantic = store.search_vector("caption", qvec, limit=pool, flt=flt)
    if req.mode in ("hybrid", "keyword"):
        keyword = store.keyword_search(query, limit=pool, flt=flt)

    fused: Dict[str, Dict[str, Any]] = {}
    k = settings.rrf_k
    for rank, r in enumerate(semantic):
        e = fused.setdefault(r["id"], {"payload": r["payload"], "rrf": 0.0, "sources": [], "matched": []})
        e["rrf"] += 1.0 / (k + rank + 1)
        e["similarity"] = r["score"]
        e["sources"].append("semantic")
    max_kw = keyword[0]["score"] if keyword else 1.0
    for rank, r in enumerate(keyword):
        e = fused.setdefault(r["id"], {"payload": r["payload"], "rrf": 0.0, "sources": [], "matched": []})
        # Từ khóa xuất hiện nguyên văn trong caption là tín hiệu mạnh: trọng số 1.0 như ngữ nghĩa
        e["rrf"] += 1.0 / (k + rank + 1)
        e["keyword_score"] = r["score"] / max_kw
        e["matched"] = r["matched_terms"]
        e["sources"].append("keyword")
    for rank, r in enumerate(visual):
        e = fused.setdefault(r["id"], {"payload": r["payload"], "rrf": 0.0, "sources": [], "matched": []})
        e["rrf"] += 1.0 / (k + rank + 1)
        e["clip_score"] = r["score"]
        e["sources"].append("visual")

    ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    # Ở chế độ hybrid, kết quả chỉ khớp ngữ nghĩa / hình ảnh mà điểm quá thấp thì loại bỏ
    if req.mode == "hybrid":
        clip_floor = max(settings.clip_min_score, visual[0]["score"] - settings.clip_relative_margin) if visual else 1.0

        def keep(e) -> bool:
            if "keyword" in e["sources"] or e.get("similarity", 0) >= 0.2:
                return True
            if "visual" in e["sources"] and e["clip_score"] >= clip_floor:
                return True
            return not keyword and "semantic" in e["sources"]  # không có từ khóa nào khớp: giữ nhánh ngữ nghĩa

        ordered = [e for e in ordered if keep(e)]
    ordered = ordered[: req.limit]

    hits = [
        _to_hit(
            i + 1, e["payload"], score=round(e["rrf"], 6), similarity=e.get("similarity"),
            keyword_score=e.get("keyword_score"), clip_score=e.get("clip_score"), matched_terms=e["matched"],
            match_sources=e["sources"],
        )
        for i, e in enumerate(ordered)
    ]
    ms = (time.time() - t0) * 1000
    logger.info(f"text search '{query}' -> {len(hits)} kết quả ({ms:.0f} ms)")
    return SearchResponse(search_type="text", query=query, query_terms=list(dict.fromkeys(tokenize(query))),
                          total_results=len(hits), query_time_ms=ms, results=hits)


# ------------------------------------------------------------------ image -> image
@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(req: ImageSearchRequest):
    t0 = time.time()
    b64 = req.image_base64.split(",", 1)[1] if req.image_base64.startswith("data:") else req.image_base64
    try:
        data = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Chuỗi base64 của ảnh không hợp lệ.")
    hits = _image_search(_decode_image(data), req.limit, _filter(req))
    return SearchResponse(search_type="image", total_results=len(hits),
                          query_time_ms=(time.time() - t0) * 1000, results=hits)


@router.post("/search/image/upload", response_model=SearchResponse)
async def search_by_image_upload(
    file: UploadFile = File(...),
    limit: int = Form(60),
    object_labels: Optional[str] = Form(None, description="Danh sách nhãn, phân tách bằng dấu phẩy"),
):
    """Giống /search/image nhưng nhận file multipart (tiện cho Swagger UI)."""
    t0 = time.time()
    req = ImageSearchRequest(image_base64="x", limit=max(1, min(limit, settings.max_limit)),
                             object_labels=[s for s in (object_labels or "").split(",") if s.strip()] or None)
    hits = _image_search(_decode_image(await file.read()), req.limit, _filter(req))
    return SearchResponse(search_type="image", total_results=len(hits),
                          query_time_ms=(time.time() - t0) * 1000, results=hits)


@router.get("/images/similar", response_model=SearchResponse)
async def similar_images(key: str, limit: int = 60):
    """Ảnh tương đồng với một ảnh đã có trong kho (dùng lại vector đã lưu, không mã hóa lại)."""
    t0 = time.time()
    store = get_store()
    vec = store.get_vector(key, "image")
    if vec is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh trong kho.")
    hits = _vector_hits(vec, max(1, min(limit, settings.max_limit)), store.build_filter(exclude_keys=[key]))
    return SearchResponse(search_type="similar", query=key, total_results=len(hits),
                          query_time_ms=(time.time() - t0) * 1000, results=hits)


@router.get("/images/list", response_model=SearchResponse)
async def list_images(batch_id: Optional[str] = None, limit: int = 120):
    """Duyệt ảnh theo lô tải lên hoặc nguồn (không xếp hạng)."""
    t0 = time.time()
    store = get_store()
    pts = store._scroll(store.build_filter(batch_id=batch_id), limit=max(1, min(limit, settings.max_limit)))
    hits = [_to_hit(i + 1, p.payload, score=0.0) for i, p in enumerate(pts)]
    return SearchResponse(search_type="browse", query=batch_id, total_results=len(hits),
                          query_time_ms=(time.time() - t0) * 1000, results=hits)


@router.get("/images/item", response_model=SearchHit)
async def get_image(key: str):
    item = get_store().get(key)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh trong kho.")
    return _to_hit(1, item["payload"], score=1.0)


@router.get("/stats")
async def stats():
    s = dict(get_store().stats())
    s["collection"] = settings.qdrant_collection
    s["image_model"] = get_image_embedder().name
    s["caption_model"] = get_caption_embedder().name
    clip_text = get_clip_text_embedder()
    s["clip_text_model"] = clip_text.name if clip_text else None
    s["storage"] = settings.storage_backend
    return s
