"""
Nạp ảnh mới vào kho. Mỗi ảnh BẮT BUỘC đi kèm caption, vì tìm kiếm bằng mô tả truy vấn trực tiếp trên caption.

POST /api/v1/ingest/upload (multipart/form-data)
  files       : nhiều tệp ảnh
  captions    : JSON, dạng mảng cùng thứ tự với files  ["caption ảnh 1", ...]
                hoặc dạng object theo tên tệp          {"a.jpg": "caption...", ...}
  batch_name  : tên bộ sưu tập / công trình (tùy chọn)
  categories  : thể loại, phân tách bằng dấu phẩy (tùy chọn)
  description : mô tả chung cho cả lô (tùy chọn)
  detect_objects : true để nhận diện cấu kiện bằng YOLO-World (chậm hơn)
"""
import io
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.database.vector_store import get_store
from app.models.schemas import IngestFileResult, IngestResponse
from app.services.indexer import build_payload, clean_caption, index_items, title_from_filename
from app.services.storage import get_storage, media_url

router = APIRouter()

MAX_FILES = 200
MAX_BYTES = 25 * 1024 * 1024
_SAFE_RE = re.compile(r"[^\w.\-()]+", re.UNICODE)


def _safe_name(name: str) -> str:
    base = Path(name or "image").name
    stem, ext = Path(base).stem, Path(base).suffix.lower() or ".jpg"
    return (_SAFE_RE.sub("_", stem).strip("_") or "image")[:120] + ext


def _parse_captions(raw: Optional[str], files: List[UploadFile]) -> List[str]:
    if not raw:
        return [""] * len(files)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Trường captions phải là JSON (mảng hoặc object theo tên tệp).")
    if isinstance(data, list):
        if len(data) != len(files):
            raise HTTPException(status_code=400,
                                detail=f"Có {len(files)} ảnh nhưng {len(data)} caption. Số lượng phải bằng nhau.")
        return [str(c or "") for c in data]
    if isinstance(data, dict):
        by_name: Dict[str, str] = {}
        for k, v in data.items():
            by_name[Path(str(k)).name.lower()] = str(v or "")
            by_name[Path(str(k)).stem.lower()] = str(v or "")
        out = []
        for f in files:
            n = Path(f.filename or "").name.lower()
            out.append(by_name.get(n) or by_name.get(Path(n).stem, ""))
        return out
    raise HTTPException(status_code=400, detail="Trường captions phải là mảng hoặc object.")


@router.post("/upload", response_model=IngestResponse)
async def upload_images(
    files: List[UploadFile] = File(...),
    captions: Optional[str] = Form(None),
    batch_name: str = Form(""),
    categories: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    detect_objects: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="Chưa chọn ảnh nào.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Mỗi lần chỉ nạp tối đa {MAX_FILES} ảnh.")

    caps = _parse_captions(captions, files)
    batch_id = f"UP_{uuid.uuid4().hex[:8].upper()}"
    user_batch_name = batch_name.strip()
    batch_name = user_batch_name or "Ảnh tải lên"
    cats = [c.strip() for c in (categories or "").split(",") if c.strip()]
    storage = get_storage()

    results: List[IngestFileResult] = []
    images, payloads, used_names = [], [], set()
    yolo = None

    for f, cap in zip(files, caps):
        fname = Path(f.filename or "image").name
        caption = clean_caption(cap)
        if len(caption) < settings.min_caption_chars:
            results.append(IngestFileResult(file_name=fname, status="skipped",
                                            reason=f"Thiếu caption (cần ít nhất {settings.min_caption_chars} ký tự)."))
            continue
        data = await f.read()
        if len(data) > MAX_BYTES:
            results.append(IngestFileResult(file_name=fname, status="skipped", reason="Ảnh lớn hơn 25 MB."))
            continue
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError):
            results.append(IngestFileResult(file_name=fname, status="skipped", reason="Không đọc được tệp ảnh."))
            continue

        safe = _safe_name(fname)
        if safe in used_names:
            safe = f"{Path(safe).stem}_{len(used_names)}{Path(safe).suffix}"
        used_names.add(safe)
        key = f"uploads/{batch_id}/{safe}"

        objects = []
        if detect_objects:
            try:
                if yolo is None:
                    import cv2
                    import numpy as np
                    from app.services.yolo_service import yolo_service as yolo
                objects, _ = yolo.detect_objects(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                                                 context_categories=cats or [batch_name, caption])
            except Exception as e:  # YOLO lỗi không làm hỏng cả lô
                logger.warning(f"YOLO lỗi với {fname}: {e}")

        try:
            storage.put(key, data, content_type=f.content_type or "image/jpeg")
        except Exception as e:
            logger.error(f"Không lưu được {fname}: {e}")
            results.append(IngestFileResult(file_name=fname, status="skipped", reason="Lỗi lưu ảnh vào kho lưu trữ."))
            continue

        payload = build_payload(key=key, file_name=safe, caption=caption, source="upload", batch_id=batch_id,
                                batch_name=batch_name, categories=cats, description=description,
                                objects=objects, width=img.width, height=img.height)
        if not user_batch_name:
            payload["title"] = title_from_filename(fname)
        images.append(img)
        payloads.append(payload)
        results.append(IngestFileResult(file_name=fname, status="indexed", key=key, image_url=media_url(key),
                                        object_labels=payload["object_labels"]))

    indexed = 0
    if payloads:
        try:
            indexed = index_items(images, payloads)
        except Exception as e:
            logger.exception("Lập chỉ mục thất bại")
            for p in payloads:
                storage.delete(p["key"])
            raise HTTPException(status_code=500, detail=f"Lỗi lập chỉ mục vector: {e}")

    skipped = len(results) - indexed
    msg = (f"Đã nạp {indexed} ảnh vào kho" + (f", bỏ qua {skipped} ảnh." if skipped else ".")) if indexed \
        else "Không nạp được ảnh nào. Kiểm tra lý do ở từng tệp."
    return IngestResponse(success=indexed > 0, batch_id=batch_id if indexed else None, batch_name=batch_name,
                          indexed_count=indexed, skipped_count=skipped, files=results, message=msg)


@router.delete("/batches/{batch_id}")
async def delete_batch(batch_id: str):
    """Hoàn tác một lần tải lên: xóa điểm vector và ảnh đã lưu."""
    if not batch_id.startswith("UP_"):
        raise HTTPException(status_code=400, detail="Chỉ xóa được các lô do người dùng tải lên.")
    keys = get_store().delete_batch(batch_id)
    storage = get_storage()
    for k in keys:
        try:
            storage.delete(k)
        except Exception:
            pass
    return {"success": True, "deleted": len(keys), "batch_id": batch_id}
