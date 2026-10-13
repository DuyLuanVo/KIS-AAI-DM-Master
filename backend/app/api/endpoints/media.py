"""
GET /media/{key}: trả ảnh gốc (local) hoặc chuyển hướng tới presigned URL (MinIO).
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from app.services.storage import get_storage

router = APIRouter()


@router.get("/media/{key:path}")
async def get_media(key: str):
    storage = get_storage()
    path = storage.local_path(key)
    if path is not None:
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
    try:
        url = storage.presigned_url(key)
    except Exception:
        url = None
    if not url:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh.")
    return RedirectResponse(url)
