"""
KIS Architect API: tìm kiếm kho ảnh kiến trúc Việt Nam bằng hình ảnh hoặc bằng mô tả (caption).
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.api import api_router
from app.config import settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    from app.services.indexer import ensure_collection
    from app.services.storage import get_storage

    # Nạp mô hình + kiểm tra collection ngay khi khởi động để truy vấn đầu tiên không bị chậm
    ensure_collection()
    get_storage()
    logger.info("KIS Architect API sẵn sàng")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Tìm ảnh kiến trúc: bằng ảnh (CLIP image embedding) hoặc bằng mô tả (truy vấn caption).",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)


@app.get("/api")
async def api_root():
    return {"message": settings.app_name, "version": settings.app_version, "docs": "/docs", "health": "/health"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status_code": exc.status_code})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Lỗi không xử lý: {exc}")
    return JSONResponse(status_code=500, content={"error": "Lỗi máy chủ", "detail": str(exc) if settings.debug else None})


# Giao diện web tại http://localhost:8000/ (đặt sau cùng để không che các route API)
if settings.serve_frontend and Path(settings.frontend_dir).is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
