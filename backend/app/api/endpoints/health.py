"""
Health check endpoints
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(version=settings.app_version)


@router.get("/health/qdrant")
async def qdrant_health_check():
    try:
        from app.database.vector_store import get_store

        store = get_store()
        return {"status": "healthy", "collection": store.collection, "points": store.count()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})
