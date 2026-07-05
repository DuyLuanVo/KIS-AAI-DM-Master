"""
Health check endpoints
"""
from app.models.schemas import HealthResponse
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    pass


@router.get("/health/qdrant")
async def qdrant_health_check():
    """Qdrant specific health check"""
    pass
