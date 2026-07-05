"""
API router configuration
"""
from app.api.endpoints import health, video_search
from fastapi import APIRouter

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(
    video_search.router,
    prefix="/api/v1/videos",
    tags=["video-search"]
)
