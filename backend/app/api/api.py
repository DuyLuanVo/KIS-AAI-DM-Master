"""
API router configuration
"""
from app.api.endpoints import video_search
from fastapi import APIRouter

# Create main API router
api_router = APIRouter()

# Include endpoint routers
# api_router.include_router(health.router, tags=["health"])
# api_router.include_router(vectors.router, prefix="/api/v1", tags=["vectors"])
api_router.include_router(
    video_search.router,
    prefix="/api/v1/videos",
    tags=["video-search"]
)
