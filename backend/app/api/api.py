"""
API router
"""
from fastapi import APIRouter

from app.api.endpoints import health, ingest, media, search

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(search.router, prefix="/api/v1", tags=["search"])
api_router.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
api_router.include_router(media.router, tags=["media"])
