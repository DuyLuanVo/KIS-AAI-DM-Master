"""
Health check endpoints
"""
from app.models.schemas import HealthResponse
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse()


@router.get("/health/qdrant")
async def qdrant_health_check():
    """Qdrant specific health check"""
    try:
        from app.database.qdrant_client import qdrant_client
        info = qdrant_client.get_collection_info()
        return {"status": "healthy", "qdrant": info}
    except Exception as e:
        from fastapi import Response
        # Return 503 Service Unavailable if Qdrant is unhealthy/offline
        return Response(
            content=f'{{"status": "unhealthy", "error": "{str(e)}"}}',
            media_type="application/json",
            status_code=503
        )
