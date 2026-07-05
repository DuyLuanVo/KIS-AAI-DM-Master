"""
FastAPI application entry point
"""
from contextlib import asynccontextmanager

from app.api.api import api_router
from app.config import settings
from app.core.logging import setup_logging
from app.database.qdrant_client import qdrant_client
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FastAPI application...")

    # Setup logging
    setup_logging()

    # Initialize Qdrant collection
    logger.info("Initializing Qdrant collection...")
    success = await qdrant_client.init_collection()
    if not success:
        logger.error("Failed to initialize Qdrant collection")
        raise RuntimeError("Qdrant initialization failed")

    logger.info("FastAPI application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Video Retrieval API with Qdrant Vector Database",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Retrieval API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "Something went wrong"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
