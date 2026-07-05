"""
Video search endpoints
"""
import time
from typing import Any, Dict, List

from app.database.qdrant_client import qdrant_client
from app.models.schemas import (
    VideoGroupedResult,
    VideoImageSearchRequest,
    VideoSearchResponse,
    VideoSearchResult,
    VideoTextSearchRequest,
)
from app.services.clip_service import clip_service
from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter()


def format_search_results(
    raw_results: List[Dict[str, Any]],
    query_time_ms: float
) -> VideoSearchResponse:
    """Format raw search results into API response"""

    # Convert to VideoSearchResult objects
    results = []
    for result in raw_results:
        payload = result["payload"]

        search_result = VideoSearchResult(
            rank=result["rank"],
            original_id=payload["original_id"],
            video_id=payload["video_id"],
            keyframe_idx=payload["keyframe_idx"],
            jpg_path=payload["jpg_path"],
            pts_time=payload["pts_time"],
            frame_idx=payload["frame_idx"],
            similarity_score=result["score"],
            objects=payload.get("objects", [])
        )
        results.append(search_result)

    # Group by video
    grouped_dict = {}
    for result in results:
        video_id = result.video_id
        if video_id not in grouped_dict:
            grouped_dict[video_id] = []
        grouped_dict[video_id].append(result)

    # Create grouped results
    grouped_by_video = []
    for video_id, video_results in grouped_dict.items():
        video_results.sort(key=lambda x: x.similarity_score, reverse=True)

        grouped_result = VideoGroupedResult(
            video_id=video_id,
            total_frames=len(video_results),
            best_score=video_results[0].similarity_score if video_results else 0.0,
            frames=video_results
        )
        grouped_by_video.append(grouped_result)

    # Sort groups by best score
    grouped_by_video.sort(key=lambda x: x.best_score, reverse=True)

    return VideoSearchResponse(
        total_results=len(results),
        query_time_ms=query_time_ms,
        results=results,
        grouped_by_video=grouped_by_video
    )


@router.post("/search/text", response_model=VideoSearchResponse)
async def search_videos_by_text(request: VideoTextSearchRequest):
    """
    Search for video frames based on text queries
    """
    try:
        start_time = time.time()

        logger.info(f"Text search: {len(request.query_texts)} queries")
        logger.info(f"Object filters: {request.object_filters}")

        # Encode text queries to vectors
        query_vectors = clip_service.encode_text(request.query_texts)

        # Search in Qdrant
        raw_results = qdrant_client.search_multiple_vectors(
            query_vectors=query_vectors,
            limit=request.limit,
            object_filters=request.object_filters,
            score_threshold=request.score_threshold
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(raw_results, query_time_ms)

    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text search failed: {str(e)}"
        )


@router.post("/search/image", response_model=VideoSearchResponse)
async def search_videos_by_image(request: VideoImageSearchRequest):
    """
    Search for video frames based on image similarity
    """
    try:
        start_time = time.time()

        logger.info("Image search request received")
        logger.info(f"Object filters: {request.object_filters}")

        # Encode image to vector
        query_vector = clip_service.encode_image_from_base64(
            request.image_base64
        )

        # Search in Qdrant
        raw_results = qdrant_client.search_by_vector(
            query_vector=query_vector,
            limit=request.limit,
            object_filters=request.object_filters,
            score_threshold=request.score_threshold
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(raw_results, query_time_ms)

    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image search failed: {str(e)}"
        )


@router.get("/collection/info")
async def get_collection_info():
    """Get Qdrant collection information"""
    try:
        info = qdrant_client.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}"
        )
