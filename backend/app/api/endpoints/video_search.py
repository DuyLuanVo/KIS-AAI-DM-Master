"""
Video search endpoints
"""
import time
from typing import Any, Dict, List, Optional

from app.database.qdrant_client import qdrant_client
from app.models.schemas import (
    VideoGroupedResult,
    VideoImageSearchRequest,
    VideoSearchResponse,
    VideoSearchResult,
    VideoTextSearchRequest,
)
from app.services.clip_service import clip_service
from app.services.minio_service import minio_service
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
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

        # Generate presigned URL
        image_url = None
        try:
            image_url = minio_service.generate_presigned_url(payload["jpg_path"])
        except Exception as e:
            logger.error(f"Error generating presigned URL for {payload['jpg_path']}: {e}")

        search_result = VideoSearchResult(
            rank=result["rank"],
            original_id=payload["original_id"],
            video_id=payload["video_id"],
            keyframe_idx=payload["keyframe_idx"],
            jpg_path=payload["jpg_path"],
            image_url=image_url,
            pts_time=payload["pts_time"],
            frame_idx=payload["frame_idx"],
            similarity_score=result["score"],
            objects=payload.get("objects", []),
            monument_name=payload.get("monument_name"),
            description=payload.get("description"),
            categories=payload.get("categories", []),
            date=payload.get("date"),
            author=payload.get("author"),
            source_url=payload.get("source_url"),
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude")
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


# Danh sách tất cả các nhãn cấu kiện kiến trúc / vật thể được hệ thống nhận diện
KNOWN_OBJECT_LABELS = [
    "curved roof", "tiled roof", "wooden column", "temple gate",
    "stone statue", "bell tower", "dragon carving", "arched entrance",
    "arched window", "brick wall", "courtyard", "altar", "steeple",
    "cross", "colonnade", "pediment", "balcony", "dome", "wooden louvers",
    "staircase", "wooden screen", "carved chair", "human scale",
    "person", "car", "motorcycle", "truck", "chair", "dining table", "potted plant"
]

OBJECT_SYNONYMS = {
    # English keywords, abbreviations and synonyms
    "tower": ["bell tower", "steeple"],
    "statue": ["stone statue"],
    "column": ["wooden column", "colonnade"],
    "pillar": ["wooden column", "colonnade"],
    "roof": ["curved roof", "tiled roof"],
    "gate": ["temple gate", "arched entrance"],
    "wall": ["brick wall"],
    "window": ["arched window"],
    "door": ["arched entrance", "wooden screen"],
    "screen": ["wooden screen"],
    "stair": ["staircase"],
    "steps": ["staircase"],
    "dragon": ["dragon carving"],
    "carving": ["dragon carving"],
    "people": ["human scale", "person"],
    "scale": ["human scale"]
}

def normalize_object_filters(filters: Optional[List[str]]) -> Optional[List[str]]:
    if not filters:
        return None
    normalized = set()
    for f in filters:
        clean = f.strip().lower()
        if not clean:
            continue
        normalized.add(clean)

        # 1. Tra cứu từ đồng nghĩa / viết tắt
        if clean in OBJECT_SYNONYMS:
            for syn in OBJECT_SYNONYMS[clean]:
                normalized.add(syn)

        # 2. Khớp từ con / tiền tố / hậu tố (Substring & token matching)
        # Ví dụ: gõ "tower" thì tự động khớp với "bell tower", gõ "statue" khớp "stone statue"
        for known in KNOWN_OBJECT_LABELS:
            if clean in known.split() or clean in known:
                normalized.add(known)
            elif known in clean:
                normalized.add(known)

    return list(normalized) if normalized else None


@router.post("/search/text", response_model=VideoSearchResponse)
async def search_videos_by_text(request: VideoTextSearchRequest):
    """
    Search for video frames based on text queries
    """
    try:
        start_time = time.time()

        normalized_filters = normalize_object_filters(request.object_filters)
        logger.info(f"Text search: {len(request.query_texts)} queries")
        logger.info(f"Object filters: {request.object_filters} -> {normalized_filters}")

        # Encode text queries to vectors
        query_vectors = clip_service.encode_text(request.query_texts)

        # Search in Qdrant
        raw_results = qdrant_client.search_multiple_vectors(
            query_vectors=query_vectors,
            limit=request.limit,
            object_filters=normalized_filters,
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
        normalized_filters = normalize_object_filters(request.object_filters)
        logger.info(f"Object filters: {request.object_filters} -> {normalized_filters}")

        # Encode image to vector
        query_vector = clip_service.encode_image_from_base64(
            request.image_base64
        )

        # Search in Qdrant
        raw_results = qdrant_client.search_by_vector(
            query_vector=query_vector,
            limit=request.limit,
            object_filters=normalized_filters,
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


@router.get("/keyframes/{key:path}")
async def get_keyframe_image(key: str):
    """
    Generate a pre-signed URL and redirect to MinIO to download the image.
    This simplifies loading nearby frames in the carousel.
    """
    try:
        url = minio_service.generate_presigned_url(key)
        return RedirectResponse(url)
    except Exception as e:
        logger.error(f"Failed to generate redirect URL for key {key}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Keyframe image not found: {str(e)}"
        )
