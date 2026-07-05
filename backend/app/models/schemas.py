"""
Pydantic schemas for API requests and responses
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class VectorSearchRequest(BaseModel):
    """Request model for vector search"""
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None, description="Filter conditions")


class VectorSearchResult(BaseModel):
    """Individual search result"""
    id: str
    score: float
    payload: Dict[str, Any]


class VectorSearchResponse(BaseModel):
    """Response model for vector search"""
    results: List[VectorSearchResult]
    total_found: int
    query_time_ms: float


class VectorUploadRequest(BaseModel):
    """Request model for uploading vectors"""
    vectors: List[List[float]] = Field(..., description="List of vectors to upload")
    ids: List[str] = Field(..., description="IDs for each vector")
    payloads: Optional[List[Dict[str, Any]]] = Field(default=None, description="Payload data for each vector")


class VectorUploadResponse(BaseModel):
    """Response model for vector upload"""
    success: bool
    uploaded_count: int
    message: str


class VectorDeleteRequest(BaseModel):
    """Request model for deleting vectors"""
    ids: List[str] = Field(..., description="IDs of vectors to delete")


class VectorDeleteResponse(BaseModel):
    """Response model for vector deletion"""
    success: bool
    deleted_count: int
    message: str


class CollectionInfoResponse(BaseModel):
    """Response model for collection information"""
    name: str
    vector_size: int
    points_count: int
    status: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Video Search API Schemas
class VideoTextSearchRequest(BaseModel):
    """Request model for text-based video search"""
    query_texts: List[str] = Field(
        ..., description="List of text queries"
    )
    object_filters: Optional[List[str]] = Field(
        default=None, description="Filter by object types"
    )
    limit: int = Field(default=500, ge=1, le=1000)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class VideoImageSearchRequest(BaseModel):
    """Request model for image-based video search"""
    image_base64: str = Field(..., description="Base64 encoded image")
    object_filters: Optional[List[str]] = Field(
        default=None, description="Filter by object types"
    )
    limit: int = Field(default=500, ge=1, le=1000)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class VideoSearchResult(BaseModel):
    """Individual video search result"""
    rank: int = Field(..., description="Ranking position")
    original_id: str = Field(..., description="Original frame ID")
    video_id: str = Field(..., description="Video ID")
    keyframe_idx: int = Field(..., description="Keyframe index")
    jpg_path: str = Field(..., description="Path to JPEG file")
    pts_time: float = Field(..., description="Timestamp in video")
    frame_idx: int = Field(..., description="Frame index")
    similarity_score: float = Field(..., description="Similarity score")
    objects: List[Dict[str, Any]] = Field(
        default=[], description="Detected objects"
    )


class VideoGroupedResult(BaseModel):
    """Video search results grouped by video"""
    video_id: str = Field(..., description="Video ID")
    total_frames: int = Field(..., description="Number of matching frames")
    best_score: float = Field(..., description="Highest similarity score")
    frames: List[VideoSearchResult] = Field(
        ..., description="List of matching frames"
    )


class VideoSearchResponse(BaseModel):
    """Response model for video search"""
    total_results: int = Field(..., description="Total number of results")
    query_time_ms: float = Field(..., description="Query execution time")
    results: List[VideoSearchResult] = Field(
        ..., description="List of search results"
    )
    grouped_by_video: List[VideoGroupedResult] = Field(
        ..., description="Results grouped by video"
    )
