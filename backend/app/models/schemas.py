"""
Pydantic schemas cho API tìm kiếm & nạp ảnh kiến trúc.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "2.0.0"


class SearchFilters(BaseModel):
    object_labels: Optional[List[str]] = Field(default=None, description="Lọc theo cấu kiện YOLO (khớp bất kỳ)")
    batch_id: Optional[str] = None


class TextSearchRequest(SearchFilters):
    query: str = Field(..., min_length=1, description="Câu mô tả cần tìm (trong caption và nội dung ảnh)")
    limit: int = Field(default=60, ge=1, le=300)
    mode: str = Field(default="hybrid", description="'hybrid' (ngữ nghĩa + từ khóa + CLIP), 'semantic', 'keyword' "
                                                    "hoặc 'visual' (chỉ CLIP text -> image)")


class ImageSearchRequest(SearchFilters):
    image_base64: str = Field(..., description="Ảnh mã hóa base64 (có thể kèm tiền tố data:image/...)")
    limit: int = Field(default=60, ge=1, le=300)


class SearchHit(BaseModel):
    rank: int
    key: str
    image_url: str
    title: str
    caption: str
    score: float = Field(..., description="Điểm xếp hạng cuối cùng")
    similarity: Optional[float] = Field(default=None, description="Cosine của vector (ảnh hoặc caption)")
    keyword_score: Optional[float] = Field(default=None, description="Điểm BM25 đã chuẩn hóa 0-1")
    clip_score: Optional[float] = Field(default=None, description="Cosine CLIP giữa câu truy vấn và ảnh")
    matched_terms: List[str] = []
    match_sources: List[str] = Field(default=[], description="'semantic', 'keyword', 'visual', 'image'")
    file_name: str = ""
    description: str = ""
    categories: List[str] = []
    depicts: List[str] = []
    date: str = ""
    author: str = ""
    license: str = ""
    source_url: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    batch_id: Optional[str] = None
    batch_name: Optional[str] = None
    objects: List[Dict[str, Any]] = []
    object_labels: List[str] = []


class SearchResponse(BaseModel):
    search_type: str = Field(..., description="'text' (truy vấn caption) | 'image' (ảnh tương đồng) | 'similar'")
    query: Optional[str] = None
    query_terms: List[str] = []
    total_results: int
    query_time_ms: float
    results: List[SearchHit]


class IngestFileResult(BaseModel):
    file_name: str
    status: str = Field(..., description="'indexed' hoặc 'skipped'")
    reason: Optional[str] = None
    key: Optional[str] = None
    image_url: Optional[str] = None
    object_labels: List[str] = []


class IngestResponse(BaseModel):
    success: bool
    batch_id: Optional[str]
    batch_name: str
    indexed_count: int
    skipped_count: int
    files: List[IngestFileResult]
    message: str
