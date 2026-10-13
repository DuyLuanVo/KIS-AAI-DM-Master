"""
Cấu hình ứng dụng KIS Architect.

Mọi giá trị đều đọc được từ biến môi trường hoặc file backend/.env (không phân biệt hoa thường).
"""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / "backend" / ".env"), case_sensitive=False, extra="ignore")

    # Ứng dụng
    app_name: str = "KIS Architect API"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    log_level: str = "INFO"
    # Phục vụ giao diện frontend/ ngay tại http://localhost:8000/
    serve_frontend: bool = True
    frontend_dir: str = str(PROJECT_ROOT / "frontend")

    # Qdrant: "server" (Docker, mặc định) | "local" (lưu vào thư mục, không cần Docker) | "memory" (test)
    qdrant_mode: str = "server"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_path: str = str(PROJECT_ROOT / "qdrant_local")
    qdrant_collection: str = "kis_architecture"

    # Lưu trữ ảnh: "minio" (mặc định) | "local" (đọc/ghi thư mục trên máy)
    storage_backend: str = "minio"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "kis-keyframes"
    minio_secure: bool = False
    # Thư mục ảnh của bộ dữ liệu gốc (key "dataset/<tên file>")
    dataset_dir: str = str(PROJECT_ROOT / "data" / "vietnam_architecture")
    # Thư mục lưu ảnh người dùng tải lên khi storage_backend=local (key "uploads/...")
    media_dir: str = str(PROJECT_ROOT / "media")

    # Mô hình nhúng: "clip" (CLIP + sentence-transformers, mặc định) | "lite" (nhẹ, không cần GPU/tải model; dùng cho test/demo)
    embedder_backend: str = "clip"
    clip_model: str = "ViT-B/32"
    # Model nhúng caption. Caption là tiếng Anh; dùng model đa ngôn ngữ nếu muốn gõ truy vấn tiếng Việt,
    # ví dụ: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    text_model: str = "sentence-transformers/all-mpnet-base-v2"

    # Tìm kiếm
    default_limit: int = 60
    max_limit: int = 300
    rrf_k: int = 60  # hằng số Reciprocal Rank Fusion khi trộn kết quả ngữ nghĩa + từ khóa
    min_caption_chars: int = 10


settings = Settings()
