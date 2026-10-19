"""
Lưu trữ ảnh gốc.

Key ảnh có dạng:
* "dataset/<tên file>"           : ảnh trong bộ dữ liệu gốc (data/vietnam_architecture)
* "uploads/<batch_id>/<tên file>" : ảnh người dùng tải lên

Frontend luôn lấy ảnh qua GET /media/{key}; backend trả file trực tiếp (local) hoặc chuyển hướng tới
presigned URL (MinIO), nên giao diện không phụ thuộc kiểu lưu trữ.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings


def _safe_join(root: Path, rel: str) -> Optional[Path]:
    p = (root / rel).resolve()
    return p if root.resolve() in p.parents or p == root.resolve() else None


class LocalStorage:
    kind = "local"

    def __init__(self):
        self.roots = {
            "dataset": Path(settings.dataset_dir),
            "uploads": Path(settings.media_dir) / "uploads",
        }
        self.roots["uploads"].mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Optional[Path]:
        prefix, _, rest = key.lstrip("/").partition("/")
        root = self.roots.get(prefix)
        return _safe_join(root, rest) if root and rest else None

    def put(self, key: str, data: bytes, content_type: str = "image/png") -> None:
        p = self._path(key)
        if p is None:
            raise ValueError(f"Key không hợp lệ: {key}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def put_file(self, key: str, src: Path) -> None:
        # Ảnh dataset đã nằm sẵn trong dataset_dir: không cần sao chép
        if key.startswith("dataset/"):
            return
        self.put(key, Path(src).read_bytes())

    def local_path(self, key: str) -> Optional[Path]:
        p = self._path(key)
        return p if p and p.is_file() else None

    def get(self, key: str) -> Optional[bytes]:
        p = self.local_path(key)
        return p.read_bytes() if p else None

    def presigned_url(self, key: str) -> Optional[str]:
        return None

    def delete(self, key: str) -> None:
        p = self.local_path(key)
        if p and key.startswith("uploads/"):
            p.unlink(missing_ok=True)


class MinioStorage:
    kind = "minio"

    def __init__(self):
        import boto3
        from botocore.client import Config

        scheme = "https" if settings.minio_secure else "http"
        self.client = boto3.client(
            "s3",
            endpoint_url=f"{scheme}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self.bucket = settings.minio_bucket
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self.client.create_bucket(Bucket=self.bucket)
                logger.info(f"Đã tạo bucket {self.bucket}")
            except Exception as e:  # MinIO chưa chạy: báo lỗi khi dùng
                logger.warning(f"Không kết nối được MinIO ({e})")

    def put(self, key: str, data: bytes, content_type: str = "image/png") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def put_file(self, key: str, src: Path) -> None:
        ctype = mimetypes.guess_type(str(src))[0] or "image/png"
        self.client.upload_file(str(src), self.bucket, key, ExtraArgs={"ContentType": ctype})

    def local_path(self, key: str) -> Optional[Path]:
        return None

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key.lstrip("/"))["Body"].read()
        except Exception:
            return None

    def presigned_url(self, key: str) -> Optional[str]:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key.lstrip("/")}, ExpiresIn=3600
        )

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        _storage = MinioStorage() if settings.storage_backend == "minio" else LocalStorage()
        logger.info(f"Lưu trữ ảnh: {_storage.kind}")
    return _storage


def media_url(key: str) -> str:
    return f"/media/{key}"
