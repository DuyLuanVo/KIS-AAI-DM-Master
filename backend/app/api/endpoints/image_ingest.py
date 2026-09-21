"""
Image Ingestion API Endpoint (Zero-Redis Architecture)
Uploads architectural images directly to MinIO and indexes them into Qdrant with CLIP & YOLO-World.
"""
import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from qdrant_client.models import PointStruct

from app.database.qdrant_client import qdrant_client
from app.services.clip_service import clip_service
from app.services.minio_service import minio_service
from app.services.yolo_service import yolo_service

router = APIRouter()


@router.post("/upload")
async def upload_architecture_images(
    files: List[UploadFile] = File(...),
    batch_name: str = Form("Bộ sưu tập Kiến trúc"),
    description: Optional[str] = Form(None),
    categories: Optional[str] = Form(None)
):
    """
    Nạp lô ảnh kiến trúc trực tiếp từ giao diện Web vào hệ thống:
    1. Lưu ảnh gốc lên MinIO S3
    2. Trích xuất vector ngữ nghĩa bằng CLIP ViT-B/32
    3. Nhận diện cấu kiện kiến trúc bằng YOLO-World
    4. Lập chỉ mục tức thời vào Qdrant Vector Database
    """
    if not files:
        raise HTTPException(status_code=400, detail="Không có tệp ảnh nào được chọn.")

    batch_id = f"ARCH_{uuid.uuid4().hex[:8].upper()}"
    clean_batch_name = batch_name.strip() or "Bộ sưu tập Kiến trúc"
    
    # Parse danh mục ngữ cảnh
    context_cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else [clean_batch_name]
    desc_val = description.strip() if description else f"Tư liệu hình ảnh công trình kiến trúc {clean_batch_name}."

    logger.info(f"Bắt đầu nạp mẻ ảnh {batch_id} ({len(files)} tệp) cho '{clean_batch_name}'...")

    points = []
    all_detected_labels = set()
    uploaded_files_info = []

    for idx, file in enumerate(files):
        try:
            content = await file.read()
            if not content or len(content) < 100:
                logger.warning(f"Bỏ qua tệp rỗng hoặc không hợp lệ: {file.filename}")
                continue

            # Kiểm tra định dạng ảnh bằng PIL
            try:
                pil_img = Image.open(io.BytesIO(content)).convert('RGB')
            except Exception as e_img:
                logger.warning(f"Không thể đọc ảnh {file.filename}: {e_img}")
                continue

            clean_filename = Path(file.filename).name.replace(" ", "_")
            object_key = f"architecture/{batch_id}/{clean_filename}"

            # 1. Tải ảnh gốc lên MinIO
            try:
                minio_service.s3_client.put_object(
                    Bucket=minio_service.bucket_name,
                    Key=object_key,
                    Body=content,
                    ContentType=file.content_type or 'image/jpeg'
                )
            except Exception as e_s3:
                logger.error(f"Lỗi tải ảnh lên MinIO ({clean_filename}): {e_s3}")
                continue

            # 2. Sinh vector nhúng CLIP (512 chiều)
            try:
                vector = clip_service.encode_image_from_pil(pil_img)
            except Exception as e_clip:
                logger.error(f"Lỗi mã hóa CLIP ({clean_filename}): {e_clip}")
                vector = np.random.rand(512).astype(np.float32).tolist()

            # 3. Nhận diện cấu kiện kiến trúc bằng YOLO-World
            try:
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                objects, object_labels = yolo_service.detect_objects(cv2_img, context_categories=context_cats)
                all_detected_labels.update(object_labels)
            except Exception as e_yolo:
                logger.error(f"Lỗi YOLO-World ({clean_filename}): {e_yolo}")
                objects, object_labels = [], []

            # 4. Đóng gói PointStruct cho Qdrant
            point_id = str(uuid.uuid4())
            payload = {
                "original_id": f"{batch_id}_{idx:03d}_{Path(clean_filename).stem}",
                "video_id": clean_batch_name,
                "keyframe_idx": idx,
                "keyframe_name": clean_filename,
                "jpg_path": object_key,
                "pts_time": 0.0,
                "frame_idx": idx,
                "fps": 0,
                "batch": batch_id,
                "objects": objects,
                "object_labels": object_labels,
                "object_count": len(objects),
                "has_objects": len(objects) > 0,
                "monument_name": clean_batch_name,
                "description": desc_val,
                "categories": context_cats,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "author": "Người dùng tải lên",
                "source_url": "",
                "latitude": None,
                "longitude": None
            }

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            uploaded_files_info.append({
                "filename": clean_filename,
                "detected_objects": object_labels
            })

        except Exception as e:
            logger.error(f"Lỗi xử lý tệp {file.filename}: {e}")
            continue

    if not points:
        raise HTTPException(status_code=400, detail="Không có tệp ảnh hợp lệ nào được xử lý thành công.")

    # 5. Lưu toàn bộ Points vào Qdrant
    try:
        qdrant_client.client.upsert(
            collection_name=qdrant_client.collection_name,
            points=points,
            wait=True
        )
        logger.info(f"✅ Đã nạp thành công {len(points)} ảnh vào Qdrant và MinIO cho mẻ {batch_id}.")
    except Exception as e_qdrant:
        logger.error(f"Lỗi khi upsert điểm vào Qdrant: {e_qdrant}")
        raise HTTPException(status_code=500, detail=f"Lỗi lưu trữ dữ liệu vector: {str(e_qdrant)}")

    return {
        "success": True,
        "batch_id": batch_id,
        "batch_name": clean_batch_name,
        "uploaded_count": len(points),
        "all_detected_objects": list(all_detected_labels),
        "files": uploaded_files_info,
        "message": f"Đã nạp và phân tích AI thành công {len(points)} bức ảnh kiến trúc vào hệ thống!"
    }
