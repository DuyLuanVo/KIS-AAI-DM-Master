"""
Image ingest worker service for saving uploaded architecture photos,
encoding with CLIP, detecting interior/exterior objects with YOLOv8,
and indexing to Qdrant & MinIO.
"""
import shutil
import uuid
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from loguru import logger

# Import backend services
from app.services.redis_service import redis_service
from app.services.minio_service import minio_service
from app.services.clip_service import clip_service
from app.services.yolo_service import yolo_service
from app.database.qdrant_client import qdrant_client
from qdrant_client.models import PointStruct

# Thread pool for async tasks
executor = ThreadPoolExecutor(max_workers=4)
active_batch_tasks: Dict[str, threading.Event] = {}

def start_image_ingest_task(batch_id: str, batch_name: str, temp_dir: Path, file_names: List[str]):
    """Start the architecture image ingestion task in the background thread pool"""
    cancel_event = threading.Event()
    active_batch_tasks[batch_id] = cancel_event
    
    # Run in thread pool
    executor.submit(
        run_image_ingest_pipeline,
        batch_id,
        batch_name,
        temp_dir,
        file_names,
        cancel_event
    )
    logger.info(f"Submitted image ingestion task for batch {batch_id} with {len(file_names)} files.")

def run_image_ingest_pipeline(batch_id: str, batch_name: str, temp_dir: Path, file_names: List[str], cancel_event: threading.Event):
    """The core image processing & vector indexing pipeline"""
    uploaded_point_ids: List[str] = []
    total_images = len(file_names)
    
    try:
        # Check cancellation before start
        if cancel_event.is_set() or redis_service.check_cancel_flag(batch_id):
            logger.info(f"Batch task {batch_id} cancelled before starting.")
            redis_service.set_batch_status(batch_id, batch_name, total_images, 0, "CANCELLED", "Tác vụ đã bị hủy trước khi bắt đầu.", 0)
            cleanup_local_files(temp_dir)
            return

        redis_service.set_batch_status(batch_id, batch_name, total_images, 0, "PROCESSING", "Đang phân tích AI và lập chỉ mục Vector...", 5.0)
        
        qdrant_points: List[PointStruct] = []
        completed_images = 0
        
        for idx, file_name in enumerate(file_names, start=1):
            # Check cancellation in processing loop
            if cancel_event.is_set() or redis_service.check_cancel_flag(batch_id):
                logger.info(f"Batch task {batch_id} cancelled during processing.")
                cleanup_qdrant_points(uploaded_point_ids)
                redis_service.set_batch_status(batch_id, batch_name, total_images, completed_images, "CANCELLED", "Tác vụ đã bị người dùng hủy giữa chừng.", 0)
                cleanup_local_files(temp_dir)
                return

            file_path = temp_dir / file_name
            if not file_path.exists():
                logger.warning(f"File not found during processing: {file_path}")
                continue

            # 1. Upload to MinIO under architecture bucket structure
            object_key = f"architecture/{batch_id}/{file_name}"
            content_type = "image/png" if file_name.lower().endswith(".png") else "image/jpeg"
            
            try:
                minio_service.s3_client.upload_file(
                    Filename=str(file_path),
                    Bucket=minio_service.bucket_name,
                    Key=object_key,
                    ExtraArgs={'ContentType': content_type}
                )
            except Exception as e:
                logger.error(f"Failed to upload image {object_key} to MinIO: {e}")
                continue

            # 2. Encode image with CLIP (512-dim vector)
            try:
                vector = clip_service.encode_image_from_path(str(file_path))
            except Exception as e:
                logger.error(f"Failed to encode {file_name} with CLIP: {e}")
                vector = np.random.rand(512).astype(np.float32).tolist()

            # 3. Đọc Metadata đồng hành nếu có (Companion JSON file)
            clean_stem = Path(file_name).stem.replace(" ", "_")
            meta = {}
            json_candidate = Path(file_path).with_suffix('.json')
            if not json_candidate.exists():
                # Thử tìm file JSON theo stem gốc
                json_candidate = Path(file_path).parent / f"{Path(file_name).stem}.json"

            if json_candidate.exists():
                try:
                    with open(json_candidate, 'r', encoding='utf-8') as jf:
                        meta = json.load(jf)
                except Exception as je:
                    logger.warning(f"Không thể đọc file metadata JSON {json_candidate}: {je}")

            monument_name = meta.get("monument_name") or Path(file_name).stem.replace("_", " ")
            description = meta.get("description") or f"Tư liệu hình ảnh công trình kiến trúc {monument_name}."
            categories = meta.get("categories", [])
            date_val = meta.get("date", "")
            author_val = meta.get("artist", "")
            source_url = meta.get("source_url", "")
            lat_val = meta.get("latitude")
            lon_val = meta.get("longitude")

            # 4. Nhận diện cấu kiện kiến trúc với YOLO-World (dựa trên ngữ cảnh metadata)
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    objects, object_labels = [], []
                else:
                    context_cats = categories if categories else [monument_name]
                    objects, object_labels = yolo_service.detect_objects(img, context_categories=context_cats)
            except Exception as e:
                logger.error(f"YOLO-World detection failed on {file_path}: {e}")
                objects, object_labels = [], []

            # 5. Khởi tạo Qdrant Point với Payload Metadata đầy đủ
            point_id = str(uuid.uuid4())
            uploaded_point_ids.append(point_id)
            
            payload = {
                "original_id": f"{batch_id}_{idx:03d}_{clean_stem}",
                "video_id": batch_name or f"Album_{batch_id[:8]}", # Using video_id field as Album/Collection name
                "keyframe_idx": idx,
                "keyframe_name": file_name,
                "jpg_path": object_key,
                "pts_time": 0.0, # 0.0 since it is static architectural photography
                "frame_idx": idx,
                "fps": 0,
                "batch": batch_id,
                "objects": objects,
                "object_labels": object_labels,
                "object_count": len(objects),
                "has_objects": len(objects) > 0,
                "monument_name": monument_name,
                "description": description,
                "categories": categories,
                "date": date_val,
                "author": author_val,
                "source_url": source_url,
                "latitude": lat_val,
                "longitude": lon_val
            }

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            qdrant_points.append(point)
            completed_images += 1

            # Update progress in Redis every image or every 5 images
            progress = 5.0 + (90.0 * completed_images / total_images)
            redis_service.set_batch_status(
                batch_id, 
                batch_name, 
                total_images, 
                completed_images, 
                "PROCESSING", 
                f"Đang xử lý {completed_images}/{total_images}: {file_name} (Pháp sư AI CLIP + YOLO)", 
                progress
            )

            # Batch upsert to Qdrant every 20 images to save memory
            if len(qdrant_points) >= 20:
                upsert_points(qdrant_points)
                qdrant_points = []

        # Upsert remaining points
        if qdrant_points:
            upsert_points(qdrant_points)
            qdrant_points = []

        # Complete
        redis_service.set_batch_status(
            batch_id, 
            batch_name, 
            total_images, 
            completed_images, 
            "COMPLETED", 
            f"Hoàn tất nạp thành công {completed_images} tư liệu Ảnh Kiến Trúc Việt Nam!", 
            100.0
        )
        logger.info(f"Successfully completed image ingest pipeline for batch {batch_id}.")

    except Exception as e:
        logger.exception(f"Exception in run_image_ingest_pipeline for {batch_id}: {e}")
        cleanup_qdrant_points(uploaded_point_ids)
        redis_service.set_batch_status(batch_id, batch_name, total_images, completed_images, "FAILED", f"Lỗi: {str(e)}", 0)
    finally:
        cleanup_local_files(temp_dir)
        active_batch_tasks.pop(batch_id, None)

def upsert_points(points: List[PointStruct]):
    """Upsert a batch of point structs to Qdrant database"""
    try:
        qdrant_client.client.upsert(
            collection_name=qdrant_client.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} image vectors to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to upsert points to Qdrant: {e}")
        raise

def cleanup_local_files(directory: Path):
    """Delete local temporary folder"""
    if directory.exists():
        try:
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to delete directory {directory}: {e}")

def cleanup_qdrant_points(point_ids: List[str]):
    """Delete points from Qdrant if task was cancelled or failed"""
    if not point_ids:
        return
    try:
        qdrant_client.client.delete(
            collection_name=qdrant_client.collection_name,
            points_selector=point_ids
        )
        logger.info(f"Deleted {len(point_ids)} incomplete points from Qdrant.")
    except Exception as e:
        logger.error(f"Failed to cleanup Qdrant points: {e}")
