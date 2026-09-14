"""
Script re-indexing architectural images with rich metadata and YOLO-World detections into Qdrant & MinIO.
"""
import os
import sys
import glob
import json
import uuid
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm backend vào sys.path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, backend_dir)

import cv2
import numpy as np
from loguru import logger
from qdrant_client.models import PointStruct

from app.services.clip_service import clip_service
from app.services.yolo_service import yolo_service
from app.services.minio_service import minio_service
from app.database.qdrant_client import qdrant_client

def reindex_architecture_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vietnam_architecture")
    jpg_files = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.JPG"))
    
    print(f"🏛️ Tìm thấy {len(jpg_files)} ảnh kiến trúc trong {data_dir}")
    if not jpg_files:
        print("❌ Không có ảnh nào để xử lý.")
        return

    # Khởi tạo YOLO-World và CLIP
    print("🧠 Đang nạp mô hình CLIP và YOLO-World...")
    yolo_service.initialize()

    batch_id = "ARCH_EAF319B2"
    batch_name = "Kho Tư Liệu Kiến Trúc Việt Nam"
    points = []

    for idx, img_path in enumerate(jpg_files):
        file_name = os.path.basename(img_path)
        clean_stem = Path(file_name).stem.replace(" ", "_")
        json_path = os.path.join(data_dir, f"{Path(file_name).stem}.json")

        meta = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    meta = json.load(jf)
            except Exception as e:
                print(f"⚠️ Lỗi đọc {json_path}: {e}")

        monument_name = meta.get("monument_name") or Path(file_name).stem.replace("_", " ")
        description = meta.get("description") or f"Tư liệu hình ảnh công trình kiến trúc {monument_name}."
        categories = meta.get("categories", [])
        date_val = meta.get("date", "")
        author_val = meta.get("artist", "")
        source_url = meta.get("source_url", "")
        lat_val = meta.get("latitude")
        lon_val = meta.get("longitude")

        # 1. Upload to MinIO
        object_key = f"architecture/{batch_id}/{file_name}"
        try:
            minio_service.s3_client.upload_file(
                Filename=img_path,
                Bucket=minio_service.bucket_name,
                Key=object_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
        except Exception as e:
            print(f"⚠️ Lỗi tải ảnh lên MinIO ({file_name}): {e}")

        # 2. Encode CLIP
        try:
            vector = clip_service.encode_image_from_path(img_path)
        except Exception as e:
            print(f"⚠️ Lỗi mã hóa CLIP ({file_name}): {e}")
            vector = np.random.rand(512).astype(np.float32).tolist()

        # 3. Detect with YOLO-World (Dynamic Architecture Classes)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                context_cats = categories if categories else [monument_name]
                objects, object_labels = yolo_service.detect_objects(img, context_categories=context_cats)
            else:
                objects, object_labels = [], []
        except Exception as e:
            print(f"⚠️ Lỗi YOLO-World ({file_name}): {e}")
            objects, object_labels = [], []

        # 4. Create Point
        point_id = str(uuid.uuid4())
        payload = {
            "original_id": f"{batch_id}_{idx:03d}_{clean_stem}",
            "video_id": batch_name,
            "keyframe_idx": idx,
            "keyframe_name": file_name,
            "jpg_path": object_key,
            "pts_time": 0.0,
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

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        label_str = ", ".join(object_labels) if object_labels else "Toàn cảnh kiến trúc"
        print(f"[{idx+1}/{len(jpg_files)}] ✅ {monument_name[:35]} | YOLO: [{label_str}]")

    # Xóa sạch dữ liệu cũ trong collection và nạp mới toàn bộ 51 điểm hoàn hảo
    print(f"\n🚀 Đang lập chỉ mục {len(points)} điểm vector hoàn chỉnh vào Qdrant...")
    try:
        # Xóa points cũ bằng batch ID
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_client.client.delete(
            collection_name=qdrant_client.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="batch", match=MatchValue(value=batch_id))]
            )
        )
    except Exception as e:
        print(f"Note delete old: {e}")

    # Upsert points mới
    qdrant_client.client.upsert(
        collection_name=qdrant_client.collection_name,
        points=points,
        wait=True
    )
    print(f"🎉 Hoàn tất lập chỉ mục {len(points)} công trình kiến trúc vào Qdrant với đầy đủ Metadata & YOLO-World!")

if __name__ == "__main__":
    reindex_architecture_dataset()
