#!/usr/bin/env python3
"""
Simple Qdrant Video Keyframes Loader
===================================

Simplified version for loading video keyframe data into Qdrant.
Easy to use and modify.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

import sys

# Add project root folder to sys.path to allow importing pipeline config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from pipeline import config as global_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# ========== CONFIGURATION ==========
DATA_ROOT = global_config.DATA_ROOT if HAS_CONFIG else "../data"
CLIP_FEATURES_DIR = global_config.CLIP_FEATURES_DIR if HAS_CONFIG else "clip-features-32-aic25-b1/clip-features-32"
MAP_KEYFRAMES_DIR = global_config.MAP_KEYFRAMES_DIR if HAS_CONFIG else "map-keyframes-aic25-b1/map-keyframes"
OBJECTS_DIR = global_config.OBJECTS_DIR if HAS_CONFIG else "objects-aic25-b1/objects"
KEYFRAMES_DIR_PATTERN = global_config.KEYFRAMES_DIR_PATTERN if HAS_CONFIG else "Keyframes_L{batch}/keyframes"

# Qdrant settings
QDRANT_HOST = global_config.QDRANT_HOST if HAS_CONFIG else "localhost"
QDRANT_PORT = global_config.QDRANT_PORT if HAS_CONFIG else 6333
COLLECTION_NAME = global_config.COLLECTION_NAME if HAS_CONFIG else "video_keyframes"

# Processing settings
BATCH_SIZE = global_config.BATCH_SIZE if HAS_CONFIG else 1000
CONFIDENCE_THRESHOLD = global_config.CONFIDENCE_THRESHOLD if HAS_CONFIG else 0.5
HIGH_CONFIDENCE_THRESHOLD = global_config.HIGH_CONFIDENCE_THRESHOLD if HAS_CONFIG else 0.7



def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('qdrant_loader.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_paths():
    """Check if all required paths exist"""
    logger = logging.getLogger(__name__)

    if not os.path.exists(DATA_ROOT):
        logger.error(f"[ERROR] DATA_ROOT not found: {DATA_ROOT}")
        return False

    required_dirs = [CLIP_FEATURES_DIR, MAP_KEYFRAMES_DIR, OBJECTS_DIR]
    for dir_name in required_dirs:
        full_path = os.path.join(DATA_ROOT, dir_name)
        if not os.path.exists(full_path):
            logger.error(f"[ERROR] Required directory missing: {full_path}")
            return False

    logger.info("[SUCCESS] All required paths exist")
    return True


def scan_dataset():
    """Scan and validate dataset"""
    logger = logging.getLogger(__name__)

    clip_path = Path(DATA_ROOT) / CLIP_FEATURES_DIR
    npy_files = list(clip_path.glob("*.npy"))

    logger.info(f" Found {len(npy_files)} .npy files")

    total_vectors = 0
    valid_videos = 0

    for npy_file in tqdm(npy_files[:5], desc="Validating sample"):
        try:
            features = np.load(npy_file, mmap_mode='r')
            if features.shape[1] == 512:
                total_vectors += features.shape[0]
                valid_videos += 1
                logger.info(f"[SUCCESS] {npy_file.stem}: {features.shape}")
            else:
                logger.error(f"[ERROR] {npy_file.stem}: Wrong shape {features.shape}")
        except Exception as e:
            logger.error(f"[ERROR] Error loading {npy_file.stem}: {e}")

    logger.info(f" Sample validation: {valid_videos}/5 valid")
    logger.info(f" Estimated total vectors: ~{total_vectors * len(npy_files) // 5:,}")

    return len(npy_files)


def load_objects(video_id, keyframe_num):
    """Load object detection data for a keyframe"""
    try:
        objects_path = (Path(DATA_ROOT) / OBJECTS_DIR /
                        video_id / f"{keyframe_num:04d}.json")

        if not objects_path.exists():
            return []

        with open(objects_path, 'r') as f:
            data = json.load(f)

        objects = []
        scores = [float(s) for s in data.get('detection_scores', [])]
        entities = data.get('detection_class_entities', [])
        class_names = data.get('detection_class_names', [])
        boxes = data.get('detection_boxes', [])

        for score, entity, class_name, box in zip(scores, entities,
                                                  class_names, boxes):
            if score >= CONFIDENCE_THRESHOLD:
                objects.append({
                    "entity": entity,
                    "class_name": class_name,
                    "score": score,
                    "bbox": [float(x) for x in box[:4]] if len(box) >= 4 else []
                })

        return objects

    except Exception as e:
        logging.getLogger(__name__).error(f"[ERROR] Error loading objects: {e}")
        return []


def process_video(video_id):
    """Process single video and create Qdrant points"""
    logger = logging.getLogger(__name__)

    try:
        # Load CLIP features
        clip_path = Path(DATA_ROOT) / CLIP_FEATURES_DIR / f"{video_id}.npy"
        features = np.load(clip_path, mmap_mode='r')

        # Load metadata
        csv_path = Path(DATA_ROOT) / MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        metadata = pd.read_csv(csv_path)

        points = []
        
        # Try to dynamically locate keyframes directory on disk
        data_root_path = Path(DATA_ROOT)
        found_keyframes_dir = None
        for kf_dir in data_root_path.glob("Keyframes_*/keyframes"):
            if (kf_dir / video_id).exists():
                found_keyframes_dir = kf_dir.relative_to(data_root_path)
                break
        
        batch = video_id.split('_')[0]
        if found_keyframes_dir is not None:
            keyframes_dir = str(found_keyframes_dir).replace('\\', '/')
        else:
            batch_num = batch[1:] if batch.startswith('L') else "21"
            keyframes_dir = KEYFRAMES_DIR_PATTERN.format(batch=batch_num)

        for idx, row in metadata.iterrows():
            # Create point ID
            point_id = f"{video_id}_{idx+1:03d}"

            # Load objects
            objects = load_objects(video_id, idx + 1)
            object_labels = [obj["entity"] for obj in objects]
            high_conf_objects = [obj["entity"] for obj in objects
                                 if obj["score"] >= HIGH_CONFIDENCE_THRESHOLD]

            # Create keyframes path
            jpg_path = f"{keyframes_dir}/{video_id}/{idx+1:03d}.jpg"

            # Create payload
            payload = {
                "video_id": video_id,
                "keyframe_idx": idx + 1,
                "keyframe_name": f"{idx+1:03d}.jpg",
                "jpg_path": jpg_path,
                "pts_time": float(row['pts_time']),
                "frame_idx": int(row['frame_idx']),
                "fps": int(row['fps']),
                "batch": batch,
                "objects": objects,
                "object_labels": object_labels,
                "high_confidence_objects": high_conf_objects,
                "object_count": len(objects),
                "has_objects": len(objects) > 0
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=features[idx].astype(np.float32).tolist(),
                payload=payload
            )
            points.append(point)

        logger.info(f"[SUCCESS] Processed {video_id}: {len(points)} points")
        return points

    except Exception as e:
        logger.error(f"[ERROR] Error processing {video_id}: {e}")
        return []


def setup_qdrant():
    """Setup Qdrant client and collection"""
    logger = logging.getLogger(__name__)

    try:
        # Connect to Qdrant
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"[SUCCESS] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if COLLECTION_NAME not in collection_names:
            # Create collection
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            logger.info(f"[SUCCESS] Created collection: {COLLECTION_NAME}")
        else:
            logger.info(f" Collection already exists: {COLLECTION_NAME}")

        return client

    except Exception as e:
        logger.error(f"[ERROR] Error setting up Qdrant: {e}")
        return None


def upload_points(client, points):
    """Upload points to Qdrant in batches"""
    logger = logging.getLogger(__name__)

    if not points:
        return True

    try:
        # Upload in batches
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]

            operation_info = client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )

            if operation_info.status.name != "COMPLETED":
                logger.error(f"[ERROR] Upload failed: {operation_info.status}")
                return False

        logger.info(f"[SUCCESS] Uploaded {len(points)} points")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Error uploading points: {e}")
        return False


def main():
    """Main function"""
    logger = setup_logging()

    logger.info(" Starting Simple Qdrant Loader")
    logger.info("=" * 50)

    # Step 1: Validate paths
    if not validate_paths():
        logger.error("[ERROR] Path validation failed. Please update DATA_ROOT in script.")
        return 1

    # Step 2: Scan dataset
    total_videos = scan_dataset()
    if total_videos == 0:
        logger.error("[ERROR] No videos found")
        return 1

    # Step 3: Setup Qdrant
    client = setup_qdrant()
    if not client:
        logger.error("[ERROR] Failed to setup Qdrant")
        return 1

    # Step 4: Process videos
    logger.info(f" Processing {total_videos} videos...")

    clip_path = Path(DATA_ROOT) / CLIP_FEATURES_DIR
    npy_files = list(clip_path.glob("*.npy"))

    success_count = 0
    total_points = 0
    start_time = time.time()

    # Process first 5 videos as demo (remove [:5] to process all)
    for npy_file in tqdm(npy_files[:5], desc="Processing videos"):
        video_id = npy_file.stem

        points = process_video(video_id)
        if points:
            if upload_points(client, points):
                success_count += 1
                total_points += len(points)

        # Small delay to avoid overwhelming Qdrant
        time.sleep(0.1)

    # Final report
    duration = time.time() - start_time
    logger.info("=" * 50)
    logger.info(" PROCESSING COMPLETE")
    logger.info(f"Successfully processed: {success_count} videos")
    logger.info(f"Total points uploaded: {total_points:,}")
    logger.info(f"Duration: {duration:.2f} seconds")

    # Collection stats
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")

    return 0


if __name__ == "__main__":
    print(" Simple Qdrant Video Keyframes Loader")
    print("=" * 50)
    print("[WARN]  IMPORTANT: Update DATA_ROOT path in the script before running!")
    print(f"Current DATA_ROOT: {DATA_ROOT}")
    print("=" * 50)

    if DATA_ROOT == "/path/to/your/data":
        print("[ERROR] Please update DATA_ROOT in the script first")
        exit(1)

    exit(main())
