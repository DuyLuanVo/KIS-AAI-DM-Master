#!/usr/bin/env python3
"""
Quick Qdrant Loader - Video Keyframes
====================================

Simple script to load video keyframe data into Qdrant.
Update the DATA_ROOT variable below and run.
"""

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try importing Qdrant (install with: pip install qdrant-client)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    print("[ERROR] Qdrant client not installed. Run: pip install qdrant-client")
    QDRANT_AVAILABLE = False

import sys

# Add project root folder to sys.path to allow importing pipeline config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from pipeline import config as global_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# ===== CONFIGURATION =====
DATA_ROOT = global_config.DATA_ROOT if HAS_CONFIG else "../data"
CLIP_DIR = global_config.CLIP_FEATURES_DIR if HAS_CONFIG else "clip-features-32-aic25-b1/clip-features-32"
CSV_DIR = global_config.MAP_KEYFRAMES_DIR if HAS_CONFIG else "map-keyframes-aic25-b1/map-keyframes"
OBJ_DIR = global_config.OBJECTS_DIR if HAS_CONFIG else "objects-aic25-b1/objects"
KEYFRAMES_PATTERN = global_config.KEYFRAMES_DIR_PATTERN if HAS_CONFIG else "Keyframes_L{batch}/keyframes"

# Qdrant config
QDRANT_HOST = global_config.QDRANT_HOST if HAS_CONFIG else "localhost"
QDRANT_PORT = global_config.QDRANT_PORT if HAS_CONFIG else 6333
COLLECTION = global_config.COLLECTION_NAME if HAS_CONFIG else "video_keyframes"

# Processing config
BATCH_SIZE = global_config.BATCH_SIZE if HAS_CONFIG else 500
CONFIDENCE_THRESHOLD = global_config.CONFIDENCE_THRESHOLD if HAS_CONFIG else 0.5



def check_setup():
    """Check if everything is ready"""
    print("Checking setup...")

    if not QDRANT_AVAILABLE:
        return False

    if DATA_ROOT == "/path/to/your/data":
        print("[ERROR] Please update DATA_ROOT in the script")
        return False

    if not os.path.exists(DATA_ROOT):
        print(f"[ERROR] DATA_ROOT not found: {DATA_ROOT}")
        return False

    # Check required directories
    required = [CLIP_DIR, CSV_DIR, OBJ_DIR]
    for dirname in required:
        full_path = os.path.join(DATA_ROOT, dirname)
        if not os.path.exists(full_path):
            print(f"[ERROR] Missing directory: {full_path}")
            return False

    print("[SUCCESS] Setup looks good!")
    return True


def scan_videos():
    """Find all video files"""
    clip_path = Path(DATA_ROOT) / CLIP_DIR
    npy_files = list(clip_path.glob("*.npy"))

    print(f"Found {len(npy_files)} video files")

    # Quick validation on first few files
    valid_count = 0
    for npy_file in npy_files[:3]:
        try:
            features = np.load(npy_file, mmap_mode='r')
            if features.shape[1] == 512:
                valid_count += 1
                print(f"[SUCCESS] {npy_file.stem}: {features.shape}")
            else:
                print(f"[ERROR] {npy_file.stem}: Wrong shape {features.shape}")
        except Exception as e:
            print(f"[ERROR] Error with {npy_file.stem}: {e}")

    print(f"Sample check: {valid_count}/3 files valid")
    return npy_files


def load_objects_simple(video_id, keyframe_num):
    """Load object detection data"""
    try:
        # Try both naming conventions: 0037.json and 037.json
        obj_path_4digit = (Path(DATA_ROOT) / OBJ_DIR /
                           video_id / f"{keyframe_num:04d}.json")
        obj_path_3digit = (Path(DATA_ROOT) / OBJ_DIR /
                           video_id / f"{keyframe_num:03d}.json")

        obj_path = None
        if obj_path_4digit.exists():
            obj_path = obj_path_4digit
        elif obj_path_3digit.exists():
            obj_path = obj_path_3digit
        else:
            return []

        with open(obj_path, 'r') as f:
            data = json.load(f)

        # Extract objects with sufficient confidence
        objects = []
        scores = [float(s) for s in data.get('detection_scores', [])]
        entities = data.get('detection_class_entities', [])

        for score, entity in zip(scores, entities):
            if score >= CONFIDENCE_THRESHOLD:
                objects.append({"entity": entity, "score": score})

        return objects
    except BaseException:
        return []


def process_single_video(video_id):
    """Process one video file"""
    try:
        # Load CLIP features
        clip_path = Path(DATA_ROOT) / CLIP_DIR / f"{video_id}.npy"
        features = np.load(clip_path, mmap_mode='r')

        # Load CSV metadata
        csv_path = Path(DATA_ROOT) / CSV_DIR / f"{video_id}.csv"
        metadata = pd.read_csv(csv_path)

        points = []
        
        # Try to dynamically locate keyframes directory on disk
        data_root_path = Path(DATA_ROOT)
        found_keyframes_dir = None
        for kf_dir in data_root_path.glob("Keyframes_*/keyframes"):
            if (kf_dir / video_id).exists():
                found_keyframes_dir = kf_dir.relative_to(data_root_path)
                break
        
        batch_code = video_id.split('_')[0]
        if found_keyframes_dir is not None:
            keyframes_dir = str(found_keyframes_dir).replace('\\', '/')
        else:
            batch_num = batch_code[1:] if batch_code.startswith('L') else "21"
            keyframes_dir = KEYFRAMES_PATTERN.format(batch=batch_num)

        for idx, row in metadata.iterrows():
            # Generate UUID point ID (Qdrant requirement)
            point_id = str(uuid.uuid4())
            # Load objects for this keyframe
            objects = load_objects_simple(video_id, idx + 1)
            object_names = [obj["entity"] for obj in objects]

            # Build keyframe path (relative MinIO Object Key)
            img_path = f"keyframes/{video_id}/{idx+1:03d}.jpg"

            # Create payload (include original ID for reference)
            original_id = f"{video_id}_{idx+1:03d}"
            payload = {
                "original_id": original_id,  # Keep for reference
                "video_id": video_id,
                "keyframe_idx": idx + 1,
                "keyframe_name": f"{idx+1:03d}.jpg",
                "jpg_path": img_path,
                "pts_time": float(row['pts_time']),
                "frame_idx": int(row['frame_idx']),
                "fps": int(row['fps']),
                "batch": batch_code,
                "objects": objects,
                "object_labels": object_names,
                "object_count": len(objects),
                "has_objects": len(objects) > 0
            }

            # Create Qdrant point
            point = PointStruct(
                id=point_id,
                vector=features[idx].astype(np.float32).tolist(),
                payload=payload
            )
            points.append(point)

        return points

    except Exception as e:
        print(f"[ERROR] Error processing {video_id}: {e}")
        return []


def setup_qdrant_collection():
    """Setup Qdrant client and collection"""
    try:
        # Connect
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"[OK] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

        # Check/create collection
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if COLLECTION not in collection_names:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            print(f"[OK] Created collection: {COLLECTION}")

            # Create payload indexes for fast filtering
            print("Creating payload indexes...")
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="video_id",
                field_schema="keyword"
            )
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="batch",
                field_schema="keyword"
            )
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="object_labels",
                field_schema="keyword"
            )
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="has_objects",
                field_schema="bool"
            )
            print("[OK] Payload indexes created")
        else:
            print(f"Using existing collection: {COLLECTION}")

        return client
    except Exception as e:
        print(f"[ERROR] Qdrant setup failed: {e}")
        return None


def upload_batch(client, points):
    """Upload points to Qdrant"""
    try:
        result = client.upsert(collection_name=COLLECTION, points=points)
        return result.status.name == "COMPLETED"
    except Exception as e:
        print(f"[ERROR] Upload error: {e}")
        return False


def main():
    """Main execution"""
    print("Qdrant Video Keyframes Loader")
    print("=" * 40)

    # Step 1: Check setup
    if not check_setup():
        print("[ERROR] Setup check failed")
        return 1

    # Step 2: Scan videos
    video_files = scan_videos()
    if not video_files:
        print("[ERROR] No video files found")
        return 1

    # Step 3: Setup Qdrant
    client = setup_qdrant_collection()
    if not client:
        print("[ERROR] Qdrant setup failed")
        return 1

    # Step 4: Process videos
    print(f"\nProcessing {len(video_files)} videos...")
    print("(Processing first 3 as demo - edit script to process all)")

    success_count = 0
    total_points = 0
    start_time = time.time()

    # Process subset for demo (change [:3] to process all)
    for npy_file in tqdm(video_files, desc="Processing"):
        video_id = npy_file.stem

        points = process_single_video(video_id)
        if points:
            # Upload in smaller batches
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i:i + BATCH_SIZE]
                if upload_batch(client, batch):
                    total_points += len(batch)
                else:
                    print(f"[ERROR] Failed to upload batch for {video_id}")
                    break
            else:
                success_count += 1
                print(f"[OK] {video_id}: {len(points)} points uploaded")

        time.sleep(0.1)  # Small delay

    # Results
    duration = time.time() - start_time
    print("\n" + "=" * 40)
    print("PROCESSING COMPLETE")
    print(f"Videos processed: {success_count}")
    print(f"Total points: {total_points:,}")
    print(f"Duration: {duration:.1f} seconds")

    # Collection info
    try:
        info = client.get_collection(COLLECTION)
        print(f"Collection size: {info.points_count:,} points")
    except BaseException:
        print("Could not get collection info")

    print("\nReady for search queries!")
    return 0


if __name__ == "__main__":
    if DATA_ROOT == "/path/to/your/data":
        print("[WARN] Please update DATA_ROOT in the script first!")
        print(f"Current: {DATA_ROOT}")
        exit(1)

    exit(main())
