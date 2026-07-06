#!/usr/bin/env python3
"""
Qdrant Video Keyframes Loader
============================

Pipeline to load video keyframe data into Qdrant vector database.
Supports CLIP features, metadata, and object detection results.

"""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfig, OptimizersConfig, PointStruct, VectorParams
from tqdm import tqdm
import sys

# Add project root folder to sys.path to allow importing pipeline config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from pipeline import config as global_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


# Configuration
@dataclass
class Config:
    """Configuration for Qdrant loader"""
    # Data paths (adjust these to your actual data location)
    data_root: str = global_config.DATA_ROOT if HAS_CONFIG else "../data"
    clip_features_dir: str = global_config.CLIP_FEATURES_DIR if HAS_CONFIG else "clip-features-32-aic25-b1/clip-features-32"
    map_keyframes_dir: str = global_config.MAP_KEYFRAMES_DIR if HAS_CONFIG else "map-keyframes-aic25-b1/map-keyframes"
    objects_dir: str = global_config.OBJECTS_DIR if HAS_CONFIG else "objects-aic25-b1/objects"
    keyframes_dir_pattern: str = global_config.KEYFRAMES_DIR_PATTERN if HAS_CONFIG else "Keyframes_L{batch}/keyframes"  # L21, L22, etc.

    # Qdrant settings
    qdrant_host: str = global_config.QDRANT_HOST if HAS_CONFIG else "localhost"
    qdrant_port: int = global_config.QDRANT_PORT if HAS_CONFIG else 6333
    collection_name: str = global_config.COLLECTION_NAME if HAS_CONFIG else "video_keyframes"

    # Processing settings
    batch_size: int = global_config.BATCH_SIZE if HAS_CONFIG else 1000  # Points per batch upload
    confidence_threshold: float = global_config.CONFIDENCE_THRESHOLD if HAS_CONFIG else 0.5  # Object detection threshold
    high_confidence_threshold: float = global_config.HIGH_CONFIDENCE_THRESHOLD if HAS_CONFIG else 0.7
    max_workers: int = global_config.MAX_WORKERS if HAS_CONFIG else 4  # Concurrent processing

    # Validation settings
    validate_files: bool = global_config.VALIDATE_FILES if HAS_CONFIG else True
    verbose_logging: bool = global_config.VERBOSE_LOGGING if HAS_CONFIG else True


# Setup logging
def setup_logging(verbose: bool = True) -> logging.Logger:
    """Setup logging with appropriate level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('qdrant_loader.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DataValidator:
    """Validates data consistency and structure"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.validation_report = {
            'total_videos': 0,
            'valid_videos': 0,
            'missing_files': [],
            'shape_mismatches': [],
            'total_keyframes': 0
        }

    def validate_dataset(self) -> Dict[str, Any]:
        """Validate entire dataset and return report"""
        self.logger.info(" Starting dataset validation...")

        clip_features_path = Path(self.config.data_root) / self.config.clip_features_dir
        if not clip_features_path.exists():
            raise FileNotFoundError(f"CLIP features directory not found: {clip_features_path}")

        npy_files = list(clip_features_path.glob("*.npy"))
        self.validation_report['total_videos'] = len(npy_files)

        self.logger.info(f" Found {len(npy_files)} .npy files")

        for npy_file in tqdm(npy_files, desc="Validating files"):
            video_id = npy_file.stem
            if self._validate_video_files(video_id):
                self.validation_report['valid_videos'] += 1

        self._print_validation_report()
        return self.validation_report

    def _validate_video_files(self, video_id: str) -> bool:
        """Validate files for a single video"""
        try:
            # Check CLIP features
            clip_path = Path(self.config.data_root) / self.config.clip_features_dir / f"{video_id}.npy"
            features = np.load(clip_path, mmap_mode='r')
            num_keyframes = features.shape[0]

            if features.shape[1] != 512:
                self.validation_report['shape_mismatches'].append(
                    f"{video_id}: Expected (N, 512), got {features.shape}"
                )
                return False

            # Check CSV metadata
            csv_path = Path(self.config.data_root) / self.config.map_keyframes_dir / f"{video_id}.csv"
            if not csv_path.exists():
                self.validation_report['missing_files'].append(f"CSV: {csv_path}")
                return False

            csv_data = pd.read_csv(csv_path)
            if len(csv_data) != num_keyframes:
                self.validation_report['shape_mismatches'].append(
                    f"{video_id}: CSV rows ({len(csv_data)}) != features rows ({num_keyframes})"
                )
                return False

            # Check objects directory
            objects_path = Path(self.config.data_root) / self.config.objects_dir / video_id
            if not objects_path.exists():
                self.validation_report['missing_files'].append(f"Objects dir: {objects_path}")
                return False

            # Check keyframes directory
            batch = video_id.split('_')[0]  # Extract L21, L22, etc.
            keyframes_dir = self.config.keyframes_dir_pattern.format(batch=batch[1:])  # Remove 'L'
            keyframes_path = Path(self.config.data_root) / keyframes_dir / video_id

            if not keyframes_path.exists():
                self.validation_report['missing_files'].append(f"Keyframes dir: {keyframes_path}")
                return False

            self.validation_report['total_keyframes'] += num_keyframes
            return True

        except Exception as e:
            self.logger.error(f"[ERROR] Error validating {video_id}: {e}")
            return False

    def _print_validation_report(self):
        """Print validation summary"""
        report = self.validation_report
        self.logger.info("=" * 60)
        self.logger.info(" VALIDATION REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Total videos found: {report['total_videos']}")
        self.logger.info(f"Valid videos: {report['valid_videos']}")
        self.logger.info(f"Invalid videos: {report['total_videos'] - report['valid_videos']}")
        self.logger.info(f"Total keyframes: {report['total_keyframes']:,}")

        if report['missing_files']:
            self.logger.warning(f"Missing files: {len(report['missing_files'])}")
            for missing in report['missing_files'][:5]:  # Show first 5
                self.logger.warning(f"  - {missing}")
            if len(report['missing_files']) > 5:
                self.logger.warning(f"  ... and {len(report['missing_files']) - 5} more")

        if report['shape_mismatches']:
            self.logger.error(f"Shape mismatches: {len(report['shape_mismatches'])}")
            for mismatch in report['shape_mismatches']:
                self.logger.error(f"  - {mismatch}")


class VideoProcessor:
    """Processes individual video data"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def process_video(self, video_id: str) -> List[PointStruct]:
        """Process single video and return Qdrant points"""
        try:
            # Load CLIP features
            clip_path = Path(self.config.data_root) / self.config.clip_features_dir / f"{video_id}.npy"
            features = np.load(clip_path, mmap_mode='r')

            # Load metadata CSV
            csv_path = Path(self.config.data_root) / self.config.map_keyframes_dir / f"{video_id}.csv"
            metadata_df = pd.read_csv(csv_path)

            # Try to dynamically locate keyframes directory on disk
            data_root_path = Path(self.config.data_root)
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
                keyframes_dir = self.config.keyframes_dir_pattern.format(batch=batch_num)

            # Process each keyframe
            points = []
            for idx, row in metadata_df.iterrows():
                try:
                    point = self._create_point(video_id, idx, row, features[idx], keyframes_dir)
                    if point:
                        points.append(point)
                except Exception as e:
                    self.logger.error(f"[ERROR] Error processing {video_id} keyframe {idx}: {e}")
                    continue

            self.logger.info(f"[SUCCESS] Processed {video_id}: {len(points)} points created")
            return points

        except Exception as e:
            self.logger.error(f"[ERROR] Error processing video {video_id}: {e}")
            return []

    def _create_point(self, video_id: str, keyframe_idx: int, metadata_row: pd.Series,
                      vector: np.ndarray, keyframes_dir: str) -> Optional[PointStruct]:
        """Create a single Qdrant point"""
        try:
            # Generate UUID point ID (Qdrant requirement)
            point_id = str(uuid.uuid4())
            original_id = f"{video_id}_{keyframe_idx+1:03d}"

            # Load object detection results
            objects_data = self._load_objects(video_id, keyframe_idx + 1)

            # Extract batch info
            batch = video_id.split('_')[0]  # L21, L22, etc.

            # Create keyframes path
            jpg_path = f"{keyframes_dir}/{video_id}/{keyframe_idx+1:03d}.jpg"

            # Build payload
            payload = {
                "original_id": original_id,  # Keep for reference
                "video_id": video_id,
                "keyframe_idx": keyframe_idx + 1,
                "keyframe_name": f"{keyframe_idx+1:03d}.jpg",
                "jpg_path": jpg_path,
                "pts_time": float(metadata_row['pts_time']),
                "frame_idx": int(metadata_row['frame_idx']),
                "fps": int(metadata_row['fps']),
                "batch": batch,
                **objects_data
            }

            # Convert vector to list (Qdrant requirement)
            vector_list = vector.astype(np.float32).tolist()

            return PointStruct(
                id=point_id,
                vector=vector_list,
                payload=payload
            )

        except Exception as e:
            self.logger.error(f"[ERROR] Error creating point for {video_id}_{keyframe_idx}: {e}")
            return None

    def _load_objects(self, video_id: str, keyframe_num: int) -> Dict[str, Any]:
        """Load and process object detection data"""
        try:
            # Try both naming conventions: 0037.json and 037.json
            objects_path_4digit = (Path(self.config.data_root) / self.config.objects_dir /
                                   video_id / f"{keyframe_num:04d}.json")
            objects_path_3digit = (Path(self.config.data_root) / self.config.objects_dir /
                                   video_id / f"{keyframe_num:03d}.json")

            objects_path = None
            if objects_path_4digit.exists():
                objects_path = objects_path_4digit
            elif objects_path_3digit.exists():
                objects_path = objects_path_3digit
            else:
                return {
                    "objects": [],
                    "object_labels": [],
                    "high_confidence_objects": [],
                    "object_count": 0,
                    "has_objects": False
                }

            with open(objects_path, 'r') as f:
                data = json.load(f)

            # Process objects
            objects = []
            object_labels = []
            high_confidence_objects = []

            scores = [float(score) for score in data.get('detection_scores', [])]
            entities = data.get('detection_class_entities', [])
            class_names = data.get('detection_class_names', [])
            boxes = data.get('detection_boxes', [])

            for i, (score, entity, class_name, box) in enumerate(zip(scores, entities, class_names, boxes)):
                if score >= self.config.confidence_threshold:
                    obj = {
                        "entity": entity,
                        "class_name": class_name,
                        "score": score,
                        "bbox": [float(x) for x in box[:4]] if len(box) >= 4 else []
                    }
                    objects.append(obj)
                    object_labels.append(entity)

                    if score >= self.config.high_confidence_threshold:
                        high_confidence_objects.append(entity)

            # Remove duplicates while preserving order
            object_labels = list(dict.fromkeys(object_labels))
            high_confidence_objects = list(dict.fromkeys(high_confidence_objects))

            return {
                "objects": objects,
                "object_labels": object_labels,
                "high_confidence_objects": high_confidence_objects,
                "object_count": len(objects),
                "has_objects": len(objects) > 0
            }

        except Exception as e:
            self.logger.error(f"[ERROR] Error loading objects for {video_id}_{keyframe_num}: {e}")
            return {
                "objects": [],
                "object_labels": [],
                "high_confidence_objects": [],
                "object_count": 0,
                "has_objects": False
            }


class QdrantManager:
    """Manages Qdrant client and operations"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None

    def connect(self) -> bool:
        """Connect to Qdrant instance"""
        try:
            self.client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port
            )
            # Test connection
            self.logger.info(f"[SUCCESS] Connected to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
            return True
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to connect to Qdrant: {e}")
            return False

    def setup_collection(self) -> bool:
        """Create and configure collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.config.collection_name in collection_names:
                self.logger.info(f" Collection '{self.config.collection_name}' already exists")
                return True

            # Create collection
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE
                ),
                optimizers_config=OptimizersConfig(
                    default_segment_number=16,
                    memmap_threshold=20000
                ),
                hnsw_config=HnswConfig(
                    m=16,
                    ef_construct=200,
                    ef=128
                )
            )

            self.logger.info(f"[SUCCESS] Created collection '{self.config.collection_name}'")
            return True

        except Exception as e:
            self.logger.error(f"[ERROR] Error setting up collection: {e}")
            return False

    def upload_points(self, points: List[PointStruct]) -> bool:
        """Upload points to Qdrant"""
        try:
            if not points:
                return True

            # Upload in batches
            for i in range(0, len(points), self.config.batch_size):
                batch = points[i:i + self.config.batch_size]

                operation_info = self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=batch
                )

                if operation_info.status.name != "COMPLETED":
                    self.logger.error(f"[ERROR] Upload failed: {operation_info.status}")
                    return False

            self.logger.info(f"[SUCCESS] Uploaded {len(points)} points successfully")
            return True

        except Exception as e:
            self.logger.error(f"[ERROR] Error uploading points: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status.name,
                "optimizer_status": info.optimizer_status.name if info.optimizer_status else "Unknown"
            }
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting collection info: {e}")
            return {}


class QdrantLoader:
    """Main loader class orchestrating the entire process"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.verbose_logging)
        self.validator = DataValidator(config, self.logger)
        self.processor = VideoProcessor(config, self.logger)
        self.qdrant = QdrantManager(config, self.logger)

    def run(self):
        """Run the complete loading pipeline"""
        start_time = time.time()

        self.logger.info(" Starting Qdrant Video Keyframes Loader")
        self.logger.info("=" * 60)

        # Step 1: Validate dataset
        if self.config.validate_files:
            validation_report = self.validator.validate_dataset()
            if validation_report['valid_videos'] == 0:
                self.logger.error("[ERROR] No valid videos found. Exiting.")
                return False

        # Step 2: Connect to Qdrant
        if not self.qdrant.connect():
            self.logger.error("[ERROR] Failed to connect to Qdrant. Exiting.")
            return False

        # Step 3: Setup collection
        if not self.qdrant.setup_collection():
            self.logger.error("[ERROR] Failed to setup collection. Exiting.")
            return False

        # Step 4: Process and upload videos
        success_count = 0
        error_count = 0
        total_points = 0

        clip_features_path = (Path(self.config.data_root) / self.config.clip_features_dir)
        npy_files = list(clip_features_path.glob("*.npy"))

        self.logger.info(f" Processing {len(npy_files)} videos...")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_video = {
                executor.submit(self.processor.process_video, npy_file.stem): npy_file.stem
                for npy_file in npy_files
            }

            # Process results as they complete
            for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="Processing videos"):
                video_id = future_to_video[future]
                try:
                    points = future.result()
                    if points:
                        if self.qdrant.upload_points(points):
                            success_count += 1
                            total_points += len(points)
                        else:
                            error_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    self.logger.error(f"[ERROR] Exception processing {video_id}: {e}")
                    error_count += 1

        # Step 5: Final report
        end_time = time.time()
        duration = end_time - start_time

        self.logger.info("=" * 60)
        self.logger.info(" LOADING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Successfully processed: {success_count} videos")
        self.logger.info(f"Failed: {error_count} videos")
        self.logger.info(f"Total points uploaded: {total_points:,}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Average: {duration/len(npy_files):.2f} seconds/video")

        # Collection stats
        collection_info = self.qdrant.get_collection_info()
        if collection_info:
            self.logger.info(f"Collection points count: {collection_info.get('points_count', 'Unknown'):,}")
            self.logger.info(f"Collection status: {collection_info.get('status', 'Unknown')}")

        return success_count > 0


def main():
    """Main entry point"""
    # Initialize config using default properties from global_config
    config = Config()

    loader = QdrantLoader(config)
    success = loader.run()

    if success:
        print("\n[SUCCESS] Loading completed successfully!")
        print(" You can now search the collection using Qdrant API")
    else:
        print("\n[ERROR] Loading failed. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
