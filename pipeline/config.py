"""
Configuration file for Video Retrieval System
============================================

Update the paths below to match your data directory structure.
"""

import os
from pathlib import Path

# ===== DATA PATHS CONFIGURATION =====
# IMPORTANT: Update these paths to match your actual data location

# Root directory containing all data folders
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")

# Example structure should be:
# /path/to/your/data/
# ├── clip-features-32-aic25-b1/clip-features-32/
# ├── map-keyframes-aic25-b1/map-keyframes/
# ├── objects-aic25-b1/objects/
# └── Keyframes_L21/, Keyframes_L22/, etc.

# Subdirectory paths (relative to DATA_ROOT)
CLIP_FEATURES_DIR = "clip-features-32-aic25-b1/clip-features-32"
MAP_KEYFRAMES_DIR = "map-keyframes-aic25-b1/map-keyframes"
OBJECTS_DIR = "objects-aic25-b1/objects"
# L21 -> Keyframes_L21/keyframes
KEYFRAMES_DIR_PATTERN = "Keyframes_L{batch}/keyframes"

# ===== QDRANT CONFIGURATION =====
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_keyframes"

# ===== PROCESSING SETTINGS =====
BATCH_SIZE = 1000  # Points per batch upload to Qdrant
CONFIDENCE_THRESHOLD = 0.5  # Minimum object detection confidence
HIGH_CONFIDENCE_THRESHOLD = 0.7  # High confidence object threshold
MAX_WORKERS = 4  # Concurrent processing threads

# ===== VALIDATION SETTINGS =====
VALIDATE_FILES = True  # Validate file structure before processing
VERBOSE_LOGGING = True  # Enable detailed logging

# ===== AUTO-DETECT DATA ROOT =====


def auto_detect_data_root():
    """
    Try to auto-detect data root based on common patterns.
    You can modify this function to match your setup.
    """
    possible_paths = [
        "/data",
        "/mnt/data",
        "/home/data",
        "./data",
        "../data"
    ]

    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            # Check if it contains expected subdirectories
            clip_dir = path_obj / CLIP_FEATURES_DIR
            if clip_dir.exists() and any(clip_dir.glob("*.npy")):
                return str(path_obj)

    return None


# Try to auto-detect if DATA_ROOT is not set properly
if DATA_ROOT == "/path/to/your/data" or not os.path.exists(DATA_ROOT):
    detected_root = auto_detect_data_root()
    if detected_root:
        DATA_ROOT = detected_root
        print(f"[OK] Auto-detected data root: {DATA_ROOT}")
    else:
        print(f"[WARN] Could not auto-detect data root. Current: {DATA_ROOT}")

# ===== VALIDATION FUNCTIONS =====


def validate_config():
    """Validate configuration and paths"""
    issues = []

    # Check data root exists
    if not os.path.exists(DATA_ROOT):
        issues.append(f"DATA_ROOT path does not exist: {DATA_ROOT}")

    # Check required subdirectories
    required_dirs = [CLIP_FEATURES_DIR, MAP_KEYFRAMES_DIR, OBJECTS_DIR]
    for dir_name in required_dirs:
        full_path = os.path.join(DATA_ROOT, dir_name)
        if not os.path.exists(full_path):
            issues.append(f"Required directory missing: {full_path}")

    return issues


def print_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Data Root: {DATA_ROOT}")
    print(f"CLIP Features: {os.path.join(DATA_ROOT, CLIP_FEATURES_DIR)}")
    print(f"Map Keyframes: {os.path.join(DATA_ROOT, MAP_KEYFRAMES_DIR)}")
    print(f"Objects: {os.path.join(DATA_ROOT, OBJECTS_DIR)}")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Workers: {MAX_WORKERS}")
    print("=" * 60)

    # Validate and show issues if any
    issues = validate_config()
    if issues:
        print("[WARN] CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        print("=" * 60)
        return False
    else:
        print("[OK] Configuration looks good!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    print_config_summary()
