#!/usr/bin/env python3
"""
Video Preprocessing Pipeline - KIS Challenge
============================================

Extracts keyframes, encodes them with OpenAI CLIP (ViT-B/32),
and prepares metadata + placeholder object JSONs for the Qdrant loader.

Usage:
  python preprocess_video.py --video_path /path/to/video.mp4 --video_id L21_V099 --batch L21
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Add project root folder to sys.path to allow importing pipeline config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from pipeline import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Try importing cv2
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Try importing torch & clip
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def check_dependencies():
    """Verify that opencv, torch and clip are installed"""
    missing = []
    if not OPENCV_AVAILABLE:
        missing.append("opencv-python (pip install opencv-python)")
    if not CLIP_AVAILABLE:
        missing.append("torch & torchvision & openai-clip (pip install torch torchvision openai-clip)")
        
    if missing:
        print("[ERROR] Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)


def extract_keyframes(video_path: Path, output_dir: Path, csv_output_path: Path, sample_rate_sec: float = 1.0):
    """
    Step 1: Extract frames from video at a regular interval and save as JPG.
    Also generate metadata CSV matching the schema.
    """
    print(f"\n Step 1: Extracting keyframes from {video_path.name}...")
    print(f"   Rate: 1 frame every {sample_rate_sec} second(s)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Error: Cannot open video file {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   Video Info: FPS={fps:.2f}, Total Frames={total_frames}, Duration={duration:.1f}s")
    
    metadata = []
    frame_count = 0
    keyframe_idx = 1
    
    # Calculate step in frames based on requested sample rate
    frame_step = int(round(fps * sample_rate_sec))
    if frame_step <= 0:
        frame_step = 1
        
    # ProgressBar setup
    pbar = tqdm(total=total_frames, desc="Cắt frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_step == 0:
            pts_time = frame_count / fps
            img_name = f"{keyframe_idx:03d}.jpg"
            img_path = output_dir / img_name
            
            # Save image
            cv2.imwrite(str(img_path), frame)
            
            # Record metadata
            metadata.append({
                "n": keyframe_idx,
                "pts_time": pts_time,
                "fps": int(round(fps)),
                "frame_idx": frame_count
            })
            keyframe_idx += 1
            
        frame_count += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    # Save CSV
    df = pd.DataFrame(metadata)
    df.to_csv(csv_output_path, index=False)
    print(f"[SUCCESS] Extracted {len(metadata)} keyframes.")
    print(f"[SUCCESS] Metadata saved to {csv_output_path}")
    return len(metadata)


def extract_clip_features(keyframes_dir: Path, npy_output_path: Path, model_name: str = "ViT-B/32"):
    """
    Step 2: Generate CLIP 512-dim vectors for each keyframe and save to a single .npy file.
    """
    print(f"\n Step 2: Generating CLIP features ({model_name}) from keyframes...")
    
    npy_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of jpg keyframes
    img_paths = sorted(list(keyframes_dir.glob("*.jpg")))
    if not img_paths:
        print(f"[ERROR] Error: No keyframe images found in {keyframes_dir}")
        sys.exit(1)
        
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Loading CLIP model on {device.upper()}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    features_list = []
    
    for img_path in tqdm(img_paths, desc="Tạo vector embedding"):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            features_list.append(image_features.cpu().numpy().flatten())
        except Exception as e:
            print(f"[ERROR] Error processing {img_path.name}: {e}")
            sys.exit(1)
            
    # Save as .npy
    features_array = np.array(features_list, dtype=np.float16)  # Use float16 to match project specs
    np.save(npy_output_path, features_array)
    print(f"[SUCCESS] Created features array with shape: {features_array.shape}")
    print(f"[SUCCESS] CLIP features saved to {npy_output_path}")


def generate_placeholders(video_id: str, count: int, objects_dir: Path):
    """
    Step 3: Generate placeholder object detection JSON files for each keyframe.
    This prevents the Qdrant loader from failing due to missing object detection files.
    """
    print(f"\n Step 3: Generating {count} placeholder object detection JSON files...")
    
    video_objects_dir = objects_dir / video_id
    video_objects_dir.mkdir(parents=True, exist_ok=True)
    
    placeholder_data = {
        "detection_scores": [],
        "detection_class_names": [],
        "detection_class_entities": [],
        "detection_boxes": [],
        "detection_class_labels": []
    }
    
    for i in range(1, count + 1):
        # The project supports both 3-digit and 4-digit formatting (e.g. 001.json or 0001.json)
        # We will write 4-digit formatting as default (e.g., 0001.json)
        json_path = video_objects_dir / f"{i:04d}.json"
        with open(json_path, 'w') as f:
            json.dump(placeholder_data, f)
            
    print(f"[SUCCESS] Placeholders saved under {video_objects_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Video Preprocessing Pipeline - KIS Challenge")
    parser.add_argument("--video_path", type=str, required=True, help="Path to raw video file (.mp4, etc.)")
    parser.add_argument("--video_id", type=str, default=None, help="Custom ID for this video (e.g., L21_V001)")
    parser.add_argument("--batch", type=str, default="L21", help="Batch folder (e.g. L21, L22...)")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Seconds per keyframe (default: 1.0)")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for the dataset")
    
    args = parser.parse_args()
    
    check_dependencies()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[ERROR] Error: Video file not found: {video_path}")
        sys.exit(1)
        
    video_id = args.video_id or video_path.stem
    batch = args.batch
    # Standardize batch naming format to Lxx (e.g., L21)
    if not batch.startswith("L"):
        batch = f"L{batch}"
        
    # Resolve DATA_ROOT
    data_root_str = args.data_root
    if not data_root_str and HAS_CONFIG:
        # Fallback to config if it was auto-detected or configured
        if config.DATA_ROOT and config.DATA_ROOT != "/path/to/your/data":
            data_root_str = config.DATA_ROOT
            print(f" Using DATA_ROOT from config.py: {data_root_str}")
            
    if not data_root_str:
        # Final fallback
        data_root_str = "../data"
        print(f" DATA_ROOT not specified or configured. Defaulting to: {data_root_str}")
        
    data_root = Path(data_root_str)
    
    # Establish folder paths according to project configuration
    clip_dir = data_root / "clip-features-32-aic25-b1/clip-features-32"
    csv_dir = data_root / "map-keyframes-aic25-b1/map-keyframes"
    objects_dir = data_root / "objects-aic25-b1/objects"
    keyframes_dir = data_root / f"Keyframes_{batch}/keyframes" / video_id
    
    # Output file paths
    npy_output_path = clip_dir / f"{video_id}.npy"
    csv_output_path = csv_dir / f"{video_id}.csv"
    
    print("\n" + "=" * 60)
    print(" STARTING VIDEO PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Target Video: {video_path}")
    print(f"Video ID:     {video_id}")
    print(f"Batch:        {batch}")
    print(f"Data Root:    {data_root}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Extract frames
    num_keyframes = extract_keyframes(video_path, keyframes_dir, csv_output_path, args.sample_rate)
    
    # 2. Extract CLIP features
    extract_clip_features(keyframes_dir, npy_output_path)
    
    # 3. Generate objects placeholders
    generate_placeholders(video_id, num_keyframes, objects_dir)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING SUCCESSFUL!")
    print(f"Processed:    {video_id}")
    print(f"Keyframes:    {num_keyframes}")
    print(f"Total Time:   {elapsed_time:.1f} seconds")
    print("=" * 60)
    print("\n What to do next:")
    print("1. Update your loader command or config.py if needed.")
    print("2. Run 'python run_loader.py' to upload this new video to Qdrant!")
    print("=" * 60)


if __name__ == "__main__":
    main()
