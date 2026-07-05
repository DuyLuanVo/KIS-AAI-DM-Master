#!/usr/bin/env python3
"""
Video Preprocessing Pipeline - KIS Challenge
============================================

Extracts keyframes, encodes them with OpenAI CLIP (ViT-B/32),
and prepares metadata + placeholder object JSONs for the Qdrant loader.
This version is heavily optimized using in-memory pipelines, frame skipping,
batch inference, and asynchronous disk I/O.

Usage:
  python preprocess_video.py --video_path /path/to/video.mp4 --video_id L21_V099 --batch L21
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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


def generate_placeholders(video_id: str, count: int, objects_dir: Path):
    """
    Step 3: Generate placeholder object detection JSON files for each keyframe.
    This prevents the Qdrant loader from failing due to missing object detection files.
    """
    video_objects_dir = objects_dir / video_id
    video_objects_dir.mkdir(parents=True, exist_ok=True)
    
    placeholder_data = {
        "detection_scores": [],
        "detection_class_names": [],
        "detection_class_entities": [],
        "detection_boxes": [],
        "detection_class_labels": []
    }
    
    # Fast bulk writing
    for i in range(1, count + 1):
        json_path = video_objects_dir / f"{i:04d}.json"
        with open(json_path, 'w') as f:
            json.dump(placeholder_data, f)


def process_video_optimized(
    video_path: Path,
    keyframes_dir: Path,
    csv_output_path: Path,
    npy_output_path: Path,
    sample_rate_sec: float = 1.0,
    batch_size: int = 32,
    model_name: str = "ViT-B/32"
):
    """
    Optimized single-pass pipeline:
    1. Reads video, skips intermediate frames with cap.grab() (extremely fast).
    2. Decodes target frames, converts BGR->RGB in memory, and puts them into batches.
    3. Saves JPG keyframes to disk asynchronously using a ThreadPoolExecutor.
    4. Performs batched CLIP inference on GPU/CPU to maximize processing throughput.
    """
    print(f"\n[INFO] Starting optimized preprocessing for: {video_path.name}")
    print(f"       Sampling Rate: 1 frame every {sample_rate_sec} second(s)")
    print(f"       Batch Size:    {batch_size}")
    
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    npy_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Video Reader
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"[INFO] Video Details: FPS={fps:.2f}, Total Frames={total_frames}, Duration={duration:.1f}s")
    
    # 2. Initialize CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading CLIP model {model_name} on {device.upper()}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    # 3. Setup Processing State
    frame_step = int(round(fps * sample_rate_sec))
    if frame_step <= 0:
        frame_step = 1
        
    metadata = []
    features_list = []
    
    # Batch buffer
    batch_images = []
    
    # Async I/O Thread Pool (for saving JPGs)
    print(f"[INFO] Initializing async I/O worker pool for keyframe saving...")
    io_executor = ThreadPoolExecutor(max_workers=4)
    futures = []
    
    # ProgressBar
    pbar = tqdm(total=total_frames, desc="Processing Video")
    
    frame_count = 0
    keyframe_idx = 1
    
    # Helper to encode a batch of images
    def encode_batch(pil_imgs):
        if not pil_imgs:
            return
        
        # Preprocess and stack tensors
        tensors = torch.stack([preprocess(img) for img in pil_imgs]).to(device)
        
        with torch.no_grad():
            # Use mixed precision if on CUDA
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    batch_features = model.encode_image(tensors)
            else:
                batch_features = model.encode_image(tensors)
                
            # Normalize embeddings
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            features_list.extend(batch_features.cpu().numpy())

    # 4. Processing Loop
    while cap.isOpened():
        if frame_count % frame_step == 0:
            # Grab and retrieve (decode)
            ret = cap.grab()
            if not ret:
                break
            ret, frame = cap.retrieve()
            if not ret:
                break
            
            pts_time = frame_count / fps
            img_name = f"{keyframe_idx:03d}.jpg"
            img_path = keyframes_dir / img_name
            
            # Submit asynchronous disk write (using a copy of the frame to avoid buffer reuse issues)
            fut = io_executor.submit(cv2.imwrite, str(img_path), frame.copy())
            futures.append(fut)
            
            # Convert BGR (OpenCV) to RGB PIL Image in-memory for CLIP
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            batch_images.append(pil_img)
            
            # Record metadata
            metadata.append({
                "n": keyframe_idx,
                "pts_time": pts_time,
                "fps": int(round(fps)),
                "frame_idx": frame_count
            })
            
            # If batch is full, execute inference
            if len(batch_images) == batch_size:
                encode_batch(batch_images)
                batch_images = []
                
            keyframe_idx += 1
        else:
            # Fast skip: Grab frame header but do not decode pixel data
            ret = cap.grab()
            if not ret:
                break
                
        frame_count += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    # Encode remaining items in the buffer
    if batch_images:
        encode_batch(batch_images)
        
    # Wait for all asynchronous JPG saving operations to finish
    if futures:
        print(f"[INFO] Waiting for {len(futures)} asynchronous keyframe disk writes to complete...")
        for fut in tqdm(futures, desc="Writing JPGs to disk"):
            fut.result()
            
    io_executor.shutdown(wait=True)
    
    # 5. Save Outputs
    # Save CSV
    df = pd.DataFrame(metadata)
    df.to_csv(csv_output_path, index=False)
    print(f"[SUCCESS] Metadata saved to: {csv_output_path}")
    
    # Save NPY (float16 to match project specifications)
    features_array = np.array(features_list, dtype=np.float16)
    np.save(npy_output_path, features_array)
    print(f"[SUCCESS] CLIP features saved to: {npy_output_path} (Shape: {features_array.shape})")
    
    return len(metadata)


def main():
    parser = argparse.ArgumentParser(description="Optimized Video Preprocessing Pipeline - KIS Challenge")
    parser.add_argument("--video_path", type=str, required=True, help="Path to raw video file (.mp4, etc.)")
    parser.add_argument("--video_id", type=str, default=None, help="Custom ID for this video (e.g., L21_V001)")
    parser.add_argument("--batch", type=str, default="L21", help="Batch folder (e.g. L21, L22...)")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Seconds per keyframe (default: 1.0)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP inference (default: 32)")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for the dataset")
    
    args = parser.parse_args()
    
    check_dependencies()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
        
    video_id = args.video_id or video_path.stem
    batch = args.batch
    if not batch.startswith("L"):
        batch = f"L{batch}"
        
    # Resolve DATA_ROOT
    data_root_str = args.data_root
    if not data_root_str and HAS_CONFIG:
        if config.DATA_ROOT and config.DATA_ROOT != "/path/to/your/data":
            data_root_str = config.DATA_ROOT
            print(f" Using DATA_ROOT from config.py: {data_root_str}")
            
    if not data_root_str:
        data_root_str = "../data"
        print(f" DATA_ROOT not specified. Defaulting to: {data_root_str}")
        
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
    print(" STARTING OPTIMIZED VIDEO PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Target Video: {video_path}")
    print(f"Video ID:     {video_id}")
    print(f"Batch:        {batch}")
    print(f"Data Root:    {data_root}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1 & 2. Process video, extract keyframes and CLIP features
    num_keyframes = process_video_optimized(
        video_path=video_path,
        keyframes_dir=keyframes_dir,
        csv_output_path=csv_output_path,
        npy_output_path=npy_output_path,
        sample_rate_sec=args.sample_rate,
        batch_size=args.batch_size
    )
    
    # 3. Generate objects placeholders
    print(f"\n Step 3: Generating {num_keyframes} placeholder object detection JSON files...")
    generate_placeholders(video_id, num_keyframes, objects_dir)
    print(f"[SUCCESS] Placeholders saved under {objects_dir / video_id}/")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING SUCCESSFUL!")
    print(f"Processed:    {video_id}")
    print(f"Keyframes:    {num_keyframes}")
    print(f"Total Time:   {elapsed_time:.1f} seconds")
    print("=" * 60)
    print("\n What to do next:")
    print("1. Run the loader script to upload to Qdrant:")
    print("   python pipeline/loaders/run_loader.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
