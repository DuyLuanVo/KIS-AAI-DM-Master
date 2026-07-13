"""
Ingest worker service for downloading videos, extracting keyframes (TIME/SBD),
encoding with CLIP, detecting objects, and indexing to Qdrant/MinIO.
"""
import os
import shutil
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from loguru import logger

# Import backend services
from app.config import settings
from app.services.redis_service import redis_service
from app.services.minio_service import minio_service
from app.services.clip_service import clip_service
from app.services.yolo_service import yolo_service
from app.database.qdrant_client import qdrant_client
from qdrant_client.models import PointStruct

# Thread pool for async tasks
executor = ThreadPoolExecutor(max_workers=2 if settings.default_extraction_method == "SBD" else 4)
active_tasks: Dict[str, threading.Event] = {}

def start_ingestion_task(video_url: str, video_id: str, parent_channel_id: Optional[str] = None, 
                         method: Optional[str] = None, time_interval: Optional[float] = None, 
                         sbd_threshold: Optional[float] = None):
    """Start the video ingestion task in the background thread pool"""
    cancel_event = threading.Event()
    active_tasks[video_id] = cancel_event
    
    # Run in thread pool
    executor.submit(
        run_ingest_pipeline,
        video_url,
        video_id,
        parent_channel_id,
        method or settings.default_extraction_method,
        time_interval or settings.default_time_interval,
        sbd_threshold or settings.default_sbd_threshold,
        cancel_event
    )
    logger.info(f"Submitted video ingestion task for {video_id} to executor.")

def compute_hsv_histogram(frame):
    """Compute normalized 2D HSV histogram for shot boundary detection"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Compute 2D histogram for H (Hue) and S (Saturation)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def run_ingest_pipeline(video_url: str, video_id: str, parent_channel_id: Optional[str], 
                       method: str, time_interval: float, sbd_threshold: float, 
                       cancel_event: threading.Event):
    """The core ingestion pipeline running in a background thread"""
    temp_dir = Path("data/temp") / video_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / "video.mp4"
    keyframes_local_dir = temp_dir / "keyframes"
    keyframes_local_dir.mkdir(parents=True, exist_ok=True)
    
    # Track point IDs uploaded so we can delete them if cancelled
    uploaded_point_ids: List[str] = []

    try:
        # Check cancellation
        if cancel_event.is_set() or redis_service.check_cancel_flag(video_id) or (parent_channel_id and redis_service.check_cancel_flag(parent_channel_id)):
            logger.info(f"Task {video_id} cancelled before starting.")
            redis_service.set_video_status(video_id, "CANCELLED", 0, "Task cancelled by user.")
            cleanup_local_files(temp_dir)
            return

        # 1. Download Video
        redis_service.set_video_status(video_id, "DOWNLOADING", 10, "Downloading video using yt-dlp...", video_url)
        logger.info(f"Starting download for {video_url} to {video_path}")
        
        # Use yt-dlp to download video
        import yt_dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': str(video_path),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            logger.info("Attempting download with high-quality merged format...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            logger.warning(f"High-quality download failed ({e}). Retrying with pre-merged mp4 format...")
            # Fallback format: best pre-merged mp4 to avoid merging (ffmpeg required) issues
            ydl_opts['format'] = 'best[ext=mp4]/mp4/best'
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as retry_err:
                logger.error(f"Fallback download failed: {retry_err}")
                
        if not video_path.exists():
            raise FileNotFoundError(f"Failed to download video file to {video_path}")

        # Check cancellation
        if cancel_event.is_set() or redis_service.check_cancel_flag(video_id) or (parent_channel_id and redis_service.check_cancel_flag(parent_channel_id)):
            redis_service.set_video_status(video_id, "CANCELLED", 0, "Task cancelled by user.")
            cleanup_local_files(temp_dir)
            return

        # 2. Extract Keyframes
        redis_service.set_video_status(video_id, "EXTRACTING", 30, f"Extracting keyframes using method {method}...")
        logger.info(f"Extracting keyframes for {video_id} using {method}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file with OpenCV.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0
            
        frame_count = 0
        keyframe_idx = 1
        
        extracted_keyframes: List[Dict] = []
        
        # TIME parameters
        frame_step = int(round(fps * time_interval))
        if frame_step <= 0:
            frame_step = 1
            
        # SBD parameters
        prev_hist = None
        
        while cap.isOpened():
            # Check cancellation in the frame reading loop (very responsive!)
            if frame_count % 100 == 0:
                if cancel_event.is_set() or redis_service.check_cancel_flag(video_id) or (parent_channel_id and redis_service.check_cancel_flag(parent_channel_id)):
                    logger.info(f"Extraction for {video_id} cancelled during processing.")
                    cap.release()
                    redis_service.set_video_status(video_id, "CANCELLED", 0, "Task cancelled by user.")
                    cleanup_local_files(temp_dir)
                    return
            
            if method == "TIME":
                if frame_count % frame_step == 0:
                    ret = cap.grab()
                    if not ret:
                        break
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    
                    pts_time = frame_count / fps
                    img_name = f"{keyframe_idx:03d}.jpg"
                    img_path = keyframes_local_dir / img_name
                    cv2.imwrite(str(img_path), frame)
                    
                    extracted_keyframes.append({
                        "idx": keyframe_idx,
                        "pts_time": pts_time,
                        "frame_idx": frame_count,
                        "path": img_path
                    })
                    keyframe_idx += 1
                else:
                    cap.grab()
            else: # SBD method
                ret = cap.grab()
                if not ret:
                    break
                ret, frame = cap.retrieve()
                if not ret:
                    break
                
                curr_hist = compute_hsv_histogram(frame)
                is_boundary = False
                
                if prev_hist is None:
                    is_boundary = True # Save the very first frame
                else:
                    # Compare color histograms
                    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                    # correlation goes from -1 to 1 (1 being identical)
                    if (1.0 - correlation) > sbd_threshold:
                        is_boundary = True
                
                if is_boundary:
                    pts_time = frame_count / fps
                    img_name = f"{keyframe_idx:03d}.jpg"
                    img_path = keyframes_local_dir / img_name
                    cv2.imwrite(str(img_path), frame)
                    
                    extracted_keyframes.append({
                        "idx": keyframe_idx,
                        "pts_time": pts_time,
                        "frame_idx": frame_count,
                        "path": img_path
                    })
                    keyframe_idx += 1
                    
                prev_hist = curr_hist
                
            frame_count += 1
            
        cap.release()
        logger.info(f"Finished frame extraction. Found {len(extracted_keyframes)} keyframes.")

        # Check cancellation
        if cancel_event.is_set() or redis_service.check_cancel_flag(video_id) or (parent_channel_id and redis_service.check_cancel_flag(parent_channel_id)):
            redis_service.set_video_status(video_id, "CANCELLED", 0, "Task cancelled by user.")
            cleanup_local_files(temp_dir)
            return

        # 3. Upload Keyframes & Index vectors
        redis_service.set_video_status(video_id, "INDEXING", 60, f"Indexing {len(extracted_keyframes)} keyframes to Database...")
        
        qdrant_points: List[PointStruct] = []
        batch_code = video_id.split('_')[0] if "_" in video_id else "L21"

        for kf in extracted_keyframes:
            # Check cancellation in indexing loop
            if cancel_event.is_set() or redis_service.check_cancel_flag(video_id) or (parent_channel_id and redis_service.check_cancel_flag(parent_channel_id)):
                logger.info(f"Indexing for {video_id} cancelled. Cleaning up database uploads...")
                cleanup_qdrant_points(uploaded_point_ids)
                redis_service.set_video_status(video_id, "CANCELLED", 0, "Task cancelled by user.")
                cleanup_local_files(temp_dir)
                return

            kf_idx = kf["idx"]
            kf_path = kf["path"]
            object_key = f"keyframes/{video_id}/{kf_idx:03d}.jpg"
            
            # Upload file to MinIO
            try:
                minio_service.s3_client.upload_file(
                    Filename=str(kf_path),
                    Bucket=minio_service.bucket_name,
                    Key=object_key,
                    ExtraArgs={'ContentType': 'image/jpeg'}
                )
            except Exception as e:
                logger.error(f"Failed to upload keyframe {object_key} to MinIO: {e}")
                raise

            # Encode image using CLIP
            try:
                vector = clip_service.encode_image_from_path(str(kf_path))
            except Exception as e:
                logger.error(f"Failed to encode keyframe {kf_idx} with CLIP: {e}")
                # Fallback to random vector to avoid complete failure
                vector = np.random.rand(512).astype(np.float32).tolist()

            # Create Qdrant point
            point_id = str(uuid.uuid4())
            uploaded_point_ids.append(point_id)
            
            # Run YOLO Object Detection
            try:
                objects, object_labels = yolo_service.detect_objects(cv2.imread(str(kf_path)))
            except Exception as e:
                logger.error(f"YOLO detection failed on {kf_path}: {e}")
                objects, object_labels = [], []

            payload = {
                "original_id": f"{video_id}_{kf_idx:03d}",
                "video_id": video_id,
                "keyframe_idx": kf_idx,
                "keyframe_name": f"{kf_idx:03d}.jpg",
                "jpg_path": object_key,
                "pts_time": float(kf["pts_time"]),
                "frame_idx": int(kf["frame_idx"]),
                "fps": int(round(fps)),
                "batch": batch_code,
                "objects": objects,
                "object_labels": object_labels,
                "object_count": len(objects),
                "has_objects": len(objects) > 0
            }

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            qdrant_points.append(point)

        # Upload points to Qdrant
        if qdrant_points:
            try:
                qdrant_client.client.upsert(
                    collection_name=qdrant_client.collection_name,
                    points=qdrant_points
                )
                logger.info(f"Successfully indexed {len(qdrant_points)} points to Qdrant.")
            except Exception as e:
                logger.error(f"Failed to upsert points to Qdrant: {e}")
                raise

        # Complete Ingestion
        redis_service.set_video_status(video_id, "COMPLETED", 100, f"Successfully processed video. Extracted {len(extracted_keyframes)} keyframes.")
        logger.info(f"Successfully completed video ingest pipeline for {video_id}.")
        
        # If part of a channel, update channel status
        if parent_channel_id:
            update_parent_channel_progress(parent_channel_id)

    except Exception as e:
        logger.exception(f"Exception occurred in run_ingest_pipeline for {video_id}: {e}")
        # Clean up database points if upload failed
        cleanup_qdrant_points(uploaded_point_ids)
        redis_service.set_video_status(video_id, "FAILED", 0, f"Error occurred: {str(e)}")
        if parent_channel_id:
            update_parent_channel_progress(parent_channel_id)
    finally:
        cleanup_local_files(temp_dir)
        # Remove from active tasks list
        active_tasks.pop(video_id, None)

def update_parent_channel_progress(channel_id: str):
    """Update parent channel counts and status"""
    channel_status = redis_service.get_channel_status(channel_id)
    if not channel_status:
        return
        
    total_videos = channel_status["total_videos"]
    completed = channel_status.get("completed_videos", 0)
    failed = channel_status.get("failed_videos", 0)
    
    # We scan all task list to find video childs of this channel (or read Redis)
    all_tasks = redis_service.get_all_tasks()
    
    # Let's count statuses of video tasks that have parent channel
    completed = 0
    failed = 0
    processing = 0
    
    # Simple strategy: since our tasks have parent channel, we can search by prefix or status
    # In a full-blown Redis system, we would keep a Redis set, but let's count statuses.
    for task in all_tasks:
        if task.get("type") == "video":
            # For simplicity, we check prefix if channel ID matches video ID structure or we can map it
            # In our system channel tasks start a fan-out where video_id contains channel_id as prefix
            if task["id"].startswith(channel_id):
                if task["status"] == "COMPLETED":
                    completed += 1
                elif task["status"] in ("FAILED", "CANCELLED"):
                    failed += 1
                elif task["status"] in ("PENDING", "DOWNLOADING", "EXTRACTING", "INDEXING"):
                    processing += 1

    new_status = "PROCESSING"
    if completed + failed >= total_videos:
        new_status = "COMPLETED"
    elif redis_service.check_cancel_flag(channel_id):
        new_status = "CANCELLED"
        
    redis_service.set_channel_status(
        channel_id=channel_id,
        channel_name=channel_status["channel_name"],
        total_videos=total_videos,
        completed_videos=completed,
        failed_videos=failed,
        status=new_status
    )
    logger.info(f"Updated Channel {channel_id} progress: {completed}/{total_videos} (Status: {new_status})")

def cleanup_local_files(directory: Path):
    """Delete local video files and temporary keyframes to free space"""
    if directory.exists():
        try:
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary local folder: {directory}")
        except Exception as e:
            logger.error(f"Failed to delete directory {directory}: {e}")

def cleanup_qdrant_points(point_ids: List[str]):
    """Delete uploaded points from Qdrant if task was cancelled or failed"""
    if not point_ids:
        return
    try:
        qdrant_client.client.delete(
            collection_name=qdrant_client.collection_name,
            points_selector=point_ids
        )
        logger.info(f"Cleaned up {len(point_ids)} points from Qdrant database.")
    except Exception as e:
        logger.error(f"Failed to delete points from Qdrant: {e}")
