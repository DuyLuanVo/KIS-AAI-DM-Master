"""
API router for Automated Video Ingestion.
Provides endpoints for starting, checking, cancelling, and viewing tasks.
"""
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

# Import services
from app.services.redis_service import redis_service
from app.services.ingest_worker_service import start_ingestion_task, active_tasks

router = APIRouter()

class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video or channel/playlist URL")
    extraction_method: Optional[str] = Field(default=None, description="SBD or TIME")
    time_interval: Optional[float] = Field(default=None, description="Seconds per frame if method is TIME")
    sbd_threshold: Optional[float] = Field(default=None, description="Shot Boundary threshold if method is SBD")
    video_id: Optional[str] = Field(default=None, description="Optional custom ID for the video")

@router.post("")
async def start_ingestion(request: IngestRequest):
    """
    Start video or channel ingestion task.
    If the URL is a channel/playlist, it retrieves all video IDs and starts them as separate tasks.
    """
    url = request.url
    method = request.extraction_method
    time_interval = request.time_interval
    sbd_threshold = request.sbd_threshold
    
    # Simple check for channel/playlist URL
    is_channel = "channel" in url or "playlist" in url or "list=" in url or "user" in url or "@" in url or "/c/" in url
    
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed on the system.")

    if is_channel:
        logger.info(f"Processing channel/playlist ingestion: {url}")
        try:
            ydl_opts = {
                'extract_flat': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(url, download=False)
                
            if not result:
                raise ValueError("Could not retrieve channel details.")
                
            channel_name = result.get('title') or "YouTube Playlist"
            entries = result.get('entries') or []
            
            # Flatten nested playlists if yt-dlp returns multiple tabs (e.g. Videos and Shorts)
            flat_entries = []
            for entry in entries:
                if entry.get('_type') == 'playlist':
                    flat_entries.extend(entry.get('entries') or [])
                else:
                    flat_entries.append(entry)
            
            # Filter entries to only keep actual videos (YouTube video ID is exactly 11 characters) and remove duplicates
            seen_ids = set()
            valid_entries = []
            for e in flat_entries:
                v_id = e.get('id')
                if v_id and len(v_id) == 11 and v_id not in seen_ids:
                    seen_ids.add(v_id)
                    valid_entries.append(e)
            
            if not valid_entries:
                raise ValueError("No valid videos found in the channel/playlist.")
                
            channel_id = f"chan_{str(uuid.uuid4())[:8]}"
            logger.info(f"Found {len(valid_entries)} videos in channel/playlist {channel_name}. Generating ID: {channel_id}")
            
            # Save channel status
            redis_service.set_channel_status(
                channel_id=channel_id,
                channel_name=channel_name,
                total_videos=len(valid_entries),
                completed_videos=0,
                failed_videos=0,
                status="PROCESSING",
                channel_url=url
            )
            
            # Start each video as a child task
            for idx, entry in enumerate(valid_entries):
                video_url_id = entry.get('id')
                video_url = f"https://www.youtube.com/watch?v={video_url_id}"
                
                # Derive a clean video ID
                child_video_id = f"{channel_id}_v{video_url_id[:8]}"
                
                # Initialize status as pending
                redis_service.set_video_status(
                    video_id=child_video_id,
                    status="PENDING",
                    progress=0,
                    message="Waiting in channel queue...",
                    video_url=video_url
                )
                
                # Submit ingestion
                start_ingestion_task(
                    video_url=video_url,
                    video_id=child_video_id,
                    parent_channel_id=channel_id,
                    method=method,
                    time_interval=time_interval,
                    sbd_threshold=sbd_threshold
                )
                
            return {
                "type": "channel",
                "channel_id": channel_id,
                "channel_name": channel_name,
                "total_videos": len(valid_entries),
                "message": f"Successfully queued channel ingestion task '{channel_name}' with {len(valid_entries)} videos."
            }
            
        except Exception as e:
            logger.error(f"Failed to process channel flat playlist: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process channel/playlist URL: {str(e)}")
            
    else:
        logger.info(f"Processing single video ingestion: {url}")
        
        # Resolve clean video ID
        video_id = request.video_id
        if not video_id:
            # Extract video ID from URL or use UUID
            try:
                ydl_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_title_id = info.get('id') or str(uuid.uuid4())[:8]
                    video_id = f"L21_V{video_title_id[:8]}"
            except Exception:
                video_id = f"L21_V{str(uuid.uuid4())[:8]}"

        # Initialize status
        redis_service.set_video_status(
            video_id=video_id,
            status="PENDING",
            progress=0,
            message="Queued video task...",
            video_url=url
        )
        
        # Start background task
        start_ingestion_task(
            video_url=url,
            video_id=video_id,
            method=method,
            time_interval=time_interval,
            sbd_threshold=sbd_threshold
        )
        
        return {
            "type": "video",
            "video_id": video_id,
            "message": f"Successfully queued video ingestion task for '{video_id}'."
        }

@router.get("/status/video/{video_id}")
async def get_video_status(video_id: str):
    """Retrieve status of a single video task"""
    status = redis_service.get_video_status(video_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task {video_id} not found.")
    return status

@router.get("/status/channel/{channel_id}")
async def get_channel_status(channel_id: str):
    """Retrieve status of a channel batch task"""
    status = redis_service.get_channel_status(channel_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found.")
    return status

@router.get("/tasks")
async def get_all_tasks():
    """Retrieve a list of all tracked video and channel ingestion tasks"""
    return redis_service.get_all_tasks()

@router.post("/cancel/video/{video_id}")
async def cancel_video_ingest(video_id: str):
    """Cancel a running video ingestion task"""
    redis_service.set_cancel_flag(video_id)
    
    # Trigger cancellation event in thread if active
    if video_id in active_tasks:
        active_tasks[video_id].set()
        logger.info(f"Cancellation event triggered for running video ID {video_id}.")
        
    # Update status immediately
    redis_service.set_video_status(
        video_id=video_id,
        status="CANCELLED",
        progress=0,
        message="Cancellation request received."
    )
    return {"status": "success", "message": f"Cancellation request sent for video {video_id}."}

@router.post("/cancel/channel/{channel_id}")
async def cancel_channel_ingest(channel_id: str):
    """Cancel a running channel ingestion task (cancels all pending child videos)"""
    redis_service.set_cancel_flag(channel_id)
    
    # Retrieve all tasks to cancel children
    all_tasks = redis_service.get_all_tasks()
    cancelled_children = 0
    
    for task in all_tasks:
        if task.get("type") == "video" and task["id"].startswith(channel_id):
            child_id = task["id"]
            redis_service.set_cancel_flag(child_id)
            if child_id in active_tasks:
                active_tasks[child_id].set()
            
            # If not yet finished, set to CANCELLED
            if task["status"] in ("PENDING", "DOWNLOADING", "EXTRACTING", "INDEXING"):
                redis_service.set_video_status(
                    video_id=child_id,
                    status="CANCELLED",
                    progress=0,
                    message="Cancelled by parent channel request."
                )
                cancelled_children += 1

    # Update channel status
    channel_status = redis_service.get_channel_status(channel_id)
    if channel_status:
        redis_service.set_channel_status(
            channel_id=channel_id,
            channel_name=channel_status["channel_name"],
            total_videos=channel_status["total_videos"],
            completed_videos=channel_status.get("completed_videos", 0),
            failed_videos=channel_status.get("failed_videos", 0) + cancelled_children,
            status="CANCELLED"
        )
        
    return {"status": "success", "message": f"Cancellation request sent for channel {channel_id} and {cancelled_children} child videos."}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time progress updates"""
    await websocket.accept()
    logger.info("Ingest monitoring WebSocket client connected.")
    try:
        while True:
            # Periodically poll status and send update
            tasks = redis_service.get_all_tasks()
            await websocket.send_json({"type": "tasks_update", "tasks": tasks})
            await asyncio.sleep(1.0)
    except Exception as e:
        logger.info(f"WebSocket client disconnected: {e}")
