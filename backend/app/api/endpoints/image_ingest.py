"""
API router for Architecture Image Ingestion.
Provides endpoints for uploading batches of images, monitoring status, and cancellation.
"""
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, File, UploadFile, Form
from loguru import logger

from app.services.redis_service import redis_service
from app.services.image_worker_service import start_image_ingest_task, active_batch_tasks

router = APIRouter()

@router.post("/upload")
async def upload_and_ingest_images(
    files: List[UploadFile] = File(...),
    batch_name: Optional[str] = Form("Bộ sưu tập Kiến Trúc")
):
    """
    Receive multiple uploaded image files, save to temp storage, 
    and submit to background processing pipeline.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Vui lòng chọn ít nhất một tệp ảnh.")

    batch_id = f"ARCH_{str(uuid.uuid4())[:8].upper()}"
    temp_dir = Path("data/temp") / batch_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_filenames: List[str] = []
    
    try:
        for file in files:
            # Simple validation for image filenames
            filename = file.filename or f"image_{uuid.uuid4().hex[:6]}.jpg"
            # Ensure unique filename if duplicate names uploaded
            target_path = temp_dir / filename
            counter = 1
            while target_path.exists():
                stem = Path(filename).stem
                ext = Path(filename).suffix
                filename = f"{stem}_{counter}{ext}"
                target_path = temp_dir / filename
                counter += 1

            content = await file.read()
            with open(target_path, "wb") as f:
                f.write(content)
            saved_filenames.append(filename)
            
        logger.info(f"Saved {len(saved_filenames)} files for batch {batch_id} to {temp_dir}")

        # Initialize status in Redis
        display_name = batch_name or f"Album {batch_id}"
        redis_service.set_batch_status(
            batch_id=batch_id,
            batch_name=display_name,
            total_images=len(saved_filenames),
            completed_images=0,
            status="PENDING",
            message="Đã tiếp nhận file ảnh. Đang đưa vào hàng đợi AI...",
            progress=2.0
        )

        # Start background task
        start_image_ingest_task(
            batch_id=batch_id,
            batch_name=display_name,
            temp_dir=temp_dir,
            file_names=saved_filenames
        )

        return {
            "status": "success",
            "type": "image_batch",
            "batch_id": batch_id,
            "batch_name": display_name,
            "total_images": len(saved_filenames),
            "message": f"Đã gửi thành công {len(saved_filenames)} ảnh kiến trúc vào tiến trình xử lý AI."
        }

    except Exception as e:
        logger.exception(f"Failed to save uploaded files for {batch_id}: {e}")
        # Cleanup
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file tải lên: {str(e)}")


@router.get("/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Retrieve status of an image batch task"""
    status = redis_service.get_batch_status(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Tác vụ {batch_id} không tìm thấy.")
    return status


@router.get("/tasks")
async def get_all_tasks():
    """Retrieve a list of all tracked ingestion tasks (both videos and image batches)"""
    return redis_service.get_all_tasks()


@router.post("/cancel/{batch_id}")
async def cancel_batch_ingest(batch_id: str):
    """Cancel a running image batch ingestion task"""
    redis_service.set_cancel_flag(batch_id)
    
    if batch_id in active_batch_tasks:
        active_batch_tasks[batch_id].set()
        logger.info(f"Cancellation event triggered for running batch ID {batch_id}.")
        
    status = redis_service.get_batch_status(batch_id)
    total = status["total_images"] if status else 0
    name = status["batch_name"] if status else f"Album {batch_id}"
    
    redis_service.set_batch_status(
        batch_id=batch_id,
        batch_name=name,
        total_images=total,
        completed_images=status.get("completed_images", 0) if status else 0,
        status="CANCELLED",
        message="Yêu cầu hủy đã được gửi.",
        progress=status.get("progress", 0) if status else 0
    )
    return {"status": "success", "message": f"Yêu cầu hủy đã được gửi tới bộ xử lý cho batch {batch_id}."}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time progress updates"""
    await websocket.accept()
    logger.info("Image ingest monitoring WebSocket client connected.")
    try:
        while True:
            tasks = redis_service.get_all_tasks()
            await websocket.send_json({"type": "tasks_update", "tasks": tasks})
            await asyncio.sleep(1.0)
    except Exception as e:
        logger.info(f"WebSocket client disconnected: {e}")
