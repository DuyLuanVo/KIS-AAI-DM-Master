"""
Redis service for storing ingest tasks status and cancellation flags.
Supports automatic fallback to in-memory dictionary if Redis is unavailable.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import redis
from loguru import logger
from app.config import settings

class RedisService:
    """Redis service with in-memory fallback for local development"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.use_fallback = False
        
        # In-memory storage for fallback mode
        self._memory_db: Dict[str, str] = {}
        self._task_list: List[str] = [] # Track task IDs to list them

        try:
            logger.info(f"Connecting to Redis at {settings.redis_host}:{settings.redis_port}")
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=2.0
            )
            # Ping to test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory state storage.")
            self.use_fallback = True

    def _get_key(self, prefix: str, key_id: str) -> str:
        return f"ingest:{prefix}:{key_id}"

    def _track_task(self, task_id: str):
        """Track task IDs to be able to list them later"""
        if self.use_fallback:
            if task_id not in self._task_list:
                self._task_list.append(task_id)
        else:
            try:
                self.redis_client.sadd("ingest:tasks_set", task_id)
            except Exception as e:
                logger.error(f"Error tracking task in Redis: {e}")

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Return a list of all tracked tasks with their status"""
        tasks = []
        task_ids = []
        
        if self.use_fallback:
            task_ids = list(self._task_list)
        else:
            try:
                task_ids = list(self.redis_client.smembers("ingest:tasks_set"))
            except Exception as e:
                logger.error(f"Error listing Redis tasks: {e}")
                task_ids = list(self._task_list)

        for task_id in task_ids:
            # Try to get single video status
            status_data = self.get_video_status(task_id)
            if status_data:
                status_data["id"] = task_id
                status_data["type"] = "video"
                tasks.append(status_data)
                continue
            
            # If not video, try to get channel status
            channel_data = self.get_channel_status(task_id)
            if channel_data:
                channel_data["id"] = task_id
                channel_data["type"] = "channel"
                tasks.append(channel_data)

        # Sort tasks by creation time (or just simple sort)
        return tasks

    def set_video_status(self, video_id: str, status: str, progress: float, message: str, video_url: Optional[str] = None):
        """Set Single Video Status"""
        key = self._get_key("video:status", video_id)
        self._track_task(video_id)
        
        # Get existing details to preserve video_url
        existing = self.get_video_status(video_id) or {}
        url = video_url or existing.get("video_url") or ""

        data = {
            "status": status,
            "progress": round(progress, 1),
            "message": message,
            "video_url": url,
            "updated_at": str(datetime.now())
        }
        
        value_str = json.dumps(data)
        if self.use_fallback:
            self._memory_db[key] = value_str
            logger.info(f"[In-Memory] Task {video_id} -> {status} ({progress}%)")
        else:
            try:
                # Expire status after 24 hours (86400 seconds)
                self.redis_client.set(key, value_str, ex=86400)
                # Publish event for WebSockets
                self.redis_client.publish("ingest:status_channel", json.dumps({"id": video_id, "type": "video", **data}))
            except Exception as e:
                logger.error(f"Redis set_video_status failed: {e}")
                self._memory_db[key] = value_str

    def get_video_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get Single Video Status"""
        key = self._get_key("video:status", video_id)
        if self.use_fallback:
            val = self._memory_db.get(key)
        else:
            try:
                val = self.redis_client.get(key)
            except Exception as e:
                logger.error(f"Redis get_video_status failed: {e}")
                val = self._memory_db.get(key)
                
        return json.loads(val) if val else None

    def set_channel_status(self, channel_id: str, channel_name: str, total_videos: int, completed_videos: int, failed_videos: int, status: str, channel_url: Optional[str] = None):
        """Set Channel Status"""
        key = self._get_key("channel:status", channel_id)
        self._track_task(channel_id)
        
        # Get existing details
        existing = self.get_channel_status(channel_id) or {}
        url = channel_url or existing.get("channel_url") or ""

        data = {
            "channel_name": channel_name,
            "total_videos": total_videos,
            "completed_videos": completed_videos,
            "failed_videos": failed_videos,
            "status": status,
            "channel_url": url
        }
        
        value_str = json.dumps(data)
        if self.use_fallback:
            self._memory_db[key] = value_str
            logger.info(f"[In-Memory] Channel {channel_id} -> {status} ({completed_videos}/{total_videos})")
        else:
            try:
                # Expire channel status after 48 hours
                self.redis_client.set(key, value_str, ex=172800)
                # Publish event for WebSockets
                self.redis_client.publish("ingest:status_channel", json.dumps({"id": channel_id, "type": "channel", **data}))
            except Exception as e:
                logger.error(f"Redis set_channel_status failed: {e}")
                self._memory_db[key] = value_str

    def get_channel_status(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get Channel Status"""
        key = self._get_key("channel:status", channel_id)
        if self.use_fallback:
            val = self._memory_db.get(key)
        else:
            try:
                val = self.redis_client.get(key)
            except Exception as e:
                logger.error(f"Redis get_channel_status failed: {e}")
                val = self._memory_db.get(key)
                
        return json.loads(val) if val else None

    def set_cancel_flag(self, target_id: str):
        """Set cancel flag for a video or channel"""
        key = self._get_key("cancel", target_id)
        if self.use_fallback:
            self._memory_db[key] = "true"
            logger.info(f"[In-Memory] Cancellation flag set for {target_id}")
        else:
            try:
                # Cancel flag expires after 1 hour (3600 seconds)
                self.redis_client.set(key, "true", ex=3600)
            except Exception as e:
                logger.error(f"Redis set_cancel_flag failed: {e}")
                self._memory_db[key] = "true"

    def check_cancel_flag(self, target_id: str) -> bool:
        """Check if a video or channel has a cancel flag set"""
        key = self._get_key("cancel", target_id)
        if self.use_fallback:
            return self._memory_db.get(key) == "true"
        else:
            try:
                return self.redis_client.get(key) == "true"
            except Exception as e:
                logger.error(f"Redis check_cancel_flag failed: {e}")
                return self._memory_db.get(key) == "true"

# Global Redis service instance
redis_service = RedisService()
