"""
Unit test script for Video Ingestion, Redis state tracking, and SBD logic.
"""
import unittest
import numpy as np
import cv2

import sys
from pathlib import Path
# Add backend directory to sys.path so we can import app modules directly
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

# Import project modules
from app.services.redis_service import redis_service
from app.services.ingest_worker_service import compute_hsv_histogram

class TestVideoIngest(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment"""
        # Ensure fallback state starts clean
        redis_service._memory_db.clear()
        redis_service._task_list.clear()

    def test_redis_service_fallback(self):
        """Test that redis_service stores and retrieves statuses correctly (even in fallback)"""
        video_id = "test_video_123"
        redis_service.set_video_status(video_id, "DOWNLOADING", 45.0, "Test downloading video...")
        
        status = redis_service.get_video_status(video_id)
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "DOWNLOADING")
        self.assertEqual(status["progress"], 45.0)
        self.assertEqual(status["message"], "Test downloading video...")

    def test_redis_service_cancellation(self):
        """Test setting and checking cancel flags"""
        video_id = "test_cancel_video"
        self.assertFalse(redis_service.check_cancel_flag(video_id))
        
        redis_service.set_cancel_flag(video_id)
        self.assertTrue(redis_service.check_cancel_flag(video_id))

    def test_redis_service_channel(self):
        """Test channel status setting and retrieval"""
        channel_id = "test_channel_abc"
        redis_service.set_channel_status(
            channel_id=channel_id,
            channel_name="Test channel name",
            total_videos=5,
            completed_videos=2,
            failed_videos=1,
            status="PROCESSING"
        )
        
        status = redis_service.get_channel_status(channel_id)
        self.assertIsNotNone(status)
        self.assertEqual(status["channel_name"], "Test channel name")
        self.assertEqual(status["total_videos"], 5)
        self.assertEqual(status["completed_videos"], 2)
        self.assertEqual(status["failed_videos"], 1)
        self.assertEqual(status["status"], "PROCESSING")

    def test_hsv_histogram_sbd(self):
        """Test SBD HSV histogram computation"""
        # Create a mock BGR frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill it with red color
        frame[:, :] = [0, 0, 255]
        
        hist = compute_hsv_histogram(frame)
        self.assertIsNotNone(hist)
        self.assertEqual(hist.shape, (18, 16))
        
        # Test correlation on identical frame
        corr = cv2.compareHist(hist, hist, cv2.HISTCMP_CORREL)
        self.assertAlmostEqual(corr, 1.0, places=5)
        
        # Test correlation on completely different frame (green frame)
        green_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        green_frame[:, :] = [0, 255, 0]
        green_hist = compute_hsv_histogram(green_frame)
        
        corr_diff = cv2.compareHist(hist, green_hist, cv2.HISTCMP_CORREL)
        # Correlation should be much lower (close to -1.0 or 0.0)
        self.assertLess(corr_diff, 0.5)

if __name__ == "__main__":
    unittest.main()
