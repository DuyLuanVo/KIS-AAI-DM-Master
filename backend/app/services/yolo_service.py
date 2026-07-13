"""
YOLO Service for Object Detection in Keyframes
"""
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from loguru import logger

class YoloService:
    def __init__(self):
        self.model = None
        self._initialized = False

    def initialize(self):
        """Lazy load the YOLO model to save startup time and memory if not used"""
        if self._initialized:
            return True

        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8 model (yolov8n.pt)...")
            # This will load or download the small yolov8n model (6MB)
            self.model = YOLO("yolov8n.pt")
            self._initialized = True
            logger.info("YOLOv8 model loaded successfully.")
            return True
        except ImportError:
            logger.warning(
                "ultralytics package is not installed. "
                "YOLO object detection will run in fallback mode (empty labels)."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            return False

    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Detect objects in a frame using YOLO.
        Returns:
            - objects: List of dicts with keys: 'label', 'confidence', 'bbox' ([x1, y1, x2, y2] normalized)
            - object_labels: List of unique labels detected
        """
        if not self._initialized:
            success = self.initialize()
            if not success or not self.model:
                return [], []

        try:
            # Perform inference
            results = self.model(frame, verbose=False)
            if not results:
                return [], []

            result = results[0]
            objects = []
            object_labels = set()

            h, w = frame.shape[:2]

            # Parse bounding boxes, confidences, and class IDs
            boxes = result.boxes
            for box in boxes:
                # Class name
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                
                # Confidence score
                conf = float(box.conf[0])
                
                # Normalize bounding box coordinates
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                bbox_normalized = [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(x2 / w, 4),
                    round(y2 / h, 4)
                ]

                objects.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": bbox_normalized
                })
                object_labels.add(label)

            return objects, list(object_labels)

        except Exception as e:
            logger.error(f"Error during YOLO object detection: {e}")
            return [], []

yolo_service = YoloService()
