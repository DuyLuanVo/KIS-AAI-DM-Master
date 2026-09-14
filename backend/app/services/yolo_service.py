"""
YOLO Service for Architectural Element Detection using YOLO-World
Supports dynamic context-driven open-vocabulary object detection tailored to Vietnamese architecture.
"""
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from loguru import logger

# Bảng phân loại cấu kiện kiến trúc động theo ngữ cảnh
DEFAULT_ARCH_CLASSES = [
    "curved roof", "tiled roof", "wooden column", "temple gate",
    "stone statue", "bell tower", "arched entrance", "dragon carving",
    "brick wall", "courtyard", "human scale"
]

CONTEXT_TAXONOMY = {
    "heritage_temple": [
        "curved roof", "tiled roof", "wooden column", "temple gate",
        "stone statue", "dragon carving", "bell tower", "altar", "lotus pond", "stele", "human scale"
    ],
    "church_colonial": [
        "bell tower", "arched window", "steeple", "cross", "colonnade",
        "pediment", "arched entrance", "balcony", "dome", "courtyard", "human scale"
    ],
    "residential_vernacular": [
        "wooden louvers", "balcony", "tiled roof", "courtyard", "brick wall",
        "staircase", "terrace", "veranda", "window frame", "human scale"
    ],
    "interior": [
        "altar", "wooden screen", "carved chair", "ceiling beam", "pendant light",
        "dining table", "wooden floor", "human scale"
    ]
}

# Giữ nhãn kiến trúc chuẩn tiếng Anh đồng bộ với Qdrant và bộ lọc
FRIENDLY_LABEL_MAP = {c: c for c in DEFAULT_ARCH_CLASSES}


class YoloService:
    def __init__(self):
        self.model = None
        self._initialized = False
        self._is_yolo_world = False
        self._current_classes = []

    def initialize(self):
        """Khởi tạo mô hình YOLO-World open-vocabulary hoặc fallback YOLOv8"""
        if self._initialized:
            return True

        try:
            from ultralytics import YOLO
            logger.info("Đang khởi tạo mô hình YOLO-World (yolov8s-worldv2.pt)...")
            try:
                self.model = YOLO("yolov8s-worldv2.pt")
                self._is_yolo_world = True
                self.set_active_classes(DEFAULT_ARCH_CLASSES)
                logger.info("✅ YOLO-World đã tải thành công với kiến trúc Open-Vocabulary.")
            except Exception as ew:
                logger.warning(f"Không thể tải yolov8s-worldv2.pt ({ew}), fallback sang yolov8n.pt...")
                self.model = YOLO("yolov8n.pt")
                self._is_yolo_world = False
                
            self._initialized = True
            return True
        except ImportError:
            logger.warning("ultralytics package chưa được cài đặt. YOLO sẽ chạy ở chế độ fallback trống.")
            return False
        except Exception as e:
            logger.error(f"Lỗi khởi tạo mô hình YOLO: {e}")
            return False

    def get_dynamic_classes(self, context_categories: Optional[List[str]] = None, custom_classes: Optional[List[str]] = None) -> List[str]:
        """Tự động suy luận danh sách cấu kiện kiến trúc theo ngữ cảnh metadata của ảnh"""
        if custom_classes:
            return custom_classes

        if not context_categories:
            return DEFAULT_ARCH_CLASSES

        # Phân tích từ khóa trong danh mục metadata
        text_context = " ".join([str(c).lower() for c in context_categories])
        chosen_classes = set()

        if any(k in text_context for k in ["temple", "pagoda", "chùa", "đền", "đình", "miếu", "tháp", "buddhist", "communal"]):
            chosen_classes.update(CONTEXT_TAXONOMY["heritage_temple"])

        if any(k in text_context for k in ["church", "cathedral", "nhà thờ", "colonial", "french", "đông dương", "indochina"]):
            chosen_classes.update(CONTEXT_TAXONOMY["church_colonial"])

        if any(k in text_context for k in ["house", "villa", "nhà ở", "biệt thự", "residential", "vernacular", "rường", "phố"]):
            chosen_classes.update(CONTEXT_TAXONOMY["residential_vernacular"])

        if any(k in text_context for k in ["interior", "nội thất", "phòng", "gian"]):
            chosen_classes.update(CONTEXT_TAXONOMY["interior"])

        if not chosen_classes:
            return DEFAULT_ARCH_CLASSES

        # Đảm bảo giữ lại các lớp nền tảng phổ biến
        chosen_classes.add("human scale")
        return list(chosen_classes)

    def set_active_classes(self, classes: List[str]):
        """Cập nhật danh sách lớp động cho YOLO-World"""
        if not self._is_yolo_world or not self.model:
            return
        if self._current_classes == classes:
            return
        try:
            self.model.set_classes(classes)
            self._current_classes = classes
            logger.debug(f"Đã cập nhật YOLO-World classes động ({len(classes)} lớp): {classes[:4]}...")
        except Exception as e:
            logger.warning(f"Không thể cập nhật dynamic classes cho YOLO-World: {e}")

    def detect_objects(
        self, 
        frame: np.ndarray, 
        context_categories: Optional[List[str]] = None,
        custom_classes: Optional[List[str]] = None,
        conf_threshold: float = 0.20
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Nhận diện các cấu kiện kiến trúc trong ảnh sử dụng YOLO-World theo ngữ cảnh động.
        """
        if not self._initialized:
            success = self.initialize()
            if not success or not self.model:
                return [], []

        try:
            # Nếu dùng YOLO-World, áp dụng danh sách cấu kiện động theo ngữ cảnh ảnh
            if self._is_yolo_world:
                active_classes = self.get_dynamic_classes(context_categories, custom_classes)
                self.set_active_classes(active_classes)

            # Thực thi inference
            results = self.model(frame, conf=conf_threshold, verbose=False)
            if not results:
                return [], []

            result = results[0]
            objects = []
            object_labels = set()

            h, w = frame.shape[:2]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return [], []

            for box in boxes:
                class_id = int(box.cls[0])
                if self._is_yolo_world and hasattr(self.model, 'names') and class_id in self.model.names:
                    raw_label = self.model.names[class_id]
                elif hasattr(self, '_current_classes') and class_id < len(self._current_classes):
                    raw_label = self._current_classes[class_id]
                else:
                    raw_label = self.model.names.get(class_id, f"class_{class_id}") if hasattr(self.model, 'names') else f"class_{class_id}"

                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                bbox_normalized = [
                    round(max(0.0, x1) / w, 4),
                    round(max(0.0, y1) / h, 4),
                    round(min(float(w), x2) / w, 4),
                    round(min(float(h), y2) / h, 4)
                ]

                friendly_label = FRIENDLY_LABEL_MAP.get(raw_label, raw_label)

                objects.append({
                    "label": raw_label,
                    "display_name": friendly_label,
                    "confidence": round(conf, 4),
                    "bbox": bbox_normalized
                })
                object_labels.add(raw_label)

            return objects, list(object_labels)

        except Exception as e:
            logger.error(f"Lỗi trong quá trình nhận diện YOLO: {e}")
            return [], []

yolo_service = YoloService()
