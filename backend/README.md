# Video Retrieval API Backend

FastAPI backend cho hệ thống Video Retrieval và nạp dữ liệu tự động bất đồng bộ với Qdrant vector database, MinIO object storage, Redis state store, và hàng đợi Kafka.

## Cấu trúc thư mục Backend

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Điểm khởi chạy chính & Lifespan setup
│   ├── config.py              # Cấu hình cài đặt (Redis, MinIO, Qdrant, CLIP, YOLO)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── api.py             # Router tổng kết hợp các endpoint
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── health.py      # Kiểm tra sức khỏe hệ thống và Qdrant
│   │       ├── video_search.py# API tìm kiếm lai (CLIP + YOLO)
│   │       └── video_ingest.py# API nạp video, quản lý task và WebSocket
│   ├── core/
│   │   ├── __init__.py
│   │   └── logging.py         # Cấu hình logging qua Loguru
│   ├── database/
│   │   ├── __init__.py
│   │   └── qdrant_client.py   # Client kết nối và truy vấn Qdrant
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Khai báo cấu trúc dữ liệu Pydantic
│   └── services/
│       ├── __init__.py
│       ├── clip_service.py    # Nhúng vector văn bản & ảnh bằng CLIP
│       ├── yolo_service.py    # Phát hiện vật thể YOLOv8
│       ├── redis_service.py   # Lưu trạng thái, cờ hủy (Có In-Memory Fallback)
│       └── ingest_worker_service.py # Xử lý tác vụ ngầm (Tải, cắt SBD/TIME, upload)
├── requirements.txt           # Danh sách thư viện Python
├── env.example               # File cấu hình môi trường mẫu
└── README.md                 # Tài liệu hướng dẫn này
```

---

## Các API Endpoints chính

### 1. Tìm kiếm (Video Search)
* `POST /api/v1/videos/search/text` - Tìm kiếm bằng câu văn bản mô tả + lọc nhãn vật thể YOLO.
* `POST /api/v1/videos/search/image` - Tìm kiếm bằng ảnh tương tự (Base64) + lọc nhãn vật thể YOLO.
* `GET /api/v1/videos/keyframes/{jpg_path:path}` - Lấy ảnh keyframe (Redirect trực tiếp từ MinIO qua pre-signed URL).

### 2. Nạp Video Tự động (Video Ingestion)
* `POST /api/v1/videos/ingest` - Khởi tạo tác vụ nạp video từ YouTube (hỗ trợ video lẻ, playlist hoặc kênh có `@`).
* `GET /api/v1/videos/ingest/tasks` - Lấy danh sách toàn bộ tác vụ đang chạy và đã hoàn thành.
* `GET /api/v1/videos/ingest/status/video/{video_id}` - Trạng thái tiến độ (%) của một video cụ thể.
* `GET /api/v1/videos/ingest/status/channel/{channel_id}` - Trạng thái của một tiến trình kênh cha.
* `POST /api/v1/videos/ingest/cancel/video/{video_id}` - Phát cờ hủy tác vụ xử lý của một video cụ thể.
* `POST /api/v1/videos/ingest/cancel/channel/{channel_id}` - Phát cờ hủy xử lý của toàn bộ video thuộc một kênh.
* `WebSocket /api/v1/videos/ingest/ws` - Đẩy danh sách cập nhật tiến trình của tất cả các task thời gian thực.

### 3. Kiểm tra sức khỏe (Health Check)
* `GET /health` - Trạng thái sức khỏe chung của ứng dụng.
* `GET /health/qdrant` - Trạng thái kết nối đến cơ sở dữ liệu Qdrant.

---

## Hướng dẫn Khởi chạy API

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Cấu hình môi trường:**
   ```bash
   cp env.example .env
   ```
3. **Khởi chạy API Gateway:**
   ```bash
   python run.py
   ```
   * Cổng hoạt động mặc định: `http://localhost:8000`.
   * Tài liệu API tương tác (Swagger UI): `http://localhost:8000/docs`.

---

## Các Dịch vụ AI & Logic chính

### 1. Shot Boundary Detection (SBD)
Mô hình phát hiện chuyển cảnh phân tích biểu đồ màu sắc HSV Color Histogram của các frame liên tiếp trong OpenCV. Nếu sự chênh lệch (đo bằng phương pháp correlation) vượt quá ngưỡng `SBD_THRESHOLD` (mặc định `0.3`), nó sẽ lưu khung hình hiện tại làm keyframe đại diện.

### 2. YOLOv8 Object Detection
Sử dụng mô hình YOLOv8n (`yolov8n.pt`) của Ultralytics để nhận dạng các vật thể trong keyframe (độ tin cậy > 0.5), từ đó thu thập nhãn đối tượng (labels) và tọa độ bounding box chuẩn hóa để ghi vào payload của Qdrant database. Hỗ trợ tự động chạy ở chế độ fallback không nhãn nếu môi trường chưa cài đặt `ultralytics`.

### 3. Cờ Hủy Tác vụ Bất đồng bộ
Khi người dùng kích hoạt hủy, cờ hủy được lưu trên Redis. Worker xử lý video định kỳ kiểm tra cờ này ở các giai đoạn (Tải xuống, Cắt ảnh, Xử lý AI, Index DB). Nếu phát hiện cờ hủy, worker sẽ tự ngắt, dọn sạch dữ liệu point dở dang trên Qdrant và các ảnh đã tải lên MinIO.
