# 🚀 Hướng dẫn triển khai Qdrant Video Keyframes Loader (Offline Batch Data)

Tài liệu này hướng dẫn nạp tập dữ liệu offline gốc (như CLIP features `.npy`, Metadata `.csv`, và Objects `.json` từ thư mục `data/` cục bộ) vào **Qdrant Vector Database**.

*(Để chạy hệ thống nạp tự động qua giao diện Web và URL YouTube, vui lòng đọc tài liệu [README.md](file:///d:/Study/Postgrad/HK2/AAI%26DM/KIS/README.md) ở thư mục gốc).*

---

## 🛠️ Chuẩn bị

### 1. Cài đặt các thư viện Python
Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Cụm dịch vụ Docker
Khởi chạy toàn bộ hạ tầng (bao gồm Qdrant):
```bash
docker-compose up -d
```
*(Qdrant sẽ chạy tại cổng `6333` và dữ liệu được lưu trữ cố định trong thư mục `./qdrant_storage`)*

### 3. Cấu trúc dữ liệu cục bộ
Đảm bảo thư mục dữ liệu của bạn có cấu trúc như sau (mặc định đặt trong thư mục `data/` của dự án):
```
data/
├── clip-features-32-aic25-b1/clip-features-32/
│   ├── L21_V001.npy
│   ├── L21_V002.npy
│   └── ... (873 files)
├── map-keyframes-aic25-b1/map-keyframes/
│   ├── L21_V001.csv
│   ├── L21_V002.csv
│   └── ... (873 files)
├── objects-aic25-b1/objects/
│   ├── L21_V001/
│   │   ├── 0001.json
│   │   ├── 0002.json
│   │   └── ...
│   └── ...
└── Keyframes_L21/keyframes/
    ├── L21_V001/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── ...
```

---

## 🔧 Cấu hình Loader

### 1. Cập nhật đường dẫn trong `pipeline/loaders/run_loader.py`
Mở file [run_loader.py](file:///d:/Study/Postgrad/HK2/AAI%26DM/KIS/pipeline/loaders/run_loader.py) và cập nhật đường dẫn tương ứng với thư mục dữ liệu cục bộ của bạn:
```python
# Cập nhật đường dẫn data của bạn
DATA_ROOT = "D:/path/to/your/data"  # ⚠️ SỬA DÒNG NÀY

CLIP_DIR = "clip-features-32-aic25-b1/clip-features-32"
CSV_DIR = "map-keyframes-aic25-b1/map-keyframes"
OBJ_DIR = "objects-aic25-b1/objects"
```

---

## 🚀 Chạy Pipeline Nạp dữ liệu Offline

### Bước 1: Chạy thử nghiệm Loader
Mặc định, loader sẽ:
* Kiểm tra kết nối Qdrant tại `localhost:6333`.
* Tạo collection `video_keyframes` với cấu hình vector 512 dimensions (Cosine Distance).
* Đọc thử cấu trúc file dữ liệu để xác thực tính hợp lệ.

Để khởi chạy:
```bash
python pipeline/loaders/run_loader.py
```

### Bước 2: Nạp toàn bộ Dataset
Mặc định script chỉ xử lý **3 video đầu tiên** để demo. Để nạp toàn bộ 873 video, hãy chỉnh sửa trong file [run_loader.py](file:///d:/Study/Postgrad/HK2/AAI%26DM/KIS/pipeline/loaders/run_loader.py):
```python
# Tìm dòng này:
for npy_file in tqdm(video_files[:3], desc="Processing"):

# Đổi thành:
for npy_file in tqdm(video_files, desc="Processing"):
```
Sau đó chạy lại lệnh: `python pipeline/loaders/run_loader.py`.

---

## 📊 Định dạng Schema dữ liệu trong Qdrant

Mỗi điểm vector sau khi được nạp thành công sẽ có dạng:
```json
{
  "id": "e4a7a8d5-12a8-48b6-96a1-a47781b2a95c",  // Định dạng chuỗi UUID (Qdrant yêu cầu UUID chuẩn)
  "vector": [512 dimensions CLIP embedding],
  "payload": {
    "original_id": "L21_V001_001",
    "video_id": "L21_V001",
    "keyframe_idx": 1,
    "keyframe_name": "001.jpg",
    "jpg_path": "keyframes/L21_V001/001.jpg",
    "pts_time": 0.0,
    "frame_idx": 0,
    "fps": 30,
    "batch": "L21",
    "objects": [
      {"entity": "Lantern", "score": 0.79673874, "class_name": "/m/01jfsr", "bbox": [0.468, 0.366, 0.581, 0.712]}
    ],
    "object_labels": ["Lantern", "Skyscraper"],
    "object_count": 2,
    "has_objects": true
  }
}
```

---

## 🔍 Kiểm tra & Truy vấn dữ liệu

### 1. Truy cập Qdrant Web Dashboard
Mở trình duyệt: `http://localhost:6333/dashboard` để trực quan hóa collection `video_keyframes` và kiểm tra số lượng points đã nạp.

### 2. Chạy Unit Test kiểm tra tính năng nạp tự động SBD/Redis
Hệ thống nạp tự động mới có thể kiểm tra qua file unit test:
```bash
python tests/test_ingest.py
```
