# 🚀 Hướng dẫn triển khai Qdrant Video Keyframes Loader

## 📋 Tổng quan

File này hướng dẫn bạn triển khai pipeline để load dữ liệu video keyframes vào Qdrant Vector Database theo đúng cấu trúc dữ liệu mà bạn đã mô tả.

## 🛠️ Chuẩn bị

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:

```bash
pip install numpy pandas tqdm qdrant-client
```

### 2. Cài đặt và chạy Qdrant

**Sử dụng Docker (khuyến nghị):**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Hoặc tải và cài đặt binary:**
- Tải từ: https://github.com/qdrant/qdrant/releases
- Chạy: `./qdrant`

### 3. Cấu trúc dữ liệu

Đảm bảo dữ liệu của bạn có cấu trúc như sau:

```
/path/to/your/data/
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

## 🔧 Cấu hình

### Mở file `run_loader.py` và cập nhật:

```python
# Cập nhật đường dẫn data của bạn
DATA_ROOT = "/path/to/your/data"  # ⚠️ SỬA DÒNG NÀY

# Kiểm tra các đường dẫn khác nếu cần
CLIP_DIR = "clip-features-32-aic25-b1/clip-features-32"
CSV_DIR = "map-keyframes-aic25-b1/map-keyframes"
OBJ_DIR = "objects-aic25-b1/objects"
```

### Kiểm tra cấu hình Qdrant:

```python
QDRANT_HOST = "localhost"    # Địa chỉ Qdrant server
QDRANT_PORT = 6333           # Port Qdrant
COLLECTION = "video_keyframes"  # Tên collection
```

## 🚀 Chạy Pipeline

### Bước 1: Test cấu hình

```bash
python run_loader.py
```

Script sẽ:
- ✅ Kiểm tra đường dẫn dữ liệu
- ✅ Kiểm tra kết nối Qdrant
- ✅ Validate mẫu file dữ liệu
- ✅ Tạo collection trong Qdrant

### Bước 2: Chạy load dữ liệu

**Mặc định script chỉ xử lý 3 video đầu tiên để demo.**

Để xử lý tất cả video, sửa dòng này trong `run_loader.py`:

```python
# Từ:
for npy_file in tqdm(video_files[:3], desc="Processing"):

# Thành:
for npy_file in tqdm(video_files, desc="Processing"):
```

### Bước 3: Theo dõi tiến độ

Script sẽ hiển thị:

```
🎬 Qdrant Video Keyframes Loader
========================================
🔍 Checking setup...
✅ Setup looks good!
📊 Found 873 video files
✅ L21_V001: (307, 512)
✅ L21_V002: (262, 512)
✅ L21_V003: (286, 512)
📈 Sample check: 3/3 files valid
✅ Connected to Qdrant at localhost:6333
✅ Created collection: video_keyframes

📊 Processing 873 videos...
(Processing first 3 as demo - edit script to process all)
Processing: 100%|██████████| 3/3 [00:15<00:00,  5.12s/it]
✅ L21_V001: 307 points uploaded
✅ L21_V002: 262 points uploaded
✅ L21_V003: 286 points uploaded

========================================
🎉 PROCESSING COMPLETE
Videos processed: 3
Total points: 855
Duration: 15.2 seconds
Collection size: 855 points

✅ Ready for search queries!
```

## 📊 Schema dữ liệu trong Qdrant

Mỗi point trong Qdrant sẽ có cấu trúc:

```json
{
  "id": "L21_V001_001",
  "vector": [512 dimensions CLIP embedding],
  "payload": {
    "video_id": "L21_V001",
    "keyframe_idx": 1,
    "keyframe_name": "001.jpg",
    "jpg_path": "Keyframes_L21/keyframes/L21_V001/001.jpg",
    "pts_time": 0.0,
    "frame_idx": 0,
    "fps": 30,
    "batch": "L21",
    "objects": [
      {"entity": "Lantern", "score": 0.79},
      {"entity": "Skyscraper", "score": 0.68}
    ],
    "object_labels": ["Lantern", "Skyscraper"],
    "object_count": 2,
    "has_objects": true
  }
}
```

## 🔍 Kiểm tra kết quả

### Sử dụng Qdrant Web UI:

1. Mở browser: http://localhost:6333/dashboard
2. Chọn collection: `video_keyframes`
3. Xem points và payload

### Sử dụng Python để query:

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Lấy thông tin collection
info = client.get_collection("video_keyframes")
print(f"Points count: {info.points_count}")

# Lấy một point mẫu
points = client.scroll(
    collection_name="video_keyframes",
    limit=1
)
print(points[0])  # In ra point đầu tiên
```

## ⚠️ Troubleshooting

### Lỗi thường gặp:

1. **"DATA_ROOT not found"**
   - Kiểm tra đường dẫn DATA_ROOT có đúng không
   - Đảm bảo thư mục tồn tại và có quyền đọc

2. **"Missing directory"**
   - Kiểm tra cấu trúc thư mục con
   - Đảm bảo tên thư mục khớp với config

3. **"Qdrant connection failed"**
   - Kiểm tra Qdrant server đã chạy chưa
   - Kiểm tra port 6333 có bị block không

4. **"Wrong shape"**
   - File .npy có thể bị lỗi
   - Kiểm tra file có đúng format (N, 512) không

### Debug logs:

Kiểm tra file log: `qdrant_loader.log`

### Giảm memory usage:

Nếu gặp lỗi memory, giảm `BATCH_SIZE`:

```python
BATCH_SIZE = 100  # Từ 500 xuống 100
```

## 📈 Tối ưu hóa

### Cho dataset lớn:

1. **Tăng batch size** (nếu có đủ RAM):
   ```python
   BATCH_SIZE = 1000
   ```

2. **Sử dụng multiple workers** (cần modify code):
   ```python
   # Thêm concurrent processing
   from concurrent.futures import ThreadPoolExecutor
   ```

3. **Monitor memory usage**:
   ```bash
   htop  # Linux/Mac
   # hoặc
   Task Manager  # Windows
   ```

### Kiểm tra hiệu suất:

```python
import time
start = time.time()
# ... processing code ...
print(f"Duration: {time.time() - start:.2f}s")
```

## 🎯 Next Steps

Sau khi load xong dữ liệu:

1. **Implement search API** với CLIP text encoding
2. **Build web interface** cho user queries
3. **Add caching layer** cho popular queries
4. **Monitor query performance**

---

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs: `qdrant_loader.log`
2. Xem Qdrant docs: https://qdrant.tech/documentation/
3. Test với subset nhỏ trước khi chạy full dataset
