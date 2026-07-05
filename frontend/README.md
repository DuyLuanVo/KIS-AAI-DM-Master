# Video Search Frontend

UI đơn giản cho video search với HTML, CSS, và JavaScript.

## Tính năng

- **Input nhiều câu query**: Nhập nhiều câu query, mỗi dòng một câu
- **Grid hiển thị 5x4**: Hiển thị 20 kết quả mỗi trang
- **Pagination**: Nút trước/sau để điều hướng
- **Tìm tiếp**: Sử dụng lại query ban đầu để tìm thêm kết quả
- **Responsive**: Tương thích với mobile và desktop

## Cấu trúc

```
frontend/
├── index.html          # File HTML chính
├── styles.css          # CSS styling
├── script.js           # JavaScript logic
├── mock-server.js      # Mock API server
├── test-mock.html      # Test page cho mock server
├── package.json        # Node.js configuration
└── README.md           # Documentation
```

## Cách sử dụng

### Option 1: Sử dụng Mock Server (Khuyến nghị)

#### 1. Chạy Mock Server
```bash
cd frontend
node mock-server.js
# Hoặc
npm start
```

#### 2. Mở frontend
Mở file `frontend/index.html` trong trình duyệt hoặc chạy local server:

```bash
cd frontend
python -m http.server 8080
# Sau đó mở http://localhost:8080
```

### Option 2: Sử dụng Backend thật

#### 1. Chạy backend trước
```bash
cd backend
python run.py
```

#### 2. Mở frontend
Mở file `frontend/index.html` trong trình duyệt hoặc chạy local server:

```bash
cd frontend
python -m http.server 8080
# Sau đó mở http://localhost:8080
```

### 3. Sử dụng UI

1. **Nhập query**: Nhập các câu query vào textarea, mỗi dòng một câu
2. **Tìm kiếm**: Click nút "Tìm kiếm" để tìm kiếm mới
3. **Tìm tiếp**: Click nút "Tìm tiếp" để tìm thêm kết quả với query hiện tại
4. **Điều hướng**: Sử dụng nút "Trước"/"Tiếp" để xem các trang kết quả

## API Integration

Frontend gọi các API endpoint của Backend:
- **Tìm kiếm bằng văn bản**:
  - **URL**: `http://localhost:8000/api/v1/videos/search/text`
  - **Method**: POST
  - **Body**: `{"query_texts": ["query1", "query2", ...], "object_filters": null}`
- **Tìm kiếm bằng hình ảnh**:
  - **URL**: `http://localhost:8000/api/v1/videos/search/image`
  - **Method**: POST
  - **Body**: `{"image_base64": "...", "object_filters": null}`
- **Tải ảnh keyframes**:
  - **URL**: `http://localhost:8000/api/v1/videos/keyframes/{key_tương_đối}`
  - **Method**: GET (Tự động redirect tới Pre-signed URL trên MinIO)

## Mock Server

Mock server cung cấp:
- **25 kết quả mẫu** với ảnh từ Picsum Photos
- **500ms delay** để simulate API processing
- **CORS enabled** cho frontend
- **Health check endpoint** tại `/health`

### Test Mock Server
Mở file `test-mock.html` để test API endpoints:
```bash
# Mở trong trình duyệt
frontend/test-mock.html
```

## Tính năng chi tiết

### Input Section
- Textarea cho phép nhập nhiều câu query
- Mỗi dòng = 1 câu query
- Validation: yêu cầu ít nhất 1 câu query

### Results Grid
- Grid 5x4 (20 items per page)
- Mỗi item hiển thị:
  - Ảnh từ jpg_path
  - original_id
  - similarity_score (phần trăm)
  - frame_idx
- Hover effects và animations

### Pagination
- Hiển thị thông tin: "Hiển thị 1-20 của 50 kết quả"
- Nút Trước/Tiếp để điều hướng
- Tự động disable nút khi ở trang đầu/cuối

### Loading & Error Handling
- Loading spinner khi đang tìm kiếm
- Error messages cho các lỗi
- Disable buttons khi đang loading

## Keyboard Shortcuts

- `Ctrl + Enter`: Tìm kiếm (khi focus vào textarea)

## Responsive Design

- **Desktop**: Grid 5x4
- **Tablet**: Grid 2x10
- **Mobile**: Grid 1x20

## Customization

### Thay đổi API URL
Trong `script.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000'; // Thay đổi URL ở đây
```

### Thay đổi Grid Size
Trong `script.js`:
```javascript
const GRID_SIZE = 20; // Thay đổi số lượng items per page
```

Trong `styles.css`:
```css
.grid-container {
    grid-template-columns: repeat(5, 1fr); /* Thay đổi số cột */
}
```

## Troubleshooting

### CORS Error
Nếu gặp CORS error, đảm bảo backend đã cấu hình CORS đúng.

### Ảnh không hiển thị
- Kiểm tra trường `image_url` trong API response
- Đảm bảo cụm dịch vụ MinIO (`docker-compose up -d`) đã hoạt động và dữ liệu ảnh đã được upload thành công qua `python scripts/upload_to_minio.py`

### API không response
- Kiểm tra backend có đang chạy không
- Kiểm tra URL API trong `script.js`
- Kiểm tra console browser để xem lỗi
