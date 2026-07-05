# Video Retrieval API Backend

FastAPI backend cho hệ thống video retrieval với Qdrant vector database.

## Cấu trúc dự án

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py              # Cấu hình ứng dụng
│   ├── api/
│   │   ├── __init__.py
│   │   ├── api.py             # API router
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── health.py      # Health check endpoints
│   │       └── vectors.py     # Vector operations endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── logging.py         # Logging configuration
│   ├── database/
│   │   ├── __init__.py
│   └── └── qdrant_client.py   # Qdrant client
│   └── models/
│       ├── __init__.py
│       └── schemas.py         # Pydantic models
├── requirements.txt           # Python dependencies
├── env.example               # Environment variables template
└── README.md                 # This file
```

## Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Cấu hình environment:**
```bash
cp env.example .env
# Chỉnh sửa .env theo nhu cầu
```

3. **Khởi chạy cụm dịch vụ Qdrant và MinIO:**
```bash
docker-compose up -d
```

## Chạy ứng dụng

### Development mode:
```bash
python -m app.main
```

### Production mode:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- `GET /health` - Health check tổng quát
- `GET /health/qdrant` - Health check Qdrant

### Vector Operations
- `POST /api/v1/vectors/search` - Tìm kiếm vector tương tự
- `POST /api/v1/vectors/upload` - Upload vectors
- `DELETE /api/v1/vectors/delete` - Xóa vectors
- `GET /api/v1/vectors/collection-info` - Thông tin collection

### Documentation
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## Ví dụ sử dụng

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload Vectors
```bash
curl -X POST "http://localhost:8000/api/v1/vectors/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[0.1, 0.2, 0.3, ...]],
    "ids": ["vector_1"],
    "payloads": [{"metadata": "test"}]
  }'
```

### Search Vectors
```bash
curl -X POST "http://localhost:8000/api/v1/vectors/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [0.1, 0.2, 0.3, ...],
    "limit": 10,
    "score_threshold": 0.7
  }'
```

## Cấu hình

Các cấu hình chính trong `app/config.py`:

- `qdrant_host`: Host của Qdrant server
- `qdrant_port`: Port của Qdrant server
- `qdrant_collection`: Tên collection
- `minio_endpoint`: Endpoint của MinIO server (mặc định: `localhost:9000`)
- `minio_access_key`: Access Key của MinIO (mặc định: `minioadmin`)
- `minio_secret_key`: Secret Key của MinIO (mặc định: `minioadmin`)
- `minio_bucket`: Tên bucket chứa ảnh keyframes (mặc định: `kis-keyframes`)
- `minio_secure`: Sử dụng kết nối bảo mật HTTPS hay HTTP (mặc định: `False`)
- `qdrant_vector_size`: Kích thước vector (mặc định: 512)

## Logging

Ứng dụng sử dụng Loguru cho logging:
- Console output với màu sắc
- File logging trong production
- Rotation và compression tự động

## Development

### Code formatting:
```bash
black app/
isort app/
```

### Linting:
```bash
flake8 app/
```

### Testing:
```bash
pytest
```
