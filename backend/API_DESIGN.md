# Video Search API Design

## Tổng quan

API này được thiết kế để tìm kiếm video frames. Kết quả trả về bao gồm `original_id`, `jpg_path`, và `frame_idx` như yêu cầu.

## API Endpoints

### POST `/api/v1/videos/search`

**Mô tả:** Tìm kiếm video frames dựa trên text query

**Request Body:**
```json
{
  "query_text": [
    "a person walking",
    "a person with red hat is walking"
  ],
  "list_object": [
    "PERSON",
    "DOG"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "original_id": "video_001",
      "jpg_path": "/data/frames/video_001/frame_000123.jpg",
      "frame_idx": 123,
      "similarity_score": 0.95
    },
    {
      "original_id": "video_002",
      "jpg_path": "/data/frames/video_002/frame_000045.jpg", 
      "frame_idx": 45,
            "similarity_score": 0.95
    },
    {
      "original_id": "video_001",
      "jpg_path": "/data/frames/video_001/frame_000156.jpg",
      "frame_idx": 156,
            "similarity_score": 0.95
    }
  ]
}
```

## Data Models

### VideoSearchRequest
```python
class VideoSearchRequest(BaseModel):
    query_text: List[str]                  
```

### VideoSearchResult
```python
class VideoSearchResult(BaseModel):
    original_id: str                    # ID của video gốc
    jpg_path: str                       # Đường dẫn file JPG
    frame_idx: int                      # Index của frame
    similarity_score: float

```

### VideoSearchResponse
```python
class VideoSearchResponse(BaseModel):
    results: List[VideoSearchResult]    # Danh sách kết quả
```

## Tính năng chính

### 1. Text-based Search
- Tìm kiếm video frames dựa trên list text query
- Nhận input là câu text mô tả
- Trả về danh sách kết quả phù hợp

### 2. Structured Response
- Kết quả có cấu trúc rõ ràng
- Bao gồm original_id, jpg_path, frame_idx
- Dễ dàng xử lý ở frontend

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Usage Examples

### Basic Search
```bash
curl -X POST "http://localhost:8000/api/v1/videos/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "a person walking"
  }'
```

### Search with Different Queries
```bash
# Search for cars
curl -X POST "http://localhost:8000/api/v1/videos/search" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "car driving"}'

# Search for animals
curl -X POST "http://localhost:8000/api/v1/videos/search" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "dog running"}'
```

## Implementation Notes

### TODO Items (cần implement):
1. **Text-to-Vector Conversion**: Convert text query thành vector embedding
2. **Vector Search Logic**: Implement similarity search với Qdrant
3. **Text Processing**: Preprocess text query (lowercase, tokenization, etc.)
4. **Performance Optimization**: Optimize search performance
5. **Error Handling**: Cải thiện error handling

### Workflow:
1. Nhận text query từ user
2. Convert text thành vector embedding (có thể dùng CLIP, BERT, etc.)
3. Search vector trong Qdrant database
4. Return kết quả với original_id, jpg_path, frame_idx

## Testing

### Test Cases:
1. **Basic Search**: Test endpoint với text query
2. **Response Structure**: Test cấu trúc response đúng
3. **Data Validation**: Test data types của response
4. **Empty Query**: Test với empty text
5. **Long Query**: Test với long text query
