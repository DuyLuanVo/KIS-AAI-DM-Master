# BẢN THẢO TÓM TẮT ĐỀ TÀI ĐỒ ÁN

### 1. Tên đề tài
**"Ứng dụng Trí tuệ Nhân tạo trong truy xuất ngữ nghĩa hình ảnh: Xây dựng hệ thống quản lý và tìm kiếm kho tư liệu Kiến trúc Việt Nam."**
*(Applied AI in Semantic Image Retrieval: Building an archive management and search system for Vietnamese Architecture photography).*

### 2. Đặt vấn đề (Problem Statement)
Trong ngành kiến trúc và bảo tồn di tích, việc nghiên cứu và tham khảo các phong cách kiến trúc (từ truyền thống như đình đền, tháp Chăm, nhà rường đến kiến trúc thuộc địa Đông Dương hay kiến trúc nhiệt đới hiện đại) đóng vai trò cực kỳ quan trọng. Các văn phòng kiến trúc, viện khoa học và kiến trúc sư thường lưu trữ hàng Terabyte dữ liệu ảnh công trình, nội/ngoại thất và bản vẽ phối cảnh. Tuy nhiên, các phương pháp quản lý truyền thống theo tên file hoặc chia thư mục thủ công khiến cho việc tìm kiếm lại một thông tin cụ thể (ví dụ: *"Mặt tiền nhà phố với lam chắn nắng và cây xanh"* hoặc *"Hệ mái trần gỗ truyền thống dưới ánh nắng"*) trở nên vô cùng mất thời gian. Các công cụ tìm kiếm hiện tại không hiểu được ngôn ngữ về chất liệu, ánh sáng, hình khối và không gian kiến trúc.

### 3. Mục tiêu đề tài (Project Objectives)
Đề tài hướng tới việc xây dựng một hệ thống **Truy xuất hình ảnh dựa trên nội dung và ngữ nghĩa (Content-Based Image Retrieval - CBIR)** dành riêng cho lĩnh vực kiến trúc. Hệ thống giúp loại bỏ hoàn toàn công đoạn gán thẻ (tagging) thủ công, cho phép các kiến trúc sư và nhà nghiên cứu truy vấn bằng ngôn ngữ tự nhiên miêu tả hình khối, chất liệu, phong cách kiến trúc với độ chính xác cao.

**Mục tiêu cụ thể:**
- Xây dựng cổng nạp dữ liệu (Ingestion Pipeline) tự động phân tích và mã hóa hình ảnh các công trình kiến trúc.
- Áp dụng mô hình AI đa phương thức (Multimodal AI) để nhận dạng sâu sắc các yếu tố thị giác, bố cục và không gian.
- Xây dựng giao diện web thân thiện cho phép người dùng tra cứu nhanh chóng và lọc kết quả thông minh (Hybrid Search).

### 4. Phạm vi và Giới hạn đề tài (Scope)
Để tối ưu hóa hiệu năng, độ chính xác và tính thực thi, đề tài đặt ra các giới hạn cụ thể sau:
- **Giới hạn Dữ liệu:** Chỉ xử lý dữ liệu **Hình ảnh tĩnh (Images)** (ảnh chụp công trình thực tế, phối cảnh 3D nội/ngoại thất), không xử lý Video hay file BIM/CAD thuần tuý.
- **Giới hạn Lĩnh vực (Domain):** Tập trung vào dữ liệu **Kiến trúc Việt Nam**, chia làm các dải công trình: Kiến trúc cổ/thờ tự, Kiến trúc thuộc địa/Đông Dương, Kiến trúc nhà ở và công trình công cộng hiện đại (khoảng 5.000 - 10.000 hình ảnh chất lượng cao).
- **Giới hạn Ngôn ngữ:** Hệ thống ưu tiên xử lý câu lệnh tìm kiếm bằng Tiếng Anh để tối ưu hóa khả năng biểu diễn vector của mô hình AI lõi (Ví dụ: *"Modern tropical house facade with wooden louvers and plants"*).

### 5. Công nghệ áp dụng (Technologies Used)
Hệ thống kết hợp các kiến trúc phần mềm hiện đại và các mô hình Trí tuệ nhân tạo tiên tiến nhất:
- **Mô hình CLIP (OpenAI):** Mã hóa hình ảnh công trình và văn bản miêu tả thành các Vector 512 chiều, có khả năng thấu hiểu vượt trội về phong cách kiến trúc (minimalist, tropical, colonial), vật liệu (concrete, wood, glass) và ánh sáng (daylight, shadow, twilight).
- **Mô hình YOLOv8:** Nhận diện và trích xuất nhãn các đối tượng trong không gian (bàn, ghế, đèn, cây xanh, người...) đóng vai trò làm bộ lọc tiền xử lý phụ trợ.
- **Qdrant Vector Database:** Cơ sở dữ liệu chuyên biệt để lưu trữ và truy xuất siêu nhan sắc vector.
- **MinIO (Object Storage):** Hệ thống lưu trữ file ảnh gốc tương thích chuẩn S3.
- **FastAPI & JavaScript/React:** Xây dựng máy chủ xử lý hiệu năng cao và giao diện người dùng nhạy bén.

### 6. Đóng góp và Ý nghĩa thực tiễn (Practical Contributions)
- **Về mặt công nghệ:** Giải quyết thành công bài toán "khoảng cách ngữ nghĩa" trong mảng kiến trúc bằng sự kết hợp giữa tìm kiếm không gian vector (CLIP) và lọc thông tin đối tượng cấu thành không gian (YOLO).
- **Về mặt ứng dụng:** Đưa ra giải pháp thực tế cho các Viện nghiên cứu kiến trúc, Bảo tàng di sản và các Văn phòng thiết kế, giúp tiết kiệm hàng ngàn giờ quản lý và khai thác tư liệu, thúc đẩy hiệu suất sáng tạo trong ngành thiết kế kiến trúc Việt Nam.
