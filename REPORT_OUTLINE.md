# ĐỀ CƯƠNG BÁO CÁO ĐỒ ÁN / LUẬN VĂN
**(Dựa trên Tóm tắt đề tài: Quản lý và tìm kiếm kho tư liệu Kiến trúc Việt Nam)**

Dưới đây là cấu trúc chi tiết các chương mục bạn cần viết trong cuốn báo cáo (file Word/PDF) để nộp cho Hội đồng.

---

## CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI (INTRODUCTION)
**1.1 Đặt vấn đề**
- Bối cảnh: Ngành kiến trúc Việt Nam có sự phát triển sôi động với sự hòa lẫn từ kiến trúc truyền thống, thuộc địa Đông Dương (Indochine) đến kiến trúc nhiệt đới hiện đại. Khối lượng ảnh tư liệu công trình, phối cảnh 3D và nội/ngoại thất tại các văn phòng thiết kế ngày càng khổng lồ.
- Bất cập: Các cách lưu trữ truyền thống (chia thư mục theo năm/dự án hoặc đặt tên file) khiến việc tìm kiếm theo phong cách, vật liệu, ánh sáng, hình khối gặp hạn chế nghiêm trọng; việc con người tự gán nhãn mác (tagging) cho từng chi tiết không gian là điều bất khả thi.
- Nhu cầu thực tế: Cần một hệ thống tìm kiếm thông minh có khả năng "thấu hiểu" ngôn ngữ tạo hình và kiến trúc, giúp quay lại cảm hứng sáng tạo chỉ sau vài giây.

**1.2 Mục tiêu đề tài**
- Xây dựng hệ thống Truy xuất hình ảnh dựa trên nội dung (CBIR) chuyên sâu cho tư liệu kiến trúc và không gian nội/ngoại thất.
- Tự động hóa quá trình lập chỉ mục ngữ nghĩa và nhận biết thành phần nội thất.
- Cung cấp giao diện tra cứu nhanh chóng, phản hồi tức thì với độ chính xác cao.

**1.3 Phạm vi và Giới hạn**
- Về dữ liệu: Hình ảnh tĩnh (ảnh chụp công trình, hình chụp nội ngoại thất, phối cảnh 3D độ sắc nét cao).
- Về lĩnh vực: Tập trung vào hệ sinh thái **Kiến trúc Việt Nam** (nhà ở công sở hiện đại, nhà phố nhiệt đới có cây xanh, công trình di tích truyền thống, kiến trúc kiểu Đông Dương).
- Về ngôn ngữ truy vấn: Tiếng Anh (nhằm tối ưu hóa sức mạnh của AI với các bộ từ chép thuộc ngữ kiến trúc chuẩn quốc tế).

**1.4 Ý nghĩa và Đóng góp của đề tài**
- Ý nghĩa khoa học: Triển khai hiệu quả kỹ thuật Multimodal Embeddings vào phân tích ngôn ngữ thị giác kiến trúc (chất liệu, ánh sáng, cấu tạo).
- Ý nghĩa thực tiễn: Giúp các Kiến trúc sư, nhà thiết kế nội thất và ban bảo tồn di sản tiết kiệm hàng vạn giờ tìm kiếm và tối ưu hoá quá trình lên ý tưởng thiết kế (conceptualizing).

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT (THEORETICAL BACKGROUND)
**2.1 Bài toán Tìm kiếm Hình ảnh dựa trên Nội dung (CBIR trong Kiến trúc)**
- CBIR là gì? Sự khác biệt giữa Text-based Image Retrieval và CBIR khi phân tích hình học, chất liệu và màu sắc không gian.
- Thách thức: Khoảng cách ngữ nghĩa (Semantic Gap) - rào cản từ pixel ảnh máy tính hiểu sang phong cách kiến trúc của con người (Modernism, Tropical, Indochine, Minimalist).

**2.2 Mô hình CLIP (Contrastive Language-Image Pre-Training)**
- Giới thiệu về kiến trúc Transformer trong mô hình CLIP của OpenAI.
- Cơ chế ánh xạ Song song: Text Encoder và Image Encoder trong cùng một không gian Vector 512 chiều.
- Khả năng vượt trội trong Kiến trúc: Nhận diện cực tốt ánh sáng (ambient light, direct daylight), vật liệu thô (bê tông trần, gỗ mộc, gạch gốm), và các đường nét kiến trúc lặp lại.
- Điểm yếu và thách thức: Đôi khi thiếu tập trung vào từng đối tượng đồ nội thất nhỏ rải rác nếu bị nhầm lẫn trong không gian tổng quan rộng lớn.

**2.3 Mô hình YOLO (You Only Look Once)**
- Giới thiệu về mạng nơ-ron phát hiện vật thể YOLOv8.
- Ứng dụng: Nhận diện tức thì các thành phần cấu tạo không gian (bàn ăn, ghế sofa, giường, chậu cây, đèn, con người sinh hoạt...) để đóng vai trò làm Lớp bộ lọc Cứng (Hard pre-filter), giúp làm trọn vẹn điểm yếu của CLIP.

**2.4 Cơ sở dữ liệu Vector (Vector Database - Qdrant)**
- Kiến trúc lưu trữ chuyên dụng cho vector ngữ nghĩa và metadata cấu trúc.
- Thuật toán tìm kiếm lân cận gần nhất theo hình thái học không gian (HNSW - Hierarchical Navigable Small World) và tính toán độ đo Tương tự Cosine (Cosine Similarity).

---

## CHƯƠNG 3: THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG (SYSTEM ARCHITECTURE)
**3.1 Sơ đồ Khối và Luồng hệ thống (System Flow)**
- Kiến trúc tổng thể: Client - API Gateway (FastAPI) - Processing Worker (CLIP/YOLO) - Data Stores (Qdrant & MinIO).
- Luồng nạp dữ liệu (Ingestion Pipeline): Upload tệp/thư mục Ảnh -> Hồ sơ Object Storage (MinIO) -> Quét vật thể (YOLOv8) -> Mã hóa không gian ngữ nghĩa (CLIP) -> Lập chỉ mục Vector (Qdrant).
- Luồng tìm kiếm (Hybrid Search Pipeline): Người dùng nhập miêu tả (Text) / tải ảnh tương đồng -> Mã hóa câu truy vấn -> Qdrant thực hiện lọc tiền xử lý theo Metadata vật thể -> Tìm kiếm lân cận Vector -> Bổ trợ trả về danh sách ảnh minh họa chuẩn tay nghề.

**3.2 Thiết kế Cơ sở dữ liệu**
- Vector DB (Qdrant): Quản lý Vector 512-D, lưu trữ Payload (original_id, jpg_path, danh sách objects nội thất đã phát hiện, thể loại kiến trúc cơ sở).
- Storage (MinIO): Cấu trúc bucket quản lý lưu trữ an toàn các tệp tin ảnh độ phân giải cao gốc của công trình.

**3.3 Công nghệ và Công cụ phát triển**
- Backend: FastAPI (Python 3.10+).
- AI Serving: PyTorch, OpenCV, CLIP, Ultralytics YOLO.
- Frontend: Vanilla JavaScript / HTML5 / CSS3 hiện đại, giao diện trực quan responsive.
- Đóng gói và Quản lý triển khai: Docker & Docker Compose.

---

## CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ (EXPERIMENTS & EVALUATION)
**4.1 Tập dữ liệu thử nghiệm (Dataset Kiến trúc Việt Nam)**
- Phương pháp thu thập: Khai thác từ các tạp chí kiến trúc uy tín (ArchDaily Vietnam, Tạp chí Kiến trúc, Behance), ảnh chụp công trình và di tích Việt Nam chất lượng cao.
- Phân bố Dataset: Khoảng 5.000 tấm ảnh chia đều cho 3 cụm (Kiến trúc cổ điển & Di tích truyền thống, Nhà ở đô thị & Nhiệt đới hiện đại, Nội thất công trình công cộng/văn phòng).

**4.2 Kịch bản thử nghiệm (Test Cases & Scenarios)**
- Kịch bản 1 - Nhận biết Phong cách và Chất liệu (CLIP): Tìm *"Modern concrete tropical facade with wooden louvers and greenery"* (Mặt tiền bê tông hiện đại nhiệt đới với lam bạt gỗ và cây leo).
- Kịch bản 2 - Nhận biết Ánh sáng & Thời khắc (CLIP): Tìm *"Traditional Vietnamese Buddhist temple courtyard under golden sunset light"* (Sân đền chùa Việt Nam dưới ánh chiều tàng vinh).
- Kịch bản 3 - Tìm kiếm Lai Không gian Nội thất (Hybrid Search CLIP + YOLO): Tìm kiếm cụm từ *"Minimalist bright dining area"* (Khu vực ăn uống tối giản tươi sáng) kết hợp bộ lọc vật thể YOLO: **[chair] + [dining table] + [potted plant]** để đảm bảo luôn xuất hiện bàn ghế ăn và cây cảnh thảm hái.

**4.3 Đánh giá kết quả thực tế**
- Thời gian trễ trung bình của truy vấn (Response latency ~ vài trăm mili-giây).
- Đánh giá cảm quan và độ chuẩn xác (Precision@K) thông qua hội đồng kiến trúc sư, sinh viên ngành thử nghiệm trực diện.

---

## CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
**5.1 Kết quả đạt được**
- Giải quyết triệt để bài toán tìm kiếm tư liệu ngành kiến trúc tại thị trường Việt Nam.
- Kiến trúc tìm kiếm lai (Hybrid Search) hoạt động ổn định, mượt mà và cực kỳ trực quan.

**5.2 Hạn chế của hệ thống**
- AI chưa phân biệt sâu được các quy chuẩn kích thước hình học tuyệt đối (milimet) hay tỷ lệ vàng kỹ thuật như con người.
- Hỗ trợ tốt nhất khi truy xuất thông qua tiếng Anh do đặc tính tập huấn luyện nguyên bản của CLIP.

**5.3 Hướng phát triển tương lai**
- Nghiên cứu Fine-tune (huấn luyện bổ sung) bộ Trọng số cho CLIP với kho từ vựng kiến trúc truyền thống Đông Nam Á và Việt Nam.
- Mở rộng hỗ trợ nạp bản vẽ sơ đồ mặt bằng (Floor Plans) bên cạnh ảnh chụp phối cảnh tự nhiên.
- Tích hợp thêm mô hình dịch thuật tự động (LLM Translator) để tra cứu Tiếng Việt mượt mà ngay trên thanh công cụ.

---
*(HẾT)*
