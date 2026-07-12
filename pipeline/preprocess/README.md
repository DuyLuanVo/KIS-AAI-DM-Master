# 🎬 Công cụ Tiền xử lý Video Mới (Video Preprocessing Pipeline)

Thư mục này chứa công cụ tự động hóa toàn bộ quy trình tiền xử lý một video mới từ dạng thô (`.mp4`, `.avi`,...) thành các định dạng đặc trưng (`.npy` CLIP, `.csv` metadata, `.jpg` keyframe, `.json` placeholder objects) để sẵn sàng nạp vào Qdrant Vector Database.

---

## 🛠️ Cài đặt dependencies (Thư viện cần thiết)

Để chạy được kịch bản tiền xử lý này, bạn cần cài đặt các thư viện trong file `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Lưu ý: Nếu chưa cài đặt PyTorch, hãy tham khảo trang chủ [pytorch.org](https://pytorch.org/) để cài đặt phiên bản phù hợp nhất với thiết bị của bạn (hỗ trợ GPU/CUDA sẽ giúp tăng tốc độ trích xuất CLIP).*

---

## 🚀 Hướng dẫn sử dụng

Chạy trực tiếp file `preprocess_video.py` bằng dòng lệnh:

```bash
python preprocess_video.py --video_path <đường_dẫn_video> --video_id <ID_video> --batch <tên_batch>
```

### Các tham số cấu hình:

| Tham số | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `--video_path` | **Có** | - | Đường dẫn đến file video gốc (ví dụ: `new_movie.mp4`). |
| `--video_id` | Không | Tên file video | Tên ID định danh cho video trong database (ví dụ: `L21_V099`). |
| `--batch` | Không | `L21` | Tên Batch thư mục keyframes (ví dụ: `L21` hoặc `L22`). |
| `--sample_rate`| Không | `2.0` | Khoảng cách trích xuất (giây/khung hình). `2.0` nghĩa là cứ 2 giây cắt 1 ảnh. |
| `--data_root` | Không | Tự động nhận diện | Đường dẫn đến thư mục `data` tổng. Nếu không nhập, script sẽ lấy từ `config.py` ở thư mục gốc hoặc mặc định là `../data`. |

### Ví dụ cụ thể:

Giả sử bạn có file video `my_video.mp4` nằm ở Desktop và muốn đưa vào batch `L21`, chạy lệnh:

```bash
python preprocess_video.py --video_path "C:\Users\DLV\Desktop\my_video.mp4" --video_id "L21_V099" --batch "L21"
```

---

## 📂 Các dữ liệu đầu ra được tạo thành

Sau khi chạy xong, script sẽ tự động tạo và lưu trữ các file vào đúng vị trí của cấu trúc thư mục dữ liệu tổng (`data_root`):

1. **Khung hình Keyframes:** Các ảnh dạng `.jpg` (như `001.jpg`, `002.jpg`...) được lưu tại thư mục:
   `data/Keyframes_L21/keyframes/L21_V099/`
2. **Metadata CSV:** File ánh xạ thông tin thời gian thực xuất hiện (`pts_time`), index gốc (`frame_idx`), fps của video được lưu tại:
   `data/map-keyframes-aic25-b1/map-keyframes/L21_V099.csv`
3. **Đặc trưng CLIP (.npy):** Vector đặc trưng 512 chiều của tất cả keyframe được gộp chung lại và lưu tại:
   `data/clip-features-32-aic25-b1/clip-features-32/L21_V099.npy`
4. **Placeholder Object Detection (.json):** Tạo các file JSON trống tương ứng cho từng keyframe trong thư mục:
   `data/objects-aic25-b1/objects/L21_V099/` (Tránh việc script loader bị lỗi do thiếu cấu trúc file vật thể).

---

## ⏭️ Bước tiếp theo

Khi tiền xử lý hoàn tất thành công, bạn chỉ cần thực hiện bước cuối để nạp video mới này lên Qdrant Vector Database:

1. Chạy file `run_loader.py` ở thư mục gốc project (điều chỉnh vòng lặp xử lý video trong file đó nếu cần thiết).
2. Tận hưởng việc tìm kiếm các phân cảnh trong video mới từ giao diện Frontend!
