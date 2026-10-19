# Kết quả fine-tune CLIP

### Bảng 1. Dữ liệu và cấu hình fine-tune CLIP

| Hạng mục | Giá trị |
|---|---|
| Mô hình nền | CLIP ViT-B/32 (OpenAI) |
| Dữ liệu | 7.756 cặp ảnh–caption |
| Chia train / val / test | 6204 / 776 / 776 (theo nhóm ảnh trùng) |
| Block được train (mỗi nhánh) | 4 / 12 |
| Tham số được train | 41.6M / 151.3M (27.5%) |
| Hàm mất mát | InfoNCE đối xứng |
| Tối ưu | AdamW, lr 2e-05, weight decay 0.1, warmup 10% + cosine |
| Batch / số epoch tối đa | 256 / 15 (dừng sớm, patience 4) |
| Epoch được chọn (theo rsum trên val) | 6 |
| Phần cứng | CPU, ~6.4 phút/epoch |

### Bảng 2. Kết quả trên tập test (776 ảnh)

| Tác vụ | Chỉ số | CLIP gốc | CLIP fine-tuned | Chênh lệch |
|---|---|---:|---:|---:|
| Văn bản → ảnh | R@1 | 47.16 | **76.80** | +29.64 |
|  | R@5 | 76.93 | **96.52** | +19.59 |
|  | R@10 | 83.51 | **98.84** | +15.34 |
|  | MedR ↓ | 2.00 | **1.00** | -1.00 |
| Ảnh → văn bản | R@1 | 51.42 | **79.64** | +28.22 |
|  | R@5 | 77.71 | **97.29** | +19.59 |
|  | R@10 | 85.82 | **98.58** | +12.76 |
| Văn bản → ảnh (toàn kho) | R@1 | 19.59 | **40.21** | +20.62 |
|  | R@10 | 51.93 | **85.70** | +33.76 |
|  | R@50 | 77.19 | **96.65** | +19.46 |
| Ảnh → ảnh | P@10 | 53.49 | **57.51** | +4.03 |
|  | mAP@10 | 49.07 | **53.46** | +4.38 |
| Tổng R@K hai chiều | rsum | 422.55 | **547.68** | +125.13 |

R@K, P@10, mAP@10 tính theo %; MedR (hạng trung vị của ảnh đúng) càng nhỏ càng tốt. "Tập test": tìm trong 776 ảnh test; "toàn kho": tìm trong cả 7.756 ảnh. Ảnh → ảnh: ảnh liên quan là ảnh có chung category Wikimedia Commons cụ thể (không dùng khi train).

### Bảng 3. Chỉ số trên tập val theo epoch (epoch 0 = CLIP gốc)

| Epoch | Loss | V→A R@1 | V→A R@10 | A→V R@1 | A→V R@10 | A→A P@10 | rsum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | – | 48.6 | 85.2 | 53.4 | 85.4 | 52.4 | 429.6 |
| 1 | 0.8645 | 68.0 | 97.4 | 69.2 | 97.8 | 56.3 | 520.2 |
| 2 | 0.3319 | 75.1 | 98.3 | 73.1 | 98.7 | 56.9 | 538.0 |
| 3 | 0.1451 | 76.3 | 98.6 | 76.2 | 98.3 | 56.5 | 541.8 |
| 4 | 0.1069 | 78.0 | 98.7 | 75.6 | 98.6 | 57.0 | 543.9 |
| 5 | 0.0840 | 77.1 | 98.8 | 76.4 | 99.0 | 57.0 | 545.2 |
| 6 | 0.0639 | 78.4 | 99.0 | 76.8 | 99.1 | 56.9 | **546.5** |
| 7 | 0.0653 | 75.6 | 99.0 | 77.1 | 98.3 | 56.7 | 543.0 |
| 8 | 0.0507 | 77.2 | 98.6 | 77.8 | 99.0 | 56.7 | 546.0 |
| 9 | 0.0440 | 76.5 | 99.0 | 76.7 | 98.8 | 56.6 | 544.7 |
| 10 | 0.0365 | 77.3 | 99.4 | 76.4 | 99.0 | 56.8 | 545.4 |

V→A: văn bản → ảnh, A→V: ảnh → văn bản, A→A: ảnh → ảnh. In đậm: epoch được chọn.

### Bảng 4. Tìm ảnh tương đồng theo công trình trên tập test (P@10, %)

| Công trình (category Commons) | Số truy vấn | CLIP gốc | CLIP fine-tuned | Chênh lệch |
|---|---:|---:|---:|---:|
| Gardens in Vietnam | 27 | 85.2 | 85.2 | +0.0 |
| Gates in Vietnam | 27 | 32.2 | 40.4 | +8.1 |
| Tomb of Emperor Khải Định | 26 | 75.8 | 79.6 | +3.8 |
| Linh Phuoc Pagoda | 26 | 65.4 | 77.7 | +12.3 |
| Ho Chi Minh Mausoleum | 24 | 86.7 | 90.0 | +3.3 |
| Cemeteries in Vietnam | 18 | 65.6 | 68.9 | +3.3 |
| Buddhist temple architecture of Vietnam | 17 | 59.4 | 63.5 | +4.1 |
| Ornamental plants in Vietnam | 17 | 68.8 | 67.6 | -1.2 |
| Swimming pools in Vietnam | 16 | 98.1 | 95.6 | -2.5 |
| My Son | 16 | 65.0 | 65.6 | +0.6 |
| Stairs in Vietnam | 16 | 48.8 | 58.8 | +10.0 |
| Stupas in Vietnam | 15 | 48.7 | 60.7 | +12.0 |
