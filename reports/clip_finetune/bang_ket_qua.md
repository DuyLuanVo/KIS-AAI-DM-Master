# Kết quả fine-tune CLIP

### Bảng 1. Dữ liệu và cấu hình fine-tune CLIP

| Hạng mục | Giá trị |
|---|---|
| Mô hình nền | CLIP ViT-B/32 (OpenAI) |
| Dữ liệu | 6.285 cặp ảnh–caption |
| Chia train / val / test | 5027 / 629 / 629 (theo nhóm ảnh trùng) |
| Block được train (mỗi nhánh) | 4 / 12 |
| Tham số được train | 41.6M / 151.3M (27.5%) |
| Hàm mất mát | InfoNCE đối xứng |
| Tối ưu | AdamW, lr 2e-05, weight decay 0.1, warmup 10% + cosine |
| Batch / số epoch tối đa | 256 / 20 (dừng sớm, patience 4) |
| Epoch được chọn (theo rsum trên val) | 19 |
| Phần cứng | GPU, ~1.6 phút/epoch |

### Bảng 2. Kết quả trên tập test (629 ảnh)

| Tác vụ | Chỉ số | CLIP gốc | CLIP fine-tuned | Chênh lệch |
|---|---|---:|---:|---:|
| Văn bản → ảnh | R@1 | 51.67 | **82.83** | +31.16 |
|  | R@5 | 80.76 | **97.46** | +16.69 |
|  | R@10 | 88.24 | **99.36** | +11.13 |
|  | MedR ↓ | 1.00 | 1.00 | +0.00 |
| Ảnh → văn bản | R@1 | 59.94 | **82.67** | +22.73 |
|  | R@5 | 81.88 | **97.77** | +15.90 |
|  | R@10 | 89.67 | **99.36** | +9.70 |
| Văn bản → ảnh (toàn kho) | R@1 | 26.23 | **52.31** | +26.07 |
|  | R@10 | 59.94 | **91.41** | +31.48 |
|  | R@50 | 81.56 | **98.25** | +16.69 |
| Ảnh → ảnh | P@10 | 45.71 | **50.89** | +5.18 |
|  | mAP@10 | 40.48 | **46.15** | +5.67 |
| Tổng R@K hai chiều | rsum | 452.15 | **559.46** | +107.31 |

R@K, P@10, mAP@10 tính theo %; MedR (hạng trung vị của ảnh đúng) càng nhỏ càng tốt. "Tập test": tìm trong 629 ảnh test; "toàn kho": tìm trong cả 6.285 ảnh. Ảnh → ảnh: ảnh liên quan là ảnh có chung category Wikimedia Commons cụ thể (không dùng khi train).

### Bảng 3. Chỉ số trên tập val theo epoch (epoch 0 = CLIP gốc)

| Epoch | Loss | V→A R@1 | V→A R@10 | A→V R@1 | A→V R@10 | A→A P@10 | rsum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | – | 53.3 | 88.4 | 58.2 | 90.3 | 44.5 | 453.3 |
| 1 | 3.7058 | 64.7 | 94.0 | 68.0 | 96.8 | 45.8 | 505.2 |
| 2 | 1.7504 | 75.7 | 97.6 | 76.2 | 97.9 | 48.7 | 537.5 |
| 3 | 1.1267 | 78.5 | 98.1 | 80.8 | 98.4 | 49.1 | 549.6 |
| 4 | 0.8540 | 82.0 | 98.4 | 80.8 | 98.7 | 49.6 | 554.1 |
| 5 | 0.7018 | 82.2 | 99.0 | 82.8 | 99.0 | 49.7 | 558.8 |
| 6 | 0.6126 | 82.7 | 99.5 | 84.9 | 99.0 | 50.1 | 562.3 |
| 7 | 0.5346 | 82.7 | 99.4 | 82.8 | 99.2 | 50.0 | 559.6 |
| 8 | 0.5073 | 83.1 | 99.7 | 85.1 | 99.5 | 50.2 | 564.2 |
| 9 | 0.4528 | 83.8 | 99.4 | 83.6 | 99.5 | 50.1 | 563.0 |
| 10 | 0.4188 | 84.9 | 99.5 | 84.3 | 99.5 | 50.5 | 564.7 |
| 11 | 0.3769 | 84.7 | 99.8 | 83.8 | 99.7 | 50.4 | 564.7 |
| 12 | 0.3651 | 84.6 | 99.8 | 83.8 | 99.5 | 50.5 | 564.1 |
| 13 | 0.3520 | 85.4 | 99.8 | 84.9 | 99.5 | 50.5 | 566.1 |
| 14 | 0.3384 | 84.9 | 99.8 | 84.4 | 99.5 | 50.5 | 565.2 |
| 15 | 0.3327 | 85.2 | 99.8 | 84.4 | 99.7 | 50.6 | 565.8 |
| 16 | 0.3086 | 86.3 | 99.8 | 84.6 | 99.5 | 50.6 | 567.2 |
| 17 | 0.2956 | 86.0 | 99.8 | 85.5 | 99.5 | 50.6 | 568.0 |
| 18 | 0.3106 | 86.0 | 99.8 | 85.7 | 99.5 | 50.7 | 568.2 |
| 19 | 0.3045 | 86.0 | 99.8 | 85.9 | 99.5 | 50.7 | **568.4** |
| 20 | 0.2997 | 86.0 | 99.8 | 85.7 | 99.5 | 50.7 | 568.2 |

V→A: văn bản → ảnh, A→V: ảnh → văn bản, A→A: ảnh → ảnh. In đậm: epoch được chọn.

### Bảng 4. Tìm ảnh tương đồng theo công trình trên tập test (P@10, %)

| Công trình (category Commons) | Số truy vấn | CLIP gốc | CLIP fine-tuned | Chênh lệch |
|---|---:|---:|---:|---:|
| Gates in Vietnam | 22 | 30.5 | 35.9 | +5.5 |
| Ho Chi Minh Mausoleum | 21 | 81.9 | 83.3 | +1.4 |
| Gardens in Vietnam | 16 | 55.6 | 55.6 | +0.0 |
| Roofs in Vietnam | 15 | 32.0 | 34.7 | +2.7 |
| Ornamental plants in Vietnam | 15 | 72.0 | 68.0 | -4.0 |
| Cemeteries in Vietnam | 15 | 79.3 | 85.3 | +6.0 |
| Qianci in Vietnam | 14 | 78.6 | 80.0 | +1.4 |
| Stairs in Vietnam | 13 | 32.3 | 40.8 | +8.5 |
| Tomb of Emperor Khải Định | 13 | 72.3 | 75.4 | +3.1 |
| Linh Phuoc Pagoda | 12 | 70.0 | 71.7 | +1.7 |
| My Son | 12 | 40.0 | 48.3 | +8.3 |
| Phu Quoc Prison | 11 | 77.3 | 76.4 | -0.9 |
