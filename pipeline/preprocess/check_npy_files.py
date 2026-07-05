import os

import numpy as np

npy_folder = 'clip-features-32'  # Đổi thành đường dẫn thật

# Lấy tất cả file .npy trong folder (không tính file ẩn)
npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy') and not f.startswith('.')]

print(f"Số file npy: {len(npy_files)}\n")

total_vectors = 0

for fname in sorted(npy_files):
    npy_path = os.path.join(npy_folder, fname)
    try:
        features = np.load(npy_path, mmap_mode='r')
        shape = features.shape
        dtype = features.dtype
        num_vectors = shape[0]
        total_vectors += num_vectors
        print(f"{fname}: shape {shape}, dtype {dtype}, vectors: {num_vectors}")
    except Exception as e:
        print(f"{fname}: LỖI khi đọc file: {e}")

print("\n==== Tổng kết ====")
print(f"Tổng số file npy: {len(npy_files)}")
print(f"Tổng số vector (frame): {total_vectors}")
