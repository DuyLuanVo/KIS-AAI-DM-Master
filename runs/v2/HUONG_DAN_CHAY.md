# Chạy lại pipeline v2 trên hai máy song song

Pipeline v2: khử trùng bằng CLIP + ORB (mỗi cụm trùng giữ 1 ảnh), chia ngẫu nhiên theo ảnh 80/10/10,
train cấu hình E (drop-path 0,1, không mặt nạ nhóm).

Hai máy dùng **chung** các tệp đã commit trong thư mục này, nên dữ liệu và cách chia giống hệt nhau:

| Tệp | Nội dung |
|---|---|
| `runs/v2/dedup_groups.csv` | Kết quả khử trùng: 7.756 ảnh, bỏ 1.471, còn 6.285 |
| `runs/v2/split.json` | Train 5.027 / val 629 / test 629 |
| `runs/v2/summary.json` | Thống kê khử trùng |

| Máy | Việc | Kết quả |
|---|---|---|
| Máy 1 (máy hiện tại) | Cấu hình chính (`--aug-policy default`) | `models/clip_finetuned_v2` |
| Máy 2 | Thí nghiệm B (`--aug-policy blip_geo`), sau đó các fold 5-fold được giao | **Tất cả trong `runs/v2/may2/`** |

Thí nghiệm B = tăng cường hiện tại + RandAugment(2, 5) kiểu BLIP chỉ gồm phép hình học
(ShearX/Y 0,15, TranslateX/Y 10%, Rotate 15°, mỗi phép p = 0,5), không tăng đổi màu.

Máy 2 **không cần push**. Chạy xong, đưa nguyên thư mục `runs/v2/may2/` cho máy 1 (USB / ổ chia sẻ / nén zip).

---

## Máy 2: lần đầu cài đặt

Yêu cầu: Python 3.10, GPU NVIDIA có CUDA, Git. Trên máy đã có thư mục ảnh `vietnam_architecture`
(7.756 ảnh `.png` kèm `.json` metadata).

```bash
git clone https://github.com/DuyLuanVo/KIS-AAI-DM-Master.git
cd KIS-AAI-DM-Master
git checkout pipeline-v2

# đặt thư mục ảnh vào đúng chỗ: KIS-AAI-DM-Master/data/vietnam_architecture/<ảnh>.png
#   Windows (PowerShell), tạo junction thay vì chép:
#     New-Item -ItemType Directory -Force data
#     New-Item -ItemType Junction -Path data\vietnam_architecture -Target "D:\duong\dan\toi\vietnam_architecture"
#   Linux/macOS:
#     mkdir -p data && ln -s /duong/dan/toi/vietnam_architecture data/vietnam_architecture

python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.12.1+cu126 torchvision==0.27.1+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Kiểm tra nhanh, phải in `True` và khoảng `7756 ảnh`:

```bash
python -c "import torch, clip, cv2; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import pathlib; p=pathlib.Path('data/vietnam_architecture'); print(len(list(p.glob('*.png'))), 'ảnh')"
```

## Máy 2, bước 1: thí nghiệm B (khoảng 1 giờ trên GPU 4 GB)

```bash
git pull
mkdir -p runs/v2/may2
python scripts/finetune_clip.py \
    --dedup runs/v2/dedup_groups.csv --split-file runs/v2/split.json \
    --aug-policy blip_geo \
    --device cuda --augment --grad-checkpoint --workers 8 \
    --out-dir runs/v2/may2/clip_finetuned_v2_B  2>&1 | tee runs/v2/may2/clip_finetuned_v2_B.log
```

PowerShell không có `tee` kiểu bash, dùng: `... 2>&1 | Tee-Object -FilePath runs/v2/may2/clip_finetuned_v2_B.log`
(và viết lệnh trên một dòng, bỏ các dấu `\`).

Tùy GPU:

* GPU ≥ 8 GB: có thể bỏ `--grad-checkpoint` cho nhanh hơn.
* GPU có Tensor Core (RTX 20xx trở lên): thêm `--amp`. GPU GTX 16xx / T1200 thì **không** thêm (chậm hơn).
* `--workers`: khoảng số nhân CPU trừ 2.

Đầu log phải có `Chép split từ runs/v2/split.json`, `train 5027 | val 629 | test 629`
và `Tăng cường trực tuyến [blip_geo]`.

## Máy 2, bước 2: các fold 5-fold được giao

Chỉ chạy sau khi máy 1 báo đã chọn cấu hình (`<POLICY>` = `default` hoặc `blip_geo`).
Cách chia fold tính từ `runs/v2/dedup_groups.csv` và hạt giống 42 nên giống nhau trên hai máy.

```bash
python scripts/kfold_clip.py --dedup runs/v2/dedup_groups.csv --out-dir runs/v2/may2/kfold_v2 --folds 3,4 \
    --prefix-cache runs/v2/may2/clip_finetuned_v2_B/cache/prefix_ViT-B-32_k4.pt \
    -- --device cuda --augment --grad-checkpoint --workers 8 --aug-policy <POLICY>
```

## Máy 2: đưa kết quả cho máy 1

Đưa nguyên thư mục `runs/v2/may2/`. Trước khi nén có thể xóa các thư mục `cache/` bên trong
(khoảng 1,2 GB mỗi thư mục, máy 1 tự tính lại được). Cần giữ:

* `clip_finetuned_v2_B/`: `metrics.json`, `split.json`, `clip_vitb32_kis.pt` (khoảng 166 MB)
* `clip_finetuned_v2_B.log`
* `kfold_v2/fold3/`, `kfold_v2/fold4/` (nếu đã chạy bước 2): `metrics.json`, `split.json`, `train.log`

---

## Máy 1: cấu hình chính

```bash
python scripts/finetune_clip.py \
    --dedup runs/v2/dedup_groups.csv --split-file runs/v2/split.json \
    --aug-policy default \
    --device cuda --augment --grad-checkpoint --workers 12 \
    --out-dir models/clip_finetuned_v2  > models/clip_finetuned_v2.log 2>&1
```

## Máy 1: chọn cấu hình theo val

Sau khi nhận thư mục `runs/v2/may2/`:

```bash
python scripts/compare_runs.py models/clip_finetuned_v2 runs/v2/may2/clip_finetuned_v2_B
```

Cấu hình có `val rsum` cao hơn được chọn. Số liệu test chỉ để báo cáo.

## Máy 1: 5-fold (fold 0, 1, 2) và tổng hợp

```bash
python scripts/kfold_clip.py --dedup runs/v2/dedup_groups.csv --out-dir models/kfold_v2 --folds 0,1,2 \
    --prefix-cache models/clip_finetuned_v2/cache/prefix_ViT-B-32_k4.pt \
    -- --device cuda --augment --grad-checkpoint --workers 12 --aug-policy <POLICY>

# gộp fold 3, 4 của máy 2 rồi tổng hợp
cp -r runs/v2/may2/kfold_v2/fold3 runs/v2/may2/kfold_v2/fold4 models/kfold_v2/
python scripts/kfold_clip.py --out-dir models/kfold_v2 --summary-only
```
