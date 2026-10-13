"""
Lập chỉ mục bộ dữ liệu ảnh + caption vào Qdrant (và MinIO nếu dùng).

Đầu vào: scripts/compare_output/captions_final.csv (filepath, caption) do compare_captions.py --merge sinh ra.
Ảnh được tìm theo TÊN FILE trong data/vietnam_architecture/ (đường dẫn trong CSV có thể là của máy khác),
metadata đi kèm đọc từ file .json cùng tên (do download_commons_images.py tạo).

Ví dụ:
    python scripts/index_dataset.py                 # nạp toàn bộ, bỏ qua ảnh đã có
    python scripts/index_dataset.py --limit 200     # thử nhanh 200 ảnh
    python scripts/index_dataset.py --recreate      # xóa collection cũ, nạp lại từ đầu
    python scripts/index_dataset.py --yolo          # kèm nhận diện cấu kiện YOLO-World (chậm)
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.database.vector_store import get_store  # noqa: E402
from app.services.indexer import build_payload, clean_caption, ensure_collection, index_items  # noqa: E402
from app.services.storage import get_storage  # noqa: E402


def read_captions(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fp = (row.get("filepath") or row.get("file") or "").strip()
            cap = clean_caption(row.get("caption"))
            if fp and cap:
                yield re.split(r"[\\/]", fp)[-1], cap, fp


def load_meta(img_path: Path) -> dict:
    j = img_path.with_suffix(".json")
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def main():
    ap = argparse.ArgumentParser(description="Lập chỉ mục ảnh + caption vào Qdrant")
    ap.add_argument("--captions", default=str(ROOT / "scripts" / "compare_output" / "captions_final.csv"))
    ap.add_argument("--image-dir", default=settings.dataset_dir)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Chỉ nạp N ảnh đầu (0 = tất cả)")
    ap.add_argument("--recreate", action="store_true", help="Xóa và tạo lại collection")
    ap.add_argument("--no-skip", action="store_true", help="Nạp lại cả ảnh đã có trong collection")
    ap.add_argument("--yolo", action="store_true", help="Nhận diện cấu kiện bằng YOLO-World")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    rows = list(read_captions(Path(args.captions)))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Caption: {len(rows)} dòng từ {args.captions}")
    print(f"Thư mục ảnh: {image_dir}")
    print(f"Embedder: {settings.embedder_backend} | Qdrant: {settings.qdrant_mode} | Lưu ảnh: {settings.storage_backend}")

    ensure_collection(recreate=args.recreate)
    store, storage = get_store(), get_storage()

    yolo = None
    if args.yolo:
        import cv2
        import numpy as np
        from app.services.yolo_service import yolo_service as yolo
        yolo.initialize()

    missing, skipped, done = [], 0, 0
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start: start + args.batch_size]
        keys = [f"dataset/{name}" for name, _, _ in chunk]
        existing = set() if (args.no_skip or args.recreate) else store.existing_keys(keys)

        images, payloads = [], []
        for (name, cap, orig), key in zip(chunk, keys):
            if key in existing:
                skipped += 1
                continue
            path = image_dir / name
            if not path.exists() and Path(orig).exists():
                path = Path(orig)
            if not path.exists():
                missing.append(name)
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"  [Lỗi đọc ảnh] {name}: {e}")
                missing.append(name)
                continue
            meta = load_meta(path)
            objects = []
            if yolo is not None:
                cats = meta.get("categories") or [meta.get("monument_name", "")]
                objects, _ = yolo.detect_objects(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), context_categories=cats)
            try:
                storage.put_file(key, path)
            except Exception as e:
                print(f"  [Lỗi lưu ảnh] {name}: {e}")
                continue
            images.append(img)
            payloads.append(build_payload(key=key, file_name=name, caption=cap, source="dataset", meta=meta,
                                          objects=objects, width=img.width, height=img.height))

        done += index_items(images, payloads)
        n = start + len(chunk)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"[{n}/{len(rows)}] đã nạp {done} | bỏ qua (đã có) {skipped} | thiếu ảnh {len(missing)} | {rate:.1f} ảnh/s",
              flush=True)

    print(f"\nHoàn tất: nạp {done}, bỏ qua {skipped}, thiếu {len(missing)}. Tổng trong collection: {store.count()}")
    if missing:
        out = ROOT / "scripts" / "missing_images.txt"
        out.write_text("\n".join(missing), encoding="utf-8")
        print(f"Danh sách ảnh không tìm thấy: {out}")
    store.client.close()


if __name__ == "__main__":
    main()
