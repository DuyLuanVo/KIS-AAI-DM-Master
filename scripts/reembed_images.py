"""
Tính lại vector "image" của mọi ảnh trong Qdrant bằng mô hình CLIP đang cấu hình (CLIP gốc hoặc bản fine-tune).

Chạy sau khi đổi CLIP_FINETUNED_PATH / CLIP_MODEL. Chỉ thay vector ảnh: caption, vector caption, metadata,
kết quả YOLO và ảnh người dùng tải lên đều giữ nguyên (khác với index_dataset.py --recreate).
Ảnh đã có vector của đúng mô hình hiện tại (payload "image_model") được bỏ qua, nên chạy lại được nếu bị ngắt.

Ví dụ:
    python scripts/reembed_images.py           # tính lại những ảnh chưa dùng mô hình hiện tại
    python scripts/reembed_images.py --force   # tính lại toàn bộ
"""
import argparse
import io
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
from app.services.embedders import get_image_embedder  # noqa: E402
from app.services.storage import get_storage  # noqa: E402


def load_image(key: str, storage):
    # Ảnh dataset đọc thẳng từ thư mục gốc cho nhanh; không có thì hỏi kho lưu trữ (MinIO / media/)
    if key.startswith("dataset/"):
        p = Path(settings.dataset_dir) / key.split("/", 1)[1]
        if p.is_file():
            return Image.open(p).convert("RGB")
    data = storage.get(key)
    return Image.open(io.BytesIO(data)).convert("RGB") if data else None


def main():
    ap = argparse.ArgumentParser(description="Tính lại vector ảnh trong Qdrant bằng CLIP hiện tại")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="Tính lại cả ảnh đã dùng mô hình hiện tại")
    args = ap.parse_args()

    store, storage, emb = get_store(), get_storage(), get_image_embedder()
    if not store.client.collection_exists(store.collection):
        sys.exit("Chưa có collection, hãy chạy python scripts/index_dataset.py trước")
    print(f"Mô hình ảnh: {emb.name} | collection: {store.collection}")

    points = store._scroll(None, with_payload=["key", "image_model"])
    todo = [p.payload["key"] for p in points if args.force or (p.payload or {}).get("image_model") != emb.name]
    print(f"Tổng {len(points)} ảnh, cần tính lại {len(todo)}")

    done, failed, t0 = 0, [], time.time()
    for s in range(0, len(todo), args.batch_size):
        keys, images = [], []
        for key in todo[s: s + args.batch_size]:
            try:
                img = load_image(key, storage)
            except Exception:
                img = None
            if img is None:
                failed.append(key)
                continue
            keys.append(key)
            images.append(img)
        if images:
            vecs = emb.encode(images)
            done += store.update_image_vectors([{"key": k, "image_vector": v} for k, v in zip(keys, vecs)], emb.name)
        n = min(s + args.batch_size, len(todo))
        print(f"\r[{n}/{len(todo)}] cập nhật {done} | lỗi {len(failed)} | {done / max(time.time() - t0, 1e-6):.1f} ảnh/s",
              end="", flush=True)
    print(f"\nHoàn tất: cập nhật {done}, không đọc được {len(failed)} ảnh")
    for k in failed[:20]:
        print(f"  - {k}")
    store.client.close()


if __name__ == "__main__":
    main()
