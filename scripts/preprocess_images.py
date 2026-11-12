"""
Bước 2 của pipeline - tiền xử lý ảnh (chạy sau download_commons_images.py, trước dedup_images.py):

  1. Chuẩn hóa: xoay theo EXIF, ghép nền trắng cho ảnh trong suốt, chuyển RGB, thu cạnh dài về 640 px
     (ghi đè PNG tại chỗ; ảnh đã chuẩn hóa thì giữ nguyên).
  2. Lọc kích thước: cạnh ngắn >= 224 px, tỉ lệ cạnh <= 2,5.
  3. Lọc bản sao y hệt: pHash lệch <= 4 bit. Duyệt ảnh theo độ phân giải giảm dần (hòa thì pageid nhỏ trước),
     ảnh khớp với một ảnh đã giữ thì bị loại -> mỗi nhóm bản sao giữ ảnh phân giải lớn nhất.

Đầu ra: data/vietnam_architecture/quality/quality_report.csv
        (image_file, width, height, aspect, phash, flags, dup_of, keep); dedup_images.py chỉ dùng ảnh keep=1.

    python scripts/preprocess_images.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from image_preprocess import open_rgb  # noqa: E402

TARGET_SIZE = 640       # cạnh dài sau chuẩn hóa
MIN_SHORT_SIDE = 224    # cạnh ngắn tối thiểu (= đầu vào CLIP)
MAX_ASPECT = 2.5        # tỉ lệ cạnh dài / cạnh ngắn tối đa
DUP_DISTANCE = 4        # số bit pHash lệch tối đa để coi là bản sao y hệt


def normalize(path: Path):
    """Chuẩn hóa ảnh tại chỗ; trả về (w, h). Ảnh đã đúng dạng (RGB, không xoay EXIF, cạnh dài <= 640) thì không ghi lại."""
    with Image.open(path) as raw:
        orient = raw.getexif().get(0x0112, 1)
        need = raw.mode != "RGB" or orient != 1 or max(raw.size) > TARGET_SIZE
        size = raw.size
    if not need:
        return size
    im = open_rgb(path)
    im.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    im.save(path, "PNG", optimize=True)
    return im.size


def main():
    ap = argparse.ArgumentParser(description="Tiền xử lý ảnh: chuẩn hóa, lọc kích thước/tỉ lệ, lọc bản sao pHash")
    ap.add_argument("--image-dir", default=str(ROOT / "data" / "vietnam_architecture"))
    ap.add_argument("--out", default=str(ROOT / "data" / "vietnam_architecture" / "quality" / "quality_report.csv"))
    args = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    image_dir = Path(args.image_dir)
    recs = []
    for meta in sorted(image_dir.glob("*.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        path = image_dir / m.get("image_file", "")
        if not path.is_file():
            continue
        r = {"image_file": path.name, "pageid": int(m.get("pageid") or 0), "flags": [], "dup_of": ""}
        try:
            w, h = normalize(path)
            with Image.open(path) as im:
                r["phash"] = str(imagehash.phash(im))
        except Exception as e:
            r.update(width="", height="", aspect="", phash="")
            r["flags"].append(f"loi_doc_file ({e.__class__.__name__})")
            recs.append(r)
            continue
        r.update(width=w, height=h, aspect=round(max(w, h) / min(w, h), 2))
        if min(w, h) < MIN_SHORT_SIDE:
            r["flags"].append("qua_nho")
        if r["aspect"] > MAX_ASPECT:
            r["flags"].append("ti_le_dai")
        recs.append(r)
        if len(recs) % 1000 == 0:
            print(f"  ... {len(recs)} ảnh", flush=True)

    # Bản sao y hệt: chỉ xét ảnh đạt kích thước; giữ ảnh phân giải lớn nhất (hòa: pageid nhỏ hơn)
    cand = sorted((r for r in recs if not r["flags"]), key=lambda r: (-r["width"] * r["height"], r["pageid"]))
    kept = []
    for r in cand:
        hv = int(r["phash"], 16)
        for k, kv in kept:
            if bin(hv ^ kv).count("1") <= DUP_DISTANCE:
                r["flags"].append("trung_lap")
                r["dup_of"] = k["image_file"]
                break
        else:
            kept.append((r, hv))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_file", "width", "height", "aspect", "phash", "flags", "dup_of", "keep"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in sorted(recs, key=lambda r: r["image_file"]):
            wr.writerow({**{k: r.get(k, "") for k in fields[:-3]}, "flags": ";".join(r["flags"]),
                         "dup_of": r["dup_of"], "keep": int(not r["flags"])})
    n = lambda flag: sum(flag in r["flags"] for r in recs)
    print(f"Ảnh: {len(recs)} | cạnh ngắn < {MIN_SHORT_SIDE}: {n('qua_nho')} | tỉ lệ > {MAX_ASPECT}: {n('ti_le_dai')} | "
          f"bản sao pHash: {n('trung_lap')} | lỗi đọc: {sum(any(x.startswith('loi') for x in r['flags']) for r in recs)} | "
          f"bị loại: {sum(bool(r['flags']) for r in recs)} | còn: {sum(not r['flags'] for r in recs)}")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
