"""
Khử trùng ảnh bằng CLIP + ORB: mỗi bức ảnh / mỗi góc chụp của một cảnh chỉ giữ lại 1 ảnh.

Quy tắc cho mỗi cặp ảnh (i, j), chỉ cần thỏa một dòng:
  1. cosine embedding CLIP gốc >= --dup-thr (0,97)                 -> bản sao (cắt, chỉnh màu, nén lại)
  2. cosine trong [--verify-lo, --dup-thr) = [0,90; 0,97) và
     ORB + RANSAC homography có >= --min-inliers (40) điểm khớp      -> cùng cảnh, cùng góc chụp
CLIP ở dòng 2 chỉ dùng để chọn cặp ứng viên; quyết định nằm ở kiểm chứng hình học, nên ngưỡng 0,90 không làm tăng
gom nhầm (tỉ lệ cặp được ORB xác nhận giảm từ ~69% ở cosine 0,95-0,97 xuống ~4% ở 0,90-0,91).

Các cặp trùng được gom thành cụm bằng union-find (A trùng B, B trùng C -> cả ba chung cụm); mỗi cụm giữ ảnh có độ
phân giải lớn nhất (caption dài hơn làm tiêu chí phụ), bỏ phần còn lại. Sau bước này không còn ảnh trùng nên
finetune_clip.py chia train/val/test ngẫu nhiên theo từng ảnh (mỗi ảnh giữ lại là một nhóm riêng).

Chỉ dùng nội dung ảnh (không dùng caption hay nhãn) nên chạy trên toàn bộ dữ liệu trước khi chia không gây rò rỉ.

Đầu ra (mặc định data/vietnam_architecture/dedup/)
  * dedup_groups.csv   : mỗi ảnh một dòng: keep, dup_cluster, dup_of, reason, split_group, group_size
  * dedup_pairs.csv    : các cặp ứng viên kèm cosine, số điểm khớp ORB, kết luận
  * summary.json       : thống kê
  * pairs_<band>.jpg   : ảnh ghép mẫu theo từng vùng để kiểm tra / chỉnh ngưỡng bằng mắt
  * features.npz       : cache embedding CLIP + kích thước ảnh (chạy lại để chỉnh ngưỡng thì không phải tính lại)

Ví dụ:
    python scripts/dedup_images.py
    python scripts/dedup_images.py --dup-thr 0.96 --min-inliers 50   # chỉnh ngưỡng, dùng lại cache
"""
import argparse
import csv
import json
import ntpath
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_preprocess import open_rgb  # noqa: E402


# ---------------------------------------------------------------- đọc ảnh
def list_images(args):
    image_dir = Path(args.image_dir)
    names = []
    if args.captions and Path(args.captions).exists():
        with open(args.captions, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                fp = (r.get("filepath") or r.get("file") or "").strip()
                if fp:
                    names.append(ntpath.basename(fp))
    else:
        names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    names = sorted({n for n in names if (image_dir / n).exists()})
    return image_dir, names


# ---------------------------------------------------------------- đặc trưng: embedding CLIP + kích thước
def compute_features(image_dir: Path, names, device, batch_size=64):
    import clip
    import torch

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    n = len(names)
    sizes = np.zeros((n, 2), dtype=np.int32)
    emb = np.zeros((n, 512), dtype=np.float32)
    batch, bidx, t0 = [], [], time.time()

    def flush():
        with torch.no_grad():
            e = model.encode_image(torch.stack(batch).to(device)).float()
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        emb[bidx] = e
        batch.clear()
        bidx.clear()

    for i, name in enumerate(names):
        im = open_rgb(image_dir / name)
        sizes[i] = im.size
        batch.append(preprocess(im))
        bidx.append(i)
        if len(batch) == batch_size:
            flush()
            print(f"\r  embedding {i + 1}/{n} ({(i + 1) / (time.time() - t0):.1f} ảnh/s)", end="", flush=True)
    if batch:
        flush()
    print()
    return {"names": np.array(names), "sizes": sizes, "emb": emb}


def load_or_compute(args, image_dir, names):
    cache = Path(args.out_dir) / "features.npz"
    if cache.exists() and not args.recompute:
        f = dict(np.load(cache, allow_pickle=False))
        if list(f["names"]) == names and "emb" in f and "sizes" in f:
            print(f"Dùng cache đặc trưng: {cache}")
            return f
    print(f"Tính embedding CLIP cho {len(names)} ảnh...")
    f = compute_features(image_dir, names, args.device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **f)
    return f


# ---------------------------------------------------------------- kiểm chứng hình học
_orb_cache = {}


def orb_features(path):
    import cv2

    if path not in _orb_cache:
        g = np.asarray(open_rgb(path).convert("L"))
        orb = cv2.ORB_create(nfeatures=1500)
        _orb_cache[path] = orb.detectAndCompute(g, None)
        if len(_orb_cache) > 4000:
            _orb_cache.pop(next(iter(_orb_cache)))
    return _orb_cache[path]


def geometric_inliers(p1, p2) -> int:
    """Số cặp điểm ORB khớp nhau qua một phép biến đổi homography (RANSAC). Lớn -> cùng một cảnh/cùng góc chụp."""
    import cv2

    k1, d1 = orb_features(p1)
    k2, d2 = orb_features(p2)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for m in matcher.knnMatch(d1, d2, k=2):
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:  # ratio test của Lowe
            good.append(m[0])
    if len(good) < 10:
        return 0
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


# ---------------------------------------------------------------- union-find
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


# ---------------------------------------------------------------- tìm cặp trùng
def find_pairs(f, args, image_dir):
    names, emb = list(f["names"]), f["emb"]
    n = len(names)
    pairs = {}
    bs = 1024
    for s in range(0, n, bs):
        e = slice(s, min(s + bs, n))
        sim = emb[e] @ emb.T
        rows = np.arange(e.start, e.stop)
        for a, b in zip(*np.nonzero(sim >= args.verify_lo)):
            i, j = int(rows[a]), int(b)
            if i < j:
                pairs[(i, j)] = {"cos": float(sim[a, b])}
    print(f"Cặp ứng viên (cosine >= {args.verify_lo}): {len(pairs)}")

    n_verify = sum(1 for p in pairs.values() if p["cos"] < args.dup_thr)
    done, t0 = 0, time.time()
    for (i, j), p in pairs.items():
        p["inliers"] = -1
        if p["cos"] >= args.dup_thr:
            p["reason"] = "clip"
        else:
            p["inliers"] = geometric_inliers(str(image_dir / names[i]), str(image_dir / names[j]))
            p["reason"] = "orb" if p["inliers"] >= args.min_inliers else ""
            done += 1
            if done % 200 == 0:
                print(f"\r  kiểm chứng ORB {done}/{n_verify} ({done / (time.time() - t0):.1f} cặp/s)", end="", flush=True)
        p["dup"] = bool(p["reason"])
    if n_verify:
        print()
    return pairs


# ---------------------------------------------------------------- caption dài hơn làm tiêu chí phụ khi chọn ảnh giữ lại
def caption_lengths(captions_csv):
    out = {}
    if captions_csv and Path(captions_csv).exists():
        with open(captions_csv, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                fp = (r.get("filepath") or r.get("file") or "").strip()
                if fp:
                    out[ntpath.basename(fp)] = len((r.get("caption") or "").split())
    return out


def build_clusters(f, pairs, cap_len):
    names = list(f["names"])
    n = len(names)
    dsu = DSU(n)
    for (i, j), p in pairs.items():
        if p["dup"]:
            dsu.union(i, j)
    area = f["sizes"][:, 0].astype(np.int64) * f["sizes"][:, 1]
    clusters = {}
    for i in range(n):
        clusters.setdefault(dsu.find(i), []).append(i)
    keep, dup_of = np.ones(n, bool), [""] * n
    for members in clusters.values():
        if len(members) < 2:
            continue
        best = max(members, key=lambda k: (area[k], cap_len.get(names[k], 0), -k))
        for k in members:
            if k != best:
                keep[k], dup_of[k] = False, names[best]
    reason = [""] * n
    for (i, j), p in pairs.items():
        if p["dup"]:
            for k in (i, j):
                if not keep[k] and not reason[k]:
                    reason[k] = p["reason"]
    root = [dsu.find(i) for i in range(n)]
    size = Counter(root)
    return keep, dup_of, reason, root, size


# ---------------------------------------------------------------- ảnh ghép kiểm tra bằng mắt
def contact_sheet(pairs, names, image_dir, path, title, max_pairs=12, size=220):
    if not pairs:
        return
    pairs = pairs[:max_pairs]
    cols, pad, head = 2, 8, 34
    w = cols * (2 * size + 3 * pad)
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (w, head + rows * (size + 2 * pad + 18)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), title, fill="black")
    for k, ((i, j), p) in enumerate(pairs):
        x0 = (k % cols) * (2 * size + 3 * pad) + pad
        y0 = head + (k // cols) * (size + 2 * pad + 18)
        for t, idx in enumerate((i, j)):
            im = open_rgb(image_dir / names[idx])
            im.thumbnail((size, size))
            sheet.paste(im, (x0 + t * (size + pad), y0))
        label = f"cos {p['cos']:.3f} inl {p['inliers']} -> {p['reason'] or 'khac'}"
        d.text((x0, y0 + size + 3), label, fill="black")
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(description="Khử trùng ảnh bằng CLIP + ORB, mỗi cụm giữ 1 ảnh")
    ap.add_argument("--captions", default=str(ROOT / "scripts" / "compare_output" / "captions_final.csv"),
                    help="Chỉ xét ảnh có trong file caption (để trống -> mọi ảnh trong --image-dir)")
    ap.add_argument("--image-dir", default=str(ROOT / "data" / "vietnam_architecture"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "vietnam_architecture" / "dedup"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recompute", action="store_true", help="Tính lại embedding thay vì dùng cache")
    ap.add_argument("--dup-thr", type=float, default=0.97, help="cosine CLIP >= ngưỡng -> bản sao")
    ap.add_argument("--verify-lo", type=float, default=0.90,
                    help="cosine trong [verify-lo, dup-thr) -> kiểm chứng ORB+RANSAC")
    ap.add_argument("--min-inliers", type=int, default=40, help="số điểm khớp ORB tối thiểu để coi là cùng góc chụp")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_dir, names = list_images(args)
    print(f"Số ảnh: {len(names)}")
    f = load_or_compute(args, image_dir, names)
    pairs = find_pairs(f, args, image_dir)
    keep, dup_of, reason, root, size = build_clusters(f, pairs, caption_lengths(args.captions))

    # split_group = cụm trùng; sau khi bỏ ảnh trùng, mỗi ảnh giữ lại là một nhóm riêng -> chia ngẫu nhiên theo ảnh
    with open(out / "dedup_groups.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_file", "keep", "dup_cluster", "dup_of", "reason", "split_group", "group_size"])
        for i, nm in enumerate(names):
            w.writerow([nm, int(keep[i]), names[root[i]], dup_of[i], reason[i], names[root[i]], size[root[i]]])
    with open(out / "dedup_pairs.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "cos", "inliers", "dup", "reason"])
        for (i, j), p in sorted(pairs.items(), key=lambda kv: -kv[1]["cos"]):
            w.writerow([names[i], names[j], f"{p['cos']:.4f}", p["inliers"], int(p["dup"]), p["reason"]])

    by_reason = Counter(p["reason"] for p in pairs.values() if p["dup"])
    removed_by = Counter(r for r, k in zip(reason, keep) if not k)
    multi = [s for s in size.values() if s > 1]
    summary = {
        "n_images": len(names), "n_candidate_pairs": len(pairs),
        "n_dup_pairs": sum(p["dup"] for p in pairs.values()), "dup_pairs_by_rule": dict(by_reason),
        "n_removed": int((~keep).sum()), "removed_by_rule": dict(removed_by), "n_kept": int(keep.sum()),
        "n_clusters_gt1": len(multi), "largest_cluster": max(size.values()) if size else 0,
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ảnh ghép kiểm tra: bản sao, ORB xác nhận, ORB sát ngưỡng, vùng xám bị từ chối
    items = sorted(pairs.items(), key=lambda kv: -kv[1]["cos"])
    bands = [("a_clip_dup", lambda p: p["reason"] == "clip"),
             ("b_orb_dup", lambda p: p["reason"] == "orb"),
             ("c_orb_near_thr", lambda p: p["reason"] == "orb" and p["inliers"] < args.min_inliers + 20),
             ("d_rejected", lambda p: not p["dup"])]
    rng = np.random.default_rng(0)
    for tag, cond in bands:
        sel = [kv for kv in items if cond(kv[1])]
        if len(sel) > 12:
            sel = [sel[k] for k in sorted(rng.choice(len(sel), 12, replace=False))]
        contact_sheet(sel, names, image_dir, out / f"pairs_{tag}.jpg", f"{tag}: {len(sel)} cap mau")

    print("\n--- Kết quả ---")
    print(f"  Cặp trùng: {summary['n_dup_pairs']}  theo quy tắc: {dict(by_reason)}")
    print(f"  Bỏ khỏi dữ liệu: {summary['n_removed']} ảnh {dict(removed_by)}, còn {summary['n_kept']}")
    print(f"  Cụm có >1 ảnh: {summary['n_clusters_gt1']}, lớn nhất {summary['largest_cluster']}")
    print(f"  Đầu ra: {out}")


if __name__ == "__main__":
    main()
