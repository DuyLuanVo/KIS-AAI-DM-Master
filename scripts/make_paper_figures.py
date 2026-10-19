"""
Sinh các hình phần dữ liệu và kiến trúc cho bài báo (bổ sung cho make_finetune_report.py).

  reports/clip_finetune/hinh0_pipeline.png     sơ đồ pipeline: thu thập -> caption -> fine-tune -> triển khai
  reports/clip_finetune/hinh5_du_lieu.png      (a) độ tương đồng caption Qwen/Gemma, (b) độ dài caption,
                                               (c) loại ảnh

    python scripts/make_paper_figures.py
"""
import csv
import ntpath
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_finetune_report import CM, C_FT, C_PRE, INK, INK2, MUTED, panel, save, style  # noqa: E402

OUT = ROOT / "reports" / "clip_finetune"
CMP = ROOT / "scripts" / "compare_output"
W = 16.5 * CM  # vùng chữ của mẫu bài báo (A4, lề trái 2,5 cm, phải 2 cm)


def pipeline():
    fig, ax = plt.subplots(figsize=(W, 6.2 * CM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    stages = [  # (x, y, tiêu đề, chi tiết, màu viền)
        (1, 23, "Thu thập", "Wikimedia Commons\nPetScan, độ sâu 3\n7.846 ảnh + metadata", INK2),
        (21, 23, "Làm sạch", "cạnh ngắn ≥ 224 px\ntỉ lệ ≤ 2.5; pHash ≤ 4\n7.762 ảnh sạch", INK2),
        (41, 23, "Sinh caption", "Qwen3.6 & Gemma 4\n15–40 từ, tiếng Anh\nchỉ tả điều nhìn thấy", INK2),
        (61, 23, "Đối chiếu chéo", "cosine 2 caption\nngưỡng 0.60\n861 ảnh xem thủ công", INK2),
        (81, 23, "Bộ dữ liệu", "7.756 cặp\nảnh – caption\ncaptions_final.csv", INK2),
        (81, 3, "Fine-tune CLIP", "ViT-B/32, 4 block cuối\nInfoNCE đối xứng\nchia 80/10/10", C_FT),
        (61, 3, "Lập chỉ mục", "Qdrant: vector ảnh,\nvector caption\nMinIO / thư mục ảnh", INK2),
        (41, 3, "Truy vấn lai", "BM25, ngữ nghĩa,\nCLIP văn bản → ảnh\ntrộn bằng RRF", C_FT),
        (21, 3, "API FastAPI", "tìm bằng mô tả / ảnh\nảnh tương đồng\nnạp ảnh mới", INK2),
        (1, 3, "Giao diện web", "tìm kiếm, xem chi tiết\nthêm ảnh vào kho\nhoàn tác lần nạp", INK2),
    ]
    bw, bh = 18, 14
    for x, y, title, body, ec in stages:
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="white", ec=ec, lw=1.0 if ec == INK2 else 1.6))
        ax.text(x + bw / 2, y + bh - 2.4, title, ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=C_FT if ec == C_FT else INK)
        ax.text(x + bw / 2, y + 5.2, body, ha="center", va="center", fontsize=6.2, color=INK2, linespacing=1.35)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2,
                                     shrinkA=0, shrinkB=0))

    for i in range(4):  # hàng trên: trái -> phải
        arrow((1 + 20 * i + bw, 30), (1 + 20 * (i + 1), 30))
    arrow((90, 23), (90, 17))  # xuống
    for i in range(4):  # hàng dưới: phải -> trái
        x0 = 81 - 20 * i
        arrow((x0, 10), (x0 - 20 + bw, 10))
    ax.text(50, 39.2, "Khai thác dữ liệu (offline)", ha="center", va="center", fontsize=7, color=MUTED,
            style="italic")
    ax.text(50, 0.4, "Huấn luyện và triển khai", ha="center", va="center", fontsize=7, color=MUTED, style="italic")
    save(fig, OUT, "hinh0_pipeline")


def data_stats():
    sims, final, types = [], [], Counter()
    with open(CMP / "similarity_all.csv", encoding="utf-8-sig", newline="") as f:
        sims = [float(r["similarity"]) for r in csv.DictReader(f)]
    with open(CMP / "captions_final.csv", encoding="utf-8-sig", newline="") as f:
        final = [len(r["caption"].split()) for r in csv.DictReader(f)]
    keep = set()
    with open(CMP / "captions_final.csv", encoding="utf-8-sig", newline="") as f:
        keep = {ntpath.basename(r["filepath"]) for r in csv.DictReader(f)}
    with open(ROOT / "scripts" / "caption_project_Qwen3.6" / "captions_full.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["file"] in keep:
                types[(r.get("image_type") or "other").strip() or "other"] += 1
    sims = np.array(sims)

    fig, axes = plt.subplots(1, 3, figsize=(W, 5.4 * CM), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    a, b, c = axes
    bins = np.linspace(0.15, 1.0, 35)
    a.hist(sims[sims >= 0.60], bins=bins, color=C_FT, ec="white", lw=0.4, label="tự động lấy Qwen")
    a.hist(sims[sims < 0.60], bins=bins, color=C_PRE, ec="white", lw=0.4, hatch="////", label="kiểm tra thủ công")
    a.set_xlim(0.15, 1.0)
    a.set_ylim(0, 1000)
    a.axvline(0.60, color=INK, lw=0.8, ls="--")
    a.text(0.585, 990, "ngưỡng\n0.60", ha="right", va="top", fontsize=6.5, color=INK)
    a.axvline(0.37, color=MUTED, lw=0.8, ls=":")
    a.text(0.355, 990, "trung vị\n2 ảnh\nngẫu\nnhiên", ha="right", va="top", fontsize=6, color=MUTED)
    a.set_xlabel("Cosine giữa caption Qwen và Gemma")
    a.set_ylabel("Số ảnh")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=6, handlelength=1.4, handletextpad=0.4,
             frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    panel(a, "a")

    b.hist(final, bins=np.arange(14.5, 41.5, 1), color=C_FT, ec="white", lw=0.4)
    b.axvline(np.mean(final), color=INK, lw=0.8, ls="--")
    b.text(np.mean(final) + 0.6, b.get_ylim()[1] * 0.97, f"TB {np.mean(final):.1f} từ", ha="left", va="top",
           fontsize=6.5, color=INK)
    b.set_xlabel("Độ dài caption (từ)")
    b.set_ylabel("Số ảnh")
    panel(b, "b")

    names = {"exterior": "ngoại thất", "detail": "chi tiết", "interior": "nội thất", "statue": "tượng",
             "drawing_or_map": "bản vẽ / bản đồ", "other": "khác"}
    items = [(names.get(k, k), v) for k, v in types.most_common() if k in names]
    labels, vals = zip(*items[::-1])
    c.barh(labels, vals, color=C_FT, height=0.62)
    for i, v in enumerate(vals):
        c.text(v + max(vals) * 0.02, i, f"{v:,}".replace(",", "."), va="center", fontsize=6.5, color=INK2)
    c.set_xlim(0, max(vals) * 1.25)
    c.grid(axis="y", visible=False)
    c.set_xlabel("Số ảnh")
    panel(c, "c")
    fig.tight_layout(w_pad=1.2)
    save(fig, OUT, "hinh5_du_lieu")
    print(f"  tương đồng: n={len(sims)} trung vị={np.median(sims):.3f} <0.60: {(sims < 0.60).sum()}")
    print(f"  loại ảnh: {dict(types)}")


if __name__ == "__main__":
    style()
    pipeline()
    data_stats()
