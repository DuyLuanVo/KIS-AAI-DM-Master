"""
Sinh hình phần dữ liệu cho bài báo (bổ sung cho make_finetune_report.py).

  reports/clip_finetune/hinh5_du_lieu.png      (a) độ tương đồng caption Qwen/Gemma (toàn bộ ảnh),
                                               (b) độ dài caption, (c) loại ảnh (ảnh còn lại sau khử trùng)

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_finetune_report import CM, C_FT, C_PRE, INK, INK2, MUTED, panel, save, style  # noqa: E402

OUT = ROOT / "reports" / "clip_finetune"
CMP = ROOT / "scripts" / "compare_output"
DEDUP = ROOT / "data" / "vietnam_architecture" / "dedup" / "dedup_groups.csv"
W = 16.5 * CM  # vùng chữ của mẫu bài báo (A4, lề trái 2,5 cm, phải 2 cm)


def data_stats():
    sims, final, types = [], [], Counter()
    with open(CMP / "similarity_all.csv", encoding="utf-8-sig", newline="") as f:
        sims = [float(r["similarity"]) for r in csv.DictReader(f)]
    # (b), (c) tính trên bộ dữ liệu cuối: chỉ các ảnh còn lại sau khử trùng
    with open(DEDUP, encoding="utf-8-sig", newline="") as f:
        keep = {r["image_file"] for r in csv.DictReader(f) if r["keep"] == "1"}
    with open(CMP / "captions_final.csv", encoding="utf-8-sig", newline="") as f:
        final = [len(r["caption"].split()) for r in csv.DictReader(f) if ntpath.basename(r["filepath"]) in keep]
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
    print(f"  caption: n={len(final)} TB={np.mean(final):.2f} từ")
    print(f"  loại ảnh: {dict(types)}")


if __name__ == "__main__":
    style()
    data_stats()
