"""
Sinh hình và bảng kết quả fine-tune CLIP cho báo cáo.

Đọc checkpoint + cache do finetune_clip.py tạo ra (không train lại) và xuất vào reports/clip_finetune/:

  hinh1_qua_trinh_train.png     (a) loss train, (b) Recall@1 trên val theo epoch (epoch 0 = CLIP gốc)
  hinh2_ket_qua_test.png        so sánh CLIP gốc / fine-tuned trên tập test theo 4 nhóm chỉ số
  hinh3_phan_tich_xep_hang.png  (a) Recall@K text->ảnh trên toàn kho, (b) phân bố khoảng cách cosine
  hinh4_vi_du_truy_van.png      ví dụ truy vấn: top-5 ảnh của hai mô hình
  bang_ket_qua.md               Bảng 1-4: cấu hình, kết quả test, lịch sử train, theo công trình

Hình theo quy ước bài báo: không đặt tiêu đề trong hình (nội dung để ở chú thích của báo cáo), nhãn panel (a)/(b),
font Arial 8 pt, rộng 1 cột (8,8 cm) hoặc 2 cột (18 cm), PNG 300 dpi.

    python scripts/make_finetune_report.py
"""
import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finetune_clip import (CKPT_NAME, ClipSplit, encode_all, evaluate, load_checkpoint, load_pairs,  # noqa: E402
                           relevance_labels)

import clip  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

# Fine-tuned = xanh, CLIP gốc = cam + gạch chéo / nét đứt (in đen trắng vẫn phân biệt được)
C_FT, C_PRE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e6e6e6"
CM = 1 / 2.54
W1, W2 = 8.8 * CM, 18 * CM  # khổ 1 cột / 2 cột
PRE, FT = "CLIP gốc", "CLIP fine-tuned"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "lines.linewidth": 1.4,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel(ax, letter, text=""):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
    if text:
        ax.text(0.0, 1.04, f" {text}", transform=ax.transAxes, fontsize=8, ha="left", va="bottom", color=INK)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.png")


# ---------------------------------------------------------------- embedding
def compute_embeddings(rows, k, ckpt_path, cache_dir):
    """Embedding của CLIP gốc và fine-tuned cho toàn kho; lưu lại để các lần sau chạy ngay."""
    emb_file = cache_dir / f"eval_embeddings_k{k}.npz"
    if emb_file.exists() and emb_file.stat().st_mtime > ckpt_path.stat().st_mtime:
        e = np.load(emb_file)
        return {n: torch.from_numpy(e[n].astype(np.float32)) for n in e.files}
    cache_file = cache_dir / f"prefix_ViT-B-32_k{k}.pt"
    if not cache_file.exists():
        sys.exit(f"Thiếu {cache_file}; hãy chạy scripts/finetune_clip.py --eval-only để tạo lại")
    c = torch.load(cache_file, map_location="cpu")
    assert c["names"] == [r["name"] for r in rows], "Cache không khớp captions_final.csv, chạy lại finetune_clip.py"
    model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    model = model.float().eval()
    sp = ClipSplit(model, k)
    pre_i, pre_t = encode_all(sp, c["vis"], c["txt"], c["eot"], "cpu")
    load_checkpoint(model, ckpt_path)
    ft_i, ft_t = encode_all(sp, c["vis"], c["txt"], c["eot"], "cpu")
    out = {"pre_img": pre_i, "pre_txt": pre_t, "ft_img": ft_i, "ft_txt": ft_t}
    np.savez(emb_file, **{n: v.numpy().astype(np.float16) for n, v in out.items()})
    return out


def full_ranks(img_all, txt_all, idx):
    """Hạng của ảnh đúng khi truy vấn bằng caption trên toàn kho, và cosine cặp đúng trừ cosine ảnh sai cao nhất."""
    idx_t = torch.tensor(idx)
    sim = txt_all[idx_t] @ img_all.T
    pos = sim[torch.arange(len(idx)), idx_t]
    ranks = (sim > pos.unsqueeze(1)).sum(1) + 1
    sim[torch.arange(len(idx)), idx_t] = -2
    return ranks.numpy(), (pos - sim.max(1).values).numpy()


def per_landmark(labels, emb, test_idx, min_q=8, top=12):
    cnt = Counter(c for gi in test_idx for c in labels[gi])
    cats = [c for c, n in cnt.most_common() if n >= min_q][:top]
    res = {}
    for name in ("pre", "ft"):
        img = emb[f"{name}_img"]
        sim = img[torch.tensor(test_idx)] @ img.T
        sim[torch.arange(len(test_idx)), torch.tensor(test_idx)] = -2
        topi = sim.topk(10, dim=1).indices
        for c in cats:
            ps = [sum(1 for j in topi[qi].tolist() if c in labels[j]) / 10
                  for qi, gi in enumerate(test_idx) if c in labels[gi]]
            res[(name, c)] = float(np.mean(ps) * 100)
    return [(c, cnt[c], res[("pre", c)], res[("ft", c)]) for c in cats]


# ---------------------------------------------------------------- hình
def fig_training(metrics, out):
    hist, base, best = metrics["history"], metrics["pretrained"]["val"], metrics["best_epoch"]
    ep = [0] + [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 5.6 * CM))
    a1.plot(ep[1:], [h["loss"] for h in hist], color=C_FT, marker="o", ms=3)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss contrastive (train)")
    a1.set_ylim(0, None)
    panel(a1, "a")
    for key, ls, lab in (("t2i_R@1", "-", "Văn bản → ảnh"), ("i2t_R@1", "--", "Ảnh → văn bản")):
        a2.plot(ep, [base[key]] + [h["val"][key] for h in hist], color=C_FT, ls=ls, marker="o", ms=3, label=lab)
    a2.plot([0, 0], [base["t2i_R@1"], base["i2t_R@1"]], "o", ms=4.5, color=C_PRE, label=f"{PRE} (epoch 0)", zorder=5)
    a2.axvline(best, color=MUTED, lw=0.8, ls=":")
    a2.text(best + 0.15, 42.5, f"epoch được chọn = {best}", fontsize=6.5, color=INK2)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Recall@1 trên tập val (%)")
    a2.set_ylim(40, 85)
    a2.legend(loc="center right")
    panel(a2, "b")
    for a in (a1, a2):
        a.set_xticks(ep)
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "hinh1_qua_trinh_train")


def fig_metrics(pre, ft, out):
    groups = [
        ("Văn bản → ảnh (tập test)", ["t2i_R@1", "t2i_R@5", "t2i_R@10"], ["R@1", "R@5", "R@10"]),
        ("Ảnh → văn bản (tập test)", ["i2t_R@1", "i2t_R@5", "i2t_R@10"], ["R@1", "R@5", "R@10"]),
        ("Văn bản → ảnh (toàn kho)", ["t2i_full_R@1", "t2i_full_R@10", "t2i_full_R@50"], ["R@1", "R@10", "R@50"]),
        ("Ảnh → ảnh", ["i2i_P@10", "i2i_mAP@10"], ["P@10", "mAP@10"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 6.0 * CM), sharey=True, gridspec_kw={"width_ratios": [3, 3, 3, 2]})
    w = 0.38
    for ax, letter, (title, keys, labels) in zip(axes, "abcd", groups):
        x = np.arange(len(keys))
        b1 = ax.bar(x - w / 2 - 0.015, [pre[k] for k in keys], w, color=C_PRE, hatch="//////", edgecolor="white", lw=0)
        b2 = ax.bar(x + w / 2 + 0.015, [ft[k] for k in keys], w, color=C_FT, edgecolor="white", lw=0)
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}",
                        ha="center", va="bottom", fontsize=5.8, color=INK2)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
        panel(ax, letter, title)
    axes[0].set_ylabel("Giá trị (%)")
    fig.legend(handles=[Patch(facecolor=C_PRE, hatch="//////", edgecolor="white", label=PRE),
                        Patch(facecolor=C_FT, label=FT)], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
    save(fig, out, "hinh2_ket_qua_test")


def fig_ranking(r_pre, r_ft, g_pre, g_ft, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 6.2 * CM))
    ks = np.unique(np.round(np.logspace(0, 2, 60)).astype(int))
    for r, c, ls, lab in ((r_pre, C_PRE, "--", PRE), (r_ft, C_FT, "-", FT)):
        a1.plot(ks, [(r <= k).mean() * 100 for k in ks], color=c, ls=ls, label=lab)
        for k in (1, 10):
            v = (r <= k).mean() * 100
            a1.plot(k, v, "o", color=c, ms=3.5, mec="white", mew=0.8, zorder=5)
            # Đường đi lên bên phải nên góc dưới-phải của điểm luôn trống
            a1.annotate(f"{v:.1f}", (k, v), xytext=(4, -9), textcoords="offset points", fontsize=6.5, color=INK2)
    a1.set_xscale("log")
    a1.set_xlim(0.8, 115)
    a1.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    a1.minorticks_off()
    a1.set_ylim(0, 100)
    a1.set_xlabel("K"); a1.set_ylabel("Recall@K (%)")
    a1.legend(loc="lower right")
    panel(a1, "a", "Văn bản → ảnh trên toàn kho (7.756 ảnh)")

    bins = np.linspace(-0.12, 0.12, 49)
    for g, c, ls, lab in ((g_pre, C_PRE, "--", PRE), (g_ft, C_FT, "-", FT)):
        g = np.clip(g, bins[0], bins[-1])
        a2.hist(g, bins=bins, weights=np.full(len(g), 100 / len(g)), histtype="step", color=c, lw=1.4, ls=ls,
                label=f"{lab} ({(g > 0).mean() * 100:.1f}% > 0)")
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_xlabel("Cosine cặp đúng − cosine ảnh sai cao nhất"); a2.set_ylabel("Tỉ lệ truy vấn (%)")
    a2.set_ylim(0, a2.get_ylim()[1] * 1.3)  # chừa chỗ cho chú thích phía trên
    a2.legend(loc="upper right")
    panel(a2, "b", "Ảnh đúng xếp hạng 1 khi giá trị > 0")
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "hinh3_phan_tich_xep_hang")


def _square(path, size=300):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        m = min(w, h)
        return im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((size, size))


def fig_qualitative(rows, emb, test_idx, r_pre, r_ft, out, n_queries=3, topk=5, seed=7):
    # Truy vấn được chọn ngẫu nhiên trong các trường hợp fine-tuned xếp ảnh đúng hạng 1 còn CLIP gốc ngoài top-5
    cand = [i for i, (a, b) in enumerate(zip(r_pre, r_ft)) if b == 1 and a > topk]
    pick = np.random.default_rng(seed).choice(cand, size=min(n_queries, len(cand)), replace=False)
    fig = plt.figure(figsize=(W2, 7.2 * CM * len(pick)))
    gs = GridSpec(3 * len(pick), topk + 1, figure=fig, height_ratios=[0.38, 1, 1] * len(pick),
                  width_ratios=[1.0] + [1] * topk, hspace=0.1, wspace=0.05)
    for qi, p in enumerate(pick):
        gi = test_idx[p]
        tx = fig.add_subplot(gs[3 * qi, :])
        tx.axis("off")
        tx.text(0, 0.2, textwrap.fill(f"({'abc'[qi]})  “{rows[gi]['caption']}”", 125), fontsize=7, color=INK,
                ha="left", va="bottom", transform=tx.transAxes)
        for mi, (name, rk) in enumerate((("pre", r_pre[p]), ("ft", r_ft[p]))):
            r = 3 * qi + 1 + mi
            lab = fig.add_subplot(gs[r, 0])
            lab.axis("off")
            lab.text(0.94, 0.5, f"{PRE if name == 'pre' else FT}\nhạng ảnh đúng: {rk}", fontsize=6.5, ha="right",
                     va="center", color=C_FT if name == "ft" else C_PRE, fontweight="bold", transform=lab.transAxes)
            sim = emb[f"{name}_txt"][gi] @ emb[f"{name}_img"].T
            for j, ti in enumerate(sim.topk(topk).indices.tolist()):
                ax = fig.add_subplot(gs[r, j + 1])
                ax.imshow(_square(rows[ti]["path"]))
                ax.set_xticks([]); ax.set_yticks([])
                ax.grid(False)
                ok = ti == gi
                for sp in ax.spines.values():
                    sp.set_visible(ok); sp.set_color(INK); sp.set_linewidth(2.2)
                if ok:
                    ax.text(0.04, 0.96, "Ảnh đúng", transform=ax.transAxes, fontsize=6, color=INK, va="top",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
                if qi == 0 and mi == 0:
                    ax.set_title(f"top-{j + 1}", fontsize=7, color=INK2, pad=2)
    save(fig, out, "hinh4_vi_du_truy_van")


# ---------------------------------------------------------------- bảng
def md_table(title, header, body, align):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in body]
    return "\n".join(lines) + "\n"


def write_tables(metrics, n_trainable, split, pre, ft, per_lm, out: Path):
    a = metrics["args"]
    n_all = 151.3e6
    t1 = md_table("Bảng 1. Dữ liệu và cấu hình fine-tune CLIP", ["Hạng mục", "Giá trị"], [
        ["Mô hình nền", f"CLIP {metrics['base_model']} (OpenAI)"],
        ["Dữ liệu", f"{sum(len(v) for v in split.values()):,} cặp ảnh–caption".replace(",", ".")],
        ["Chia train / val / test", f"{len(split['train'])} / {len(split['val'])} / {len(split['test'])} (theo nhóm ảnh trùng)"],
        ["Block được train (mỗi nhánh)", f"{metrics['unfreeze_blocks']} / 12"],
        ["Tham số được train", f"{n_trainable / 1e6:.1f}M / {n_all / 1e6:.1f}M ({n_trainable / n_all * 100:.1f}%)"],
        ["Hàm mất mát", "InfoNCE đối xứng"],
        ["Tối ưu", f"AdamW, lr {a['lr']:g}, weight decay {a['weight_decay']}, warmup {int(a['warmup'] * 100)}% + cosine"],
        ["Batch / số epoch tối đa", f"{a['batch_size']} / {a['epochs']} (dừng sớm, patience {a['patience']})"],
        ["Epoch được chọn (theo rsum trên val)", str(metrics["best_epoch"])],
        ["Phần cứng", f"CPU, ~{np.mean([h['seconds'] for h in metrics['history']]) / 60:.1f} phút/epoch"],
    ], "ll")

    def better(k, m):
        return (ft[k] < pre[k]) if "MedR" in m else (ft[k] > pre[k])

    spec = [("Văn bản → ảnh", "R@1", "t2i_R@1"), ("", "R@5", "t2i_R@5"), ("", "R@10", "t2i_R@10"),
            ("", "MedR ↓", "t2i_MedR"),
            ("Ảnh → văn bản", "R@1", "i2t_R@1"), ("", "R@5", "i2t_R@5"), ("", "R@10", "i2t_R@10"),
            ("Văn bản → ảnh (toàn kho)", "R@1", "t2i_full_R@1"), ("", "R@10", "t2i_full_R@10"),
            ("", "R@50", "t2i_full_R@50"),
            ("Ảnh → ảnh", "P@10", "i2i_P@10"), ("", "mAP@10", "i2i_mAP@10"),
            ("Tổng R@K hai chiều", "rsum", "rsum")]
    t2 = md_table(f"Bảng 2. Kết quả trên tập test ({len(split['test'])} ảnh)",
                  ["Tác vụ", "Chỉ số", "CLIP gốc", "CLIP fine-tuned", "Chênh lệch"],
                  [[tsk, m, f"{pre[k]:.2f}", f"**{ft[k]:.2f}**" if better(k, m) else f"{ft[k]:.2f}", f"{ft[k] - pre[k]:+.2f}"]
                   for tsk, m, k in spec], "llrrr")
    t2 += ("\nR@K, P@10, mAP@10 tính theo %; MedR (hạng trung vị của ảnh đúng) càng nhỏ càng tốt. "
           "\"Tập test\": tìm trong 776 ảnh test; \"toàn kho\": tìm trong cả 7.756 ảnh. "
           "Ảnh → ảnh: ảnh liên quan là ảnh có chung category Wikimedia Commons cụ thể (không dùng khi train).\n")

    base = metrics["pretrained"]["val"]
    hist = [{"epoch": 0, "loss": None, "val": base}] + metrics["history"]
    t3 = md_table("Bảng 3. Chỉ số trên tập val theo epoch (epoch 0 = CLIP gốc)",
                  ["Epoch", "Loss", "V→A R@1", "V→A R@10", "A→V R@1", "A→V R@10", "A→A P@10", "rsum"],
                  [[h["epoch"], "–" if h["loss"] is None else f"{h['loss']:.4f}"]
                   + [f"{h['val'][k]:.1f}" for k in ("t2i_R@1", "t2i_R@10", "i2t_R@1", "i2t_R@10", "i2i_P@10")]
                   + [f"**{h['val']['rsum']:.1f}**" if h["epoch"] == metrics["best_epoch"] else f"{h['val']['rsum']:.1f}"]
                   for h in hist], "rrrrrrrr")
    t3 += "\nV→A: văn bản → ảnh, A→V: ảnh → văn bản, A→A: ảnh → ảnh. In đậm: epoch được chọn.\n"

    t4 = md_table("Bảng 4. Tìm ảnh tương đồng theo công trình trên tập test (P@10, %)",
                  ["Công trình (category Commons)", "Số truy vấn", "CLIP gốc", "CLIP fine-tuned", "Chênh lệch"],
                  [[lm, n, f"{p:.1f}", f"{f:.1f}", f"{f - p:+.1f}"] for lm, n, p, f in per_lm], "lrrrr")

    (out / "bang_ket_qua.md").write_text("# Kết quả fine-tune CLIP\n\n" + "\n".join([t1, t2, t3, t4]), encoding="utf-8")
    print("  bang_ket_qua.md")


def main():
    ap = argparse.ArgumentParser(description="Sinh hình + bảng kết quả fine-tune CLIP")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "clip_finetuned"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "clip_finetune"))
    ap.add_argument("--captions", default=str(ROOT / "scripts" / "compare_output" / "captions_final.csv"))
    ap.add_argument("--image-dir", default=str(ROOT / "data" / "vietnam_architecture"))
    args = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    model_dir, out = Path(args.model_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    split = json.loads((model_dir / "split.json").read_text(encoding="utf-8"))
    ckpt_path = model_dir / CKPT_NAME
    n_trainable = sum(v.numel() for v in torch.load(ckpt_path, map_location="cpu")["state_dict"].values())

    rows, _ = load_pairs(Path(args.captions), Path(args.image_dir))
    pos = {r["name"]: i for i, r in enumerate(rows)}
    test_idx = [pos[n] for n in split["test"] if n in pos]
    labels = relevance_labels(rows)

    emb = compute_embeddings(rows, metrics["unfreeze_blocks"], ckpt_path, model_dir / "cache")
    pre = evaluate(emb["pre_img"], emb["pre_txt"], test_idx, labels)
    ft = evaluate(emb["ft_img"], emb["ft_txt"], test_idx, labels)
    r_pre, g_pre = full_ranks(emb["pre_img"], emb["pre_txt"], test_idx)
    r_ft, g_ft = full_ranks(emb["ft_img"], emb["ft_txt"], test_idx)

    style()
    print(f"Xuất vào {out}")
    fig_training(metrics, out)
    fig_metrics(pre, ft, out)
    fig_ranking(r_pre, r_ft, g_pre, g_ft, out)
    fig_qualitative(rows, emb, test_idx, r_pre, r_ft, out)
    write_tables(metrics, n_trainable, split, pre, ft, per_landmark(labels, emb, test_idx), out)


if __name__ == "__main__":
    main()
