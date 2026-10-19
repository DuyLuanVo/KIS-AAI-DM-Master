"""
Fine-tune CLIP trên cặp (ảnh, caption) của bộ dữ liệu kiến trúc Việt Nam.

Mục tiêu: đưa không gian vector của CLIP về đúng miền dữ liệu, để
* tìm bằng ảnh (image -> image) phân biệt tốt hơn các kiểu kiến trúc (tháp Chăm, đình, chùa, nhà Pháp...);
* có thêm nhánh tìm bằng mô tả trực tiếp trên ảnh (CLIP text -> image), bắt được chi tiết caption bỏ sót.

Phương pháp
* Loss contrastive đối xứng (InfoNCE) như CLIP gốc: trong một batch, ảnh i phải gần caption i và xa các caption khác.
* Chỉ mở khóa K block Transformer cuối của cả nhánh ảnh lẫn nhánh văn bản (+ LayerNorm cuối, lớp chiếu, logit_scale),
  các block đầu giữ nguyên để không "quên" kiến thức gốc khi dữ liệu chỉ vài nghìn ảnh.
* Phần bị đóng băng chỉ cần chạy MỘT lần: đầu ra của nó được lưu cache, mỗi epoch chỉ chạy K block cuối
  -> train được trên CPU trong vài phút/epoch. Có GPU thì thêm --augment để cắt/xoay ảnh ngẫu nhiên mỗi epoch.
* Chia train/val/test theo nhóm ảnh trùng (phash) để ảnh gần giống nhau không rơi vào cả train lẫn test.
  Chọn checkpoint theo val, báo cáo trên test, so với CLIP gốc.

Đánh giá (tập test)
* text->image, image->text: Recall@1/5/10, MedR (gallery = ảnh test) và text->image trên toàn bộ kho.
* image->image: Precision@10 và mAP@10, ảnh "liên quan" = có chung category Wikimedia Commons cụ thể
  (vd. "Linh Phuoc Pagoda", "Turtle Tower"). Category KHÔNG dùng khi train nên đây là thước đo độc lập.

Đầu ra (mặc định models/clip_finetuned/)
* clip_vitb32_kis.pt : trọng số các tham số đã fine-tune (nạp chồng lên CLIP gốc), backend đọc file này
* metrics.json       : kết quả trước/sau fine-tune + lịch sử train
* split.json         : danh sách ảnh train/val/test (dùng lại ở các lần chạy sau)

Ví dụ:
    python scripts/finetune_clip.py                       # train với cấu hình mặc định
    python scripts/finetune_clip.py --unfreeze-blocks 6   # mở khóa nhiều block hơn
    python scripts/finetune_clip.py --eval-only           # chỉ đánh giá checkpoint đã có so với CLIP gốc
    python scripts/finetune_clip.py --device cuda --augment --epochs 20
"""
import argparse
import csv
import json
import math
import ntpath
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import clip  # noqa: E402

CKPT_NAME = "clip_vitb32_kis.pt"
_DATE_CAT_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


# ---------------------------------------------------------------- dữ liệu
def load_meta(img_path: Path) -> dict:
    j = img_path.with_suffix(".json")
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_pairs(captions_csv: Path, image_dir: Path):
    rows, missing = [], 0
    with open(captions_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fp = (r.get("filepath") or r.get("file") or "").strip()
            cap = re.sub(r"\s+", " ", r.get("caption") or "").strip()
            if not fp or not cap:
                continue
            name = ntpath.basename(fp)  # đường dẫn trong CSV có thể là Windows của máy khác
            path = image_dir / name
            if not path.exists():
                if Path(fp).exists():
                    path = Path(fp)
                else:
                    missing += 1
                    continue
            meta = load_meta(path)
            # Category theo thời gian ("2026 in Hanoi", "November 2013 in Vietnam") không nói gì về công trình
            cats = [c for c in (meta.get("categories") or []) if not _DATE_CAT_RE.search(c)]
            rows.append({"name": name, "path": str(path), "caption": cap,
                         "group": meta.get("dup_of") or meta.get("phash") or name, "cats": cats})
    return rows, missing


def make_split(rows, val_frac: float, test_frac: float, seed: int):
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["name"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(rows)
    split = {"test": [], "val": [], "train": []}
    for g in keys:
        if len(split["test"]) < test_frac * n:
            split["test"] += groups[g]
        elif len(split["val"]) < val_frac * n:
            split["val"] += groups[g]
        else:
            split["train"] += groups[g]
    return split


def relevance_labels(rows, max_frac: float = 0.05):
    """Category cụ thể của từng ảnh: xuất hiện >= 2 ảnh và không quá phổ biến (<= max_frac kho)."""
    cnt = Counter(c for r in rows for c in set(r["cats"]))
    hi = max_frac * len(rows)
    return [{c for c in r["cats"] if 2 <= cnt[c] <= hi} for r in rows]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


def augment_transform(preprocess):
    from torchvision import transforms as T

    norm = [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        norm,
    ])


# ---------------------------------------------------------------- tách CLIP thành phần đóng băng / phần train
def run_blocks(blocks, x, mask=None):
    """Chạy các ResidualAttentionBlock của CLIP trên x dạng [L, N, D] (cho phép mask ngắn hơn 77)."""
    for blk in blocks:
        h = blk.ln_1(x)
        x = x + blk.attn(h, h, h, need_weights=False, attn_mask=mask)[0]
        x = x + blk.mlp(blk.ln_2(x))
    return x


class ClipSplit:
    def __init__(self, model, k: int):
        self.m, self.k = model, k
        self.v_blocks = model.visual.transformer.resblocks
        self.t_blocks = model.transformer.resblocks
        nv, nt = len(self.v_blocks), len(self.t_blocks)
        if not 1 <= k <= min(nv, nt):
            raise ValueError(f"--unfreeze-blocks phải trong [1, {min(nv, nt)}]")
        self.nv, self.nt = nv, nt
        self.causal = torch.empty(77, 77).fill_(float("-inf")).triu_(1)

    def trainable_names(self):
        names = {"logit_scale", "visual.ln_post.weight", "visual.ln_post.bias", "visual.proj",
                 "ln_final.weight", "ln_final.bias", "text_projection"}
        for n, _ in self.m.named_parameters():
            for prefix, total in (("visual.transformer.resblocks.", self.nv), ("transformer.resblocks.", self.nt)):
                if n.startswith(prefix) and int(n[len(prefix):].split(".")[0]) >= total - self.k:
                    names.add(n)
        return names

    # ảnh
    def visual_prefix(self, x):
        v = self.m.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x).permute(1, 0, 2)
        x = run_blocks(self.v_blocks[: self.nv - self.k], x)
        return x.permute(1, 0, 2)

    def visual_suffix(self, h):
        v = self.m.visual
        x = run_blocks(self.v_blocks[self.nv - self.k:], h.permute(1, 0, 2)).permute(1, 0, 2)
        return v.ln_post(x[:, 0, :]) @ v.proj

    # văn bản (mask nhân quả: token EOT chỉ phụ thuộc token trước nó nên cắt được phần đệm phía sau)
    def text_prefix(self, tokens):
        L = tokens.shape[1]
        x = self.m.token_embedding(tokens) + self.m.positional_embedding[:L]
        x = run_blocks(self.t_blocks[: self.nt - self.k], x.permute(1, 0, 2), self.causal[:L, :L].to(x.device))
        return x.permute(1, 0, 2)

    def text_suffix(self, h, eot):
        L = h.shape[1]
        x = run_blocks(self.t_blocks[self.nt - self.k:], h.permute(1, 0, 2), self.causal[:L, :L].to(h.device))
        x = self.m.ln_final(x.permute(1, 0, 2))
        return x[torch.arange(x.shape[0], device=x.device), eot] @ self.m.text_projection


# ---------------------------------------------------------------- cache phần đóng băng
@torch.no_grad()
def build_cache(split: ClipSplit, rows, preprocess, device, batch_size, workers):
    dl = torch.utils.data.DataLoader(ImageDataset([r["path"] for r in rows], preprocess),
                                     batch_size=batch_size, num_workers=workers)
    vis, t0 = [], time.time()
    for i, x in enumerate(dl):
        vis.append(split.visual_prefix(x.to(device)).half().cpu())
        done = min((i + 1) * batch_size, len(rows))
        print(f"\r  cache ảnh {done}/{len(rows)} ({done / (time.time() - t0):.1f} ảnh/s)", end="", flush=True)
    print()
    tokens = clip.tokenize([r["caption"] for r in rows], truncate=True)
    eot = tokens.argmax(dim=-1)
    tokens = tokens[:, : int(eot.max()) + 1]
    txt = []
    for s in range(0, len(rows), 256):
        txt.append(split.text_prefix(tokens[s: s + 256].to(device)).half().cpu())
    return torch.cat(vis), torch.cat(txt), eot


def load_or_build_cache(split, rows, preprocess, device, args, out_dir: Path):
    path = out_dir / "cache" / f"prefix_{args.base_model.replace('/', '-')}_k{args.unfreeze_blocks}.pt"
    names = [r["name"] for r in rows]
    caps = [r["caption"] for r in rows]
    if path.exists() and not args.rebuild_cache:
        c = torch.load(path, map_location="cpu")
        if c.get("names") == names and c.get("captions") == caps:
            print(f"Dùng cache: {path}")
            return c["vis"], c["txt"], c["eot"]
    print(f"Tính cache cho {len(rows)} ảnh (chỉ chạy một lần)...")
    vis, txt, eot = build_cache(split, rows, preprocess, device, args.batch_size_cache, args.workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "captions": caps, "vis": vis, "txt": txt, "eot": eot}, path)
    print(f"Đã lưu cache: {path} ({path.stat().st_size / 1e9:.2f} GB)")
    return vis, txt, eot


# ---------------------------------------------------------------- đánh giá
@torch.no_grad()
def encode_all(split: ClipSplit, vis, txt, eot, device, bs=512):
    img, tx = [], []
    for s in range(0, len(vis), bs):
        img.append(F.normalize(split.visual_suffix(vis[s: s + bs].to(device).float()), dim=-1).cpu())
        tx.append(F.normalize(split.text_suffix(txt[s: s + bs].to(device).float(), eot[s: s + bs].to(device)), dim=-1).cpu())
    return torch.cat(img), torch.cat(tx)


def recall_metrics(sim: torch.Tensor, prefix: str):
    """sim[i, j]: truy vấn i với phần tử j, đáp án đúng của truy vấn i là j = i."""
    target = sim.diag().unsqueeze(1)
    ranks = (sim > target).sum(dim=1) + 1
    out = {f"{prefix}_R@{k}": float((ranks <= k).float().mean() * 100) for k in (1, 5, 10)}
    out[f"{prefix}_MedR"] = float(ranks.float().median())
    return out


def evaluate(img_all, txt_all, idx, labels):
    idx_t = torch.tensor(idx)
    img, txt = img_all[idx_t], txt_all[idx_t]
    m = {}
    m.update(recall_metrics(txt @ img.T, "t2i"))
    m.update(recall_metrics(img @ txt.T, "i2t"))
    # text -> image trên toàn bộ kho (giống khi chạy thật)
    sim_full = txt @ img_all.T
    target = sim_full[torch.arange(len(idx)), idx_t].unsqueeze(1)
    ranks = (sim_full > target).sum(dim=1) + 1
    for k in (1, 10, 50):
        m[f"t2i_full_R@{k}"] = float((ranks <= k).float().mean() * 100)
    # image -> image, liên quan = chung category cụ thể
    sim_ii = img @ img_all.T
    sim_ii[torch.arange(len(idx)), idx_t] = -2  # bỏ chính nó
    top = sim_ii.topk(10, dim=1).indices
    p10, ap10, nq = [], [], 0
    for qi, gi in enumerate(idx):
        qlab = labels[gi]
        if not qlab:
            continue
        n_rel = sum(1 for j, lab in enumerate(labels) if j != gi and lab & qlab)
        if n_rel == 0:
            continue
        hits = [1 if labels[j] & qlab else 0 for j in top[qi].tolist()]
        p10.append(sum(hits) / 10)
        prec = [sum(hits[: r + 1]) / (r + 1) for r in range(10) if hits[r]]
        ap10.append(sum(prec) / min(n_rel, 10))
        nq += 1
    m["i2i_P@10"] = float(np.mean(p10) * 100) if p10 else 0.0
    m["i2i_mAP@10"] = float(np.mean(ap10) * 100) if ap10 else 0.0
    m["i2i_queries"] = nq
    m["rsum"] = sum(m[f"{d}_R@{k}"] for d in ("t2i", "i2t") for k in (1, 5, 10))
    return m


def fmt(m):
    return (f"t2i R@1/5/10 {m['t2i_R@1']:.1f}/{m['t2i_R@5']:.1f}/{m['t2i_R@10']:.1f} | "
            f"i2t R@1/5/10 {m['i2t_R@1']:.1f}/{m['i2t_R@5']:.1f}/{m['i2t_R@10']:.1f} | "
            f"t2i(toàn kho) R@10 {m['t2i_full_R@10']:.1f} | i2i P@10 {m['i2i_P@10']:.1f} mAP@10 {m['i2i_mAP@10']:.1f}")


def print_table(base, ft):
    keys = ["t2i_R@1", "t2i_R@5", "t2i_R@10", "t2i_MedR", "i2t_R@1", "i2t_R@5", "i2t_R@10", "i2t_MedR",
            "t2i_full_R@1", "t2i_full_R@10", "t2i_full_R@50", "i2i_P@10", "i2i_mAP@10", "rsum"]
    print(f"\n{'Chỉ số (test)':<16}{'CLIP gốc':>12}{'Fine-tuned':>12}{'Chênh':>10}")
    for k in keys:
        print(f"{k:<16}{base[k]:>12.2f}{ft[k]:>12.2f}{ft[k] - base[k]:>+10.2f}")
    print(f"(i2i đánh giá trên {ft['i2i_queries']} ảnh test có category cụ thể; MedR càng nhỏ càng tốt)")


# ---------------------------------------------------------------- checkpoint
def load_checkpoint(model, path: Path):
    ck = torch.load(path, map_location="cpu")
    res = model.load_state_dict(ck["state_dict"], strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"Checkpoint không khớp mô hình: {res.unexpected_keys[:5]}")
    return ck


def save_checkpoint(model, names, path: Path, args, meta):
    sd = {n: p.detach().cpu().float().clone() for n, p in model.named_parameters() if n in names}
    torch.save({"base_model": args.base_model, "unfreeze_blocks": args.unfreeze_blocks, "state_dict": sd,
                "meta": meta}, path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Fine-tune CLIP trên cặp ảnh-caption kiến trúc")
    ap.add_argument("--captions", default=str(ROOT / "scripts" / "compare_output" / "captions_final.csv"))
    ap.add_argument("--image-dir", default=str(ROOT / "data" / "vietnam_architecture"))
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "clip_finetuned"))
    ap.add_argument("--base-model", default="ViT-B/32")
    ap.add_argument("--unfreeze-blocks", type=int, default=4, help="Số block Transformer cuối được train (mỗi nhánh)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256, help="Batch càng lớn, càng nhiều mẫu âm cho loss contrastive")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.1, help="Tỉ lệ số bước warmup")
    ap.add_argument("--patience", type=int, default=4, help="Dừng sớm nếu val không cải thiện sau N epoch")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resplit", action="store_true", help="Chia lại train/val/test thay vì dùng split.json có sẵn")
    ap.add_argument("--augment", action="store_true", help="Tăng cường ảnh mỗi epoch (chậm, nên dùng GPU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--batch-size-cache", type=int, default=64)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Chỉ dùng N ảnh đầu (thử nhanh)")
    ap.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá checkpoint có sẵn so với CLIP gốc")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / CKPT_NAME

    rows, missing = load_pairs(Path(args.captions), Path(args.image_dir))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Cặp ảnh-caption: {len(rows)} (thiếu ảnh: {missing}) | thiết bị: {device}")

    split_path = out_dir / ("split.json" if not args.limit else f"split_limit{args.limit}.json")
    if split_path.exists() and not args.resplit:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        known = set(split["train"]) | set(split["val"]) | set(split["test"])
        new = [r["name"] for r in rows if r["name"] not in known]
        split["train"] += new  # ảnh mới thêm vào CSV sau lần chia trước -> cho vào train
        print(f"Dùng split có sẵn: {split_path}" + (f" (+{len(new)} ảnh mới vào train)" if new else ""))
    else:
        split = make_split(rows, args.val_frac, args.test_frac, args.seed)
        split_path.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    pos = {r["name"]: i for i, r in enumerate(rows)}
    idx = {k: [pos[n] for n in v if n in pos] for k, v in split.items()}
    print(f"train {len(idx['train'])} | val {len(idx['val'])} | test {len(idx['test'])}")
    labels = relevance_labels(rows)

    model, preprocess = clip.load(args.base_model, device="cpu", jit=False)
    model = model.float().to(device).eval()
    split_m = ClipSplit(model, args.unfreeze_blocks)

    # Kiểm tra cách tách prefix/suffix cho kết quả giống hệt model.encode_image / encode_text
    with torch.no_grad():
        x = torch.stack([preprocess(Image.open(rows[i]["path"]).convert("RGB")) for i in range(2)]).to(device)
        tok = clip.tokenize([rows[i]["caption"] for i in range(2)], truncate=True).to(device)
        eot = tok.argmax(-1)
        tok_cut = tok[:, : int(eot.max()) + 1]
        a = split_m.visual_suffix(split_m.visual_prefix(x))
        b = split_m.text_suffix(split_m.text_prefix(tok_cut), eot)
        assert torch.allclose(a, model.encode_image(x), atol=1e-3), "Tách nhánh ảnh sai"
        assert torch.allclose(b, model.encode_text(tok), atol=1e-3), "Tách nhánh văn bản sai"

    vis, txt, eot = load_or_build_cache(split_m, rows, preprocess, device, args, out_dir)

    img_all, txt_all = encode_all(split_m, vis, txt, eot, device)
    base_val = evaluate(img_all, txt_all, idx["val"], labels)
    base_test = evaluate(img_all, txt_all, idx["test"], labels)
    print(f"CLIP gốc  | val : {fmt(base_val)}")

    if args.eval_only:
        if not ckpt_path.exists():
            sys.exit(f"Chưa có checkpoint {ckpt_path}")
        ck = load_checkpoint(model, ckpt_path)
        if ck.get("unfreeze_blocks") != args.unfreeze_blocks:
            sys.exit(f"Checkpoint dùng --unfreeze-blocks {ck.get('unfreeze_blocks')}, hãy chạy lại với giá trị đó")
        img_all, txt_all = encode_all(split_m, vis, txt, eot, device)
        print_table(base_test, evaluate(img_all, txt_all, idx["test"], labels))
        return

    # ---- train
    names = split_m.trainable_names()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        p.requires_grad = n in names
        if p.requires_grad:
            (no_decay if p.ndim < 2 or n == "logit_scale" else decay).append(p)
    n_train = sum(p.numel() for p in decay + no_decay)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Tham số được train: {n_train / 1e6:.1f}M / {n_all / 1e6:.1f}M ({100 * n_train / n_all:.1f}%)")
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    train_idx = torch.tensor(idx["train"])
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)  # bỏ batch lẻ cuối: batch nhỏ thì ít mẫu âm
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    aug_ds = None
    if args.augment:
        aug_ds = ImageDataset([r["path"] for r in rows], augment_transform(preprocess))

    history, best, best_epoch, bad = [], base_val["rsum"], 0, 0
    print(f"Train {args.epochs} epoch x {steps_per_epoch} bước, batch {args.batch_size}, lr {args.lr}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = train_idx[torch.randperm(len(train_idx))]
        for s in range(steps_per_epoch):
            b = perm[s * args.batch_size: (s + 1) * args.batch_size]
            if aug_ds is not None:
                with torch.no_grad():
                    x = torch.stack([aug_ds[i] for i in b.tolist()]).to(device)
                    vh = split_m.visual_prefix(x)
            else:
                vh = vis[b].to(device).float()
            img = F.normalize(split_m.visual_suffix(vh), dim=-1)
            tx = F.normalize(split_m.text_suffix(txt[b].to(device).float(), eot[b].to(device)), dim=-1)
            logits = model.logit_scale.exp() * img @ tx.T
            target = torch.arange(len(b), device=device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decay + no_decay, 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            losses.append(loss.item())
            print(f"\r  epoch {epoch} bước {s + 1}/{steps_per_epoch} loss {np.mean(losses):.4f}", end="", flush=True)
        model.eval()
        img_all, txt_all = encode_all(split_m, vis, txt, eot, device)
        val = evaluate(img_all, txt_all, idx["val"], labels)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val, "seconds": time.time() - t0})
        mark = ""
        if val["rsum"] > best:
            best, best_epoch, bad, mark = val["rsum"], epoch, 0, "  * lưu checkpoint"
            save_checkpoint(model, names, ckpt_path, args, {"epoch": epoch, "val": val})
        else:
            bad += 1
        print(f"\r  epoch {epoch} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s) | val: {fmt(val)}{mark}")
        if bad >= args.patience:
            print(f"Dừng sớm: val không cải thiện sau {args.patience} epoch")
            break

    if best_epoch == 0:
        print("Fine-tune không vượt được CLIP gốc trên val; không lưu checkpoint. Thử lr nhỏ hơn hoặc ít block hơn.")
        return

    # ---- báo cáo trên test với checkpoint tốt nhất
    ck = load_checkpoint(model, ckpt_path)
    img_all, txt_all = encode_all(split_m, vis, txt, eot, device)
    ft_test = evaluate(img_all, txt_all, idx["test"], labels)
    print_table(base_test, ft_test)

    # Phân bố cosine của cặp (ảnh, caption) đúng trên val, để tham khảo khi chỉnh ngưỡng CLIP_MIN_SCORE của backend
    v = torch.tensor(idx["val"])
    pos_scores = (img_all[v] * txt_all[v]).sum(-1)
    pos_stats = {f"p{q}": round(float(pos_scores.quantile(q / 100)), 4) for q in (10, 50, 90)}
    meta = {**ck["meta"], "test": ft_test, "pretrained_test": base_test, "val_pair_cosine": pos_stats,
            "trained_at": datetime.now().isoformat(timespec="seconds"), "n_train": len(idx["train"]),
            "args": {k: v for k, v in vars(args).items()}}
    save_checkpoint(model, names, ckpt_path, args, meta)
    (out_dir / "metrics.json").write_text(json.dumps({
        "base_model": args.base_model, "unfreeze_blocks": args.unfreeze_blocks, "best_epoch": best_epoch,
        "pretrained": {"val": base_val, "test": base_test}, "finetuned": {"test": ft_test},
        "val_pair_cosine": pos_stats, "history": history, "args": vars(args),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCheckpoint: {ckpt_path} (epoch {best_epoch})")
    print(f"Kết quả  : {out_dir / 'metrics.json'}")
    print("Dùng trong backend: đặt CLIP_FINETUNED_PATH trong backend/.env rồi chạy python scripts/reembed_images.py")


if __name__ == "__main__":
    main()
