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
  -> train được trên CPU trong vài phút/epoch.
* Khử trùng trước khi chia (scripts/dedup_images.py: cosine CLIP >= 0,97, hoặc 0,90-0,97 kèm ORB+RANSAC >= 40 điểm
  khớp): mỗi cụm trùng chỉ giữ một ảnh, nên mỗi ảnh còn lại là một nhóm riêng và việc chia train/val/test là ngẫu
  nhiên theo từng ảnh. Kiểm tra chống rò rỉ vẫn chạy: split cũ mà có nhóm nằm ở nhiều tập thì dừng, yêu cầu --resplit.
* Tăng cường ảnh CHỈ trên tập train (scripts/image_preprocess.py, chính sách hiểu caption):
  --aug-views V  tạo trước V biến thể ngẫu nhiên cho mỗi ảnh train, chạy phần đóng băng một lần rồi cache
                 -> mỗi epoch mỗi ảnh lấy ngẫu nhiên bản gốc hoặc một biến thể, vẫn nhanh trên CPU;
  --augment      tăng cường trực tuyến mỗi bước (biến thể không giới hạn, chậm, nên dùng GPU).
  Val/test luôn qua preprocess xác định của CLIP; mean/std chuẩn hóa là hằng số của CLIP, không tính từ dữ liệu.
* Chọn checkpoint theo val, báo cáo trên test, so với CLIP gốc.

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
    python scripts/dedup_images.py && python scripts/finetune_clip.py --resplit --device cuda --augment --grad-checkpoint
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
import torch.utils.checkpoint
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_preprocess import (POLICIES, POLICY_VERSION, caption_flags, describe_policy, open_rgb,  # noqa: E402
                              train_transform)

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


def load_dedup(path: Path):
    """dedup_groups.csv -> {tên ảnh: (keep, split_group)}"""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["image_file"]] = (r["keep"] == "1", r["split_group"])
    return out


def load_pairs(captions_csv: Path, image_dir: Path, dedup=None):
    rows, missing, dropped = [], 0, 0
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
            group = None
            if dedup is not None and name in dedup:
                keep, group = dedup[name]
                if not keep:
                    dropped += 1
                    continue
            meta = load_meta(path)
            # Category theo thời gian ("2026 in Hanoi", "November 2013 in Vietnam") không nói gì về công trình
            cats = [c for c in (meta.get("categories") or []) if not _DATE_CAT_RE.search(c)]
            rows.append({"name": name, "path": str(path), "caption": cap,
                         "group": group or name, "cats": cats})
    return rows, missing, dropped


def split_leaks(rows, split):
    """Các nhóm ảnh trùng/cùng cảnh nằm ở nhiều hơn một tập (rò rỉ)."""
    where = {n: k for k, v in split.items() for n in v}
    seen = {}
    for r in rows:
        if r["name"] in where:
            seen.setdefault(r["group"], set()).add(where[r["name"]])
    return {g: s for g, s in seen.items() if len(s) > 1}


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
    """transform: một transform chung, hoặc list transform theo từng ảnh (tăng cường hiểu caption).
    seeds: nếu có, mỗi ảnh dùng seed riêng -> biến thể tái lập được, không phụ thuộc số worker."""

    def __init__(self, paths, transform, seeds=None):
        self.paths, self.transform, self.seeds = paths, transform, seeds

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        tf = self.transform[i] if isinstance(self.transform, list) else self.transform
        if self.seeds is not None:
            torch.manual_seed(self.seeds[i])
            random.seed(self.seeds[i])
        return tf(open_rgb(self.paths[i]))


def train_transforms(rows, train_idx, preprocess, policy="default"):
    """Transform tăng cường cho từng ảnh train (chỉ train), dùng chung đối tượng cho các ảnh cùng bộ cờ."""
    cache, out = {}, []
    for i in train_idx:
        flags = caption_flags(rows[i]["caption"])
        key = tuple(sorted(flags.items()))
        if key not in cache:
            cache[key] = train_transform(preprocess, flags, policy=policy)
        out.append(cache[key])
    return out


# ---------------------------------------------------------------- tách CLIP thành phần đóng băng / phần train
GRAD_CHECKPOINT = False  # bật khi train trên GPU ít VRAM: không lưu activation, tính lại lúc backward
DROP_PATH = 0.0          # stochastic depth cho các block ĐANG TRAIN (chỉ khi train, có gradient)


def _drop_path(y, p):
    """Bỏ ngẫu nhiên nhánh residual của từng mẫu với xác suất p (y dạng [L, N, D]), giữ kỳ vọng bằng cách chia 1-p."""
    keep = (torch.rand(1, y.shape[1], 1, device=y.device) >= p).to(y.dtype)
    return y * keep / (1 - p)


def _block(blk, x, mask, dp=0.0):
    h = blk.ln_1(x)
    a = blk.attn(h, h, h, need_weights=False, attn_mask=mask)[0]
    x = x + (_drop_path(a, dp) if dp > 0 else a)
    m = blk.mlp(blk.ln_2(x))
    return x + (_drop_path(m, dp) if dp > 0 else m)


def run_blocks(blocks, x, mask=None):
    """Chạy các ResidualAttentionBlock của CLIP trên x dạng [L, N, D] (cho phép mask ngắn hơn 77)."""
    for blk in blocks:
        train_mode = torch.is_grad_enabled() and blk.training
        dp = DROP_PATH if train_mode else 0.0
        if GRAD_CHECKPOINT and train_mode:
            x = torch.utils.checkpoint.checkpoint(_block, blk, x, mask, dp, use_reentrant=False)
        else:
            x = _block(blk, x, mask, dp)
    return x


def contrastive_ce(logits):
    """Cross-entropy một chiều (hàng = truy vấn, đáp án ở đường chéo)."""
    return -F.log_softmax(logits, dim=1).diagonal().mean()


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


@torch.no_grad()
def load_or_build_aug_cache(split, rows, train_idx, preprocess, device, args, out_dir: Path, shape):
    """V biến thể tăng cường của MỖI ảnh train (không ảnh val/test nào đi qua đây), qua phần đóng băng,
    lưu fp16 dạng memmap [V, n_train, T, D]. Khóa cache = danh sách ảnh train + caption + chính sách + seed."""
    import hashlib

    V = args.aug_views
    key = hashlib.md5(json.dumps([[rows[i]["name"], rows[i]["caption"]] for i in train_idx] +
                                 [POLICY_VERSION, args.aug_policy, args.seed, V, args.base_model, args.unfreeze_blocks]
                                 ).encode("utf-8")).hexdigest()[:12]
    path = out_dir / "cache" / f"aug_{args.base_model.replace('/', '-')}_k{args.unfreeze_blocks}_v{V}_{key}.npy"
    T_, D_ = shape
    full_shape = (V, len(train_idx), T_, D_)
    done_flag = path.with_suffix(".done")
    if path.exists() and done_flag.exists() and not args.rebuild_cache:
        print(f"Dùng cache tăng cường: {path}")
        return np.load(path, mmap_mode="r")
    for old in path.parent.glob(f"aug_{args.base_model.replace('/', '-')}_k{args.unfreeze_blocks}_*"):
        old.unlink()  # cache tăng cường cũ (split/caption khác) -> xóa cho đỡ tốn đĩa
    gb = np.prod(full_shape) * 2 / 1e9
    print(f"Tạo {V} biến thể tăng cường x {len(train_idx)} ảnh train ({gb:.1f} GB trên đĩa, chỉ chạy một lần)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=full_shape)
    tfs = train_transforms(rows, train_idx, preprocess, args.aug_policy)
    paths = [rows[i]["path"] for i in train_idx]
    t0 = time.time()
    for v in range(V):
        seeds = [args.seed * 1_000_003 + v * 100_003 + int(i) for i in train_idx]
        dl = torch.utils.data.DataLoader(ImageDataset(paths, tfs, seeds), batch_size=args.batch_size_cache,
                                         num_workers=args.workers)
        s = 0
        for x in dl:
            mm[v, s: s + len(x)] = split.visual_prefix(x.to(device)).half().cpu().numpy()
            s += len(x)
            done = v * len(train_idx) + s
            print(f"\r  biến thể {v + 1}/{V}: {s}/{len(train_idx)} ({done / (time.time() - t0):.1f} ảnh/s)",
                  end="", flush=True)
    print()
    mm.flush()
    del mm
    done_flag.write_text(describe_policy(args.aug_policy), encoding="utf-8")
    return np.load(path, mmap_mode="r")


def save_aug_preview(rows, train_idx, preprocess, path: Path, n_img=4, n_view=6, seed=0, policy="default"):
    """Ảnh minh họa: mỗi hàng một ảnh train, cột đầu là preprocess gốc (như val/test), các cột sau là biến thể."""
    from torchvision.utils import make_grid, save_image

    rng = random.Random(seed)
    pick = rng.sample(list(train_idx), min(n_img, len(train_idx)))
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    tiles = []
    for i in pick:
        im = open_rgb(rows[i]["path"])
        tf = train_transform(preprocess, caption_flags(rows[i]["caption"]), policy=policy)
        tiles.append(preprocess(im))
        for v in range(n_view):
            torch.manual_seed(seed * 1000 + v)
            tiles.append(tf(im))
    grid = make_grid(torch.stack(tiles) * std + mean, nrow=n_view + 1, padding=4, pad_value=1.0)
    save_image(grid.clamp(0, 1), path)
    with open(path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        for i in pick:
            f.write(f"{rows[i]['name']}\t{caption_flags(rows[i]['caption'])}\t{rows[i]['caption']}\n")


# ---------------------------------------------------------------- đánh giá
@torch.no_grad()
def encode_all(split: ClipSplit, vis, txt, eot, device, bs=512):
    img, tx = [], []
    for s in range(0, len(vis), bs):
        img.append(F.normalize(split.visual_suffix(vis[s: s + bs].to(device).float()), dim=-1).cpu())
        tx.append(F.normalize(split.text_suffix(txt[s: s + bs].to(device).float(), eot[s: s + bs].to(device)), dim=-1).cpu())
    return torch.cat(img), torch.cat(tx)


@torch.no_grad()
def contrastive_loss(img_all, txt_all, idx, logit_scale, bs):
    """Loss InfoNCE đối xứng như lúc train, chia batch `bs` theo thứ tự cố định, bỏ batch lẻ cuối (giống train)."""
    idx_t = torch.tensor(idx)
    n_batch = max(1, len(idx_t) // bs)
    scale = float(logit_scale.exp())
    losses = []
    for s in range(n_batch):
        b = idx_t[s * bs: (s + 1) * bs]
        logits = scale * img_all[b] @ txt_all[b].T
        target = torch.arange(len(b))
        losses.append(float((F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2))
    return float(np.mean(losses))


@torch.no_grad()
def fit_report(img_all, txt_all, idx, sub_train, logit_scale, bs):
    """So train với val CÙNG kích thước và CÙNG cách đo (ảnh không tăng cường) -> khoảng cách train/val (overfit)."""
    out = {}
    for name, ids in (("train", sub_train), ("val", idx["val"])):
        ids_t = torch.tensor(ids)
        img, txt = img_all[ids_t], txt_all[ids_t]
        m = {"t2i_R@1": recall_metrics(txt @ img.T, "t2i")["t2i_R@1"],
             "i2t_R@1": recall_metrics(img @ txt.T, "i2t")["i2t_R@1"],
             "loss": contrastive_loss(img_all, txt_all, ids, logit_scale, bs)}
        out[name] = m
    return out


def fmt_fit(f):
    t, v = f["train"], f["val"]
    return (f"[train/val cùng cỡ] loss {t['loss']:.3f}/{v['loss']:.3f} | "
            f"t2i R@1 {t['t2i_R@1']:.1f}/{v['t2i_R@1']:.1f} | i2t R@1 {t['i2t_R@1']:.1f}/{v['i2t_R@1']:.1f}")


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
    keys = [k for k in keys if k in base and k in ft]
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
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256, help="Batch càng lớn, càng nhiều mẫu âm cho loss contrastive")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=0.1,
                    help="Stochastic depth cho các block đang train (0 = tắt)")
    ap.add_argument("--warmup", type=float, default=0.1, help="Tỉ lệ số bước warmup")
    ap.add_argument("--patience", type=int, default=4, help="Dừng sớm nếu val không cải thiện sau N epoch")
    ap.add_argument("--select-by", choices=["rsum", "loss"], default="rsum",
                    help="Tiêu chí chọn checkpoint / dừng sớm trên val: tổng Recall (rsum) hoặc loss contrastive (loss)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resplit", action="store_true", help="Chia lại train/val/test thay vì dùng split.json có sẵn")
    ap.add_argument("--dedup", default=str(ROOT / "data" / "vietnam_architecture" / "dedup" / "dedup_groups.csv"),
                    help="Kết quả scripts/dedup_images.py (cờ giữ/bỏ ảnh trùng); không có file thì bỏ qua")
    ap.add_argument("--no-dedup", action="store_true", help="Không dùng kết quả khử trùng")
    ap.add_argument("--aug-views", type=int, default=0,
                    help="Số biến thể tăng cường tạo trước cho mỗi ảnh train (0 = tắt; 4 là hợp lý trên CPU)")
    ap.add_argument("--augment", action="store_true", help="Tăng cường trực tuyến mỗi bước (chậm, nên dùng GPU)")
    ap.add_argument("--aug-preview", action="store_true", help="Chỉ lưu ảnh minh họa tăng cường rồi thoát")
    ap.add_argument("--aug-policy", choices=POLICIES, default="default",
                    help="default: cấu hình chính; blip_geo: thêm RandAugment(2,5) hình học kiểu BLIP (thí nghiệm B)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Gradient checkpointing cho các block train (GPU ít VRAM, vd. 4 GB)")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 hỗn hợp khi train trên GPU (chỉ nhanh hơn với GPU có Tensor Core; T1200/GTX 16xx thì chậm hơn)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--batch-size-cache", type=int, default=64)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Chỉ dùng N ảnh đầu (thử nhanh)")
    ap.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá checkpoint có sẵn so với CLIP gốc")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / CKPT_NAME

    dedup = None
    if not args.no_dedup and Path(args.dedup).exists():
        dedup = load_dedup(Path(args.dedup))
        print(f"Khử trùng: {args.dedup}")
    elif not args.no_dedup:
        print("Chưa có kết quả khử trùng (chạy scripts/dedup_images.py); dùng toàn bộ ảnh, chia ngẫu nhiên theo ảnh.")
    rows, missing, dropped = load_pairs(Path(args.captions), Path(args.image_dir), dedup)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Cặp ảnh-caption: {len(rows)} (thiếu ảnh: {missing}, bỏ vì trùng: {dropped}) | thiết bị: {device}")

    split_path = out_dir / ("split.json" if not args.limit else f"split_limit{args.limit}.json")
    if split_path.exists() and not args.resplit:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        known = set(split["train"]) | set(split["val"]) | set(split["test"])
        new = [r["name"] for r in rows if r["name"] not in known]
        split["train"] += new  # ảnh mới thêm vào CSV sau lần chia trước -> cho vào train
        print(f"Dùng split có sẵn: {split_path}" + (f" (+{len(new)} ảnh mới vào train)" if new else ""))
        leaks = split_leaks(rows, split)
        if leaks:
            sys.exit(f"Split có sẵn có {len(leaks)} nhóm ảnh trùng/cùng cảnh nằm ở nhiều tập (rò rỉ train->val/test), "
                     f"vd. {next(iter(leaks))}. Chạy lại với --resplit.")
    else:
        split = make_split(rows, args.val_frac, args.test_frac, args.seed)
        split_path.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    pos = {r["name"]: i for i, r in enumerate(rows)}
    idx = {k: [pos[n] for n in v if n in pos] for k, v in split.items()}
    # kiểm tra chống rò rỉ: không ảnh nào / nhóm nào thuộc hai tập
    sets = {k: set(v) for k, v in idx.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert not split_leaks(rows, split), "Có nhóm ảnh trùng nằm ở nhiều tập"
    print(f"train {len(idx['train'])} | val {len(idx['val'])} | test {len(idx['test'])} | "
          f"{len({rows[i]['group'] for i in idx['train']})} nhóm train, không nhóm nào lẫn sang val/test")
    labels = relevance_labels(rows)
    # tập con train cố định, cùng cỡ với val, để so train/val công bằng
    sub_train = sorted(random.Random(args.seed).sample(idx["train"], min(len(idx["val"]), len(idx["train"]))))

    model, preprocess = clip.load(args.base_model, device="cpu", jit=False)
    model = model.float().to(device).eval()
    if args.aug_preview:
        save_aug_preview(rows, idx["train"], preprocess, out_dir / "aug_preview.png", seed=args.seed,
                         policy=args.aug_policy)
        print(f"Đã lưu {out_dir / 'aug_preview.png'}")
        return
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
    base_val["loss"] = contrastive_loss(img_all, txt_all, idx["val"], model.logit_scale, args.batch_size)
    base_train_loss = contrastive_loss(img_all, txt_all, idx["train"], model.logit_scale, args.batch_size)
    print(f"CLIP gốc  | loss train {base_train_loss:.4f} val {base_val['loss']:.4f} | val : {fmt(base_val)}")
    base_fit = fit_report(img_all, txt_all, idx, sub_train, model.logit_scale, args.batch_size)
    print(f"CLIP gốc  | {fmt_fit(base_fit)}")

    if args.eval_only:
        if not ckpt_path.exists():
            sys.exit(f"Chưa có checkpoint {ckpt_path}")
        ck = load_checkpoint(model, ckpt_path)
        if ck.get("unfreeze_blocks") != args.unfreeze_blocks:
            sys.exit(f"Checkpoint dùng --unfreeze-blocks {ck.get('unfreeze_blocks')}, hãy chạy lại với giá trị đó")
        img_all, txt_all = encode_all(split_m, vis, txt, eot, device)
        print(f"Fine-tuned| {fmt_fit(fit_report(img_all, txt_all, idx, sub_train, model.logit_scale, args.batch_size))}")
        ft_test = evaluate(img_all, txt_all, idx["test"], labels)
        print_table(base_test, ft_test)
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

    # Tăng cường: chỉ dựng cho chỉ số train; val/test và đánh giá luôn dùng cache preprocess gốc (vis)
    aug_ds, aug_mm, tpos = None, None, None
    if args.augment:
        aug_ds = ImageDataset([rows[i]["path"] for i in idx["train"]], train_transforms(rows, idx["train"], preprocess, args.aug_policy))
        tpos = {g: t for t, g in enumerate(idx["train"])}
        print(f"Tăng cường trực tuyến [{args.aug_policy}]: {describe_policy(args.aug_policy)}")
    elif args.aug_views > 0:
        aug_mm = load_or_build_aug_cache(split_m, rows, idx["train"], preprocess, device, args, out_dir,
                                         tuple(vis.shape[1:]))
        tpos = np.full(len(rows), -1, dtype=np.int64)
        tpos[idx["train"]] = np.arange(len(idx["train"]))
        print(f"Tăng cường offline: mỗi ảnh train có gốc + {args.aug_views} biến thể [{args.aug_policy}] | {describe_policy(args.aug_policy)}")

    # GPU có Tensor Core: --amp (fp16 hỗn hợp) cho nhanh và đỡ VRAM; trọng số và optimizer vẫn fp32
    use_amp = device.type == "cuda" and args.amp
    global GRAD_CHECKPOINT, DROP_PATH
    GRAD_CHECKPOINT = args.grad_checkpoint
    DROP_PATH = args.drop_path
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # điểm càng lớn càng tốt: rsum, hoặc -loss
    score = {"rsum": lambda v: v["rsum"], "loss": lambda v: -v["loss"]}[args.select_by]
    history, best, best_epoch, bad = [], score(base_val), 0, 0
    print(f"Train {args.epochs} epoch x {steps_per_epoch} bước, batch {args.batch_size}, lr {args.lr}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = train_idx[torch.randperm(len(train_idx))]
        aug_iter = None
        if aug_ds is not None:
            # nhiều worker biến đổi ảnh song song, chuẩn bị trước các batch trong lúc GPU đang train
            order = [tpos[i] for i in perm[: steps_per_epoch * args.batch_size].tolist()]
            aug_iter = iter(torch.utils.data.DataLoader(
                torch.utils.data.Subset(aug_ds, order), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=device.type == "cuda", prefetch_factor=2 if args.workers else None))
        for s in range(steps_per_epoch):
            b = perm[s * args.batch_size: (s + 1) * args.batch_size]
            if aug_iter is not None:
                with torch.no_grad(), amp:
                    x = next(aug_iter).to(device, non_blocking=True)
                    # chạy phần đóng băng theo khối nhỏ để không tràn VRAM (GPU 4 GB)
                    vh = torch.cat([split_m.visual_prefix(c) for c in x.split(64)]).float()
            elif aug_mm is not None:
                # mỗi ảnh chọn ngẫu nhiên: 0 = bản gốc, 1..V = biến thể
                bn = b.numpy()
                view = np.random.randint(0, args.aug_views + 1, size=len(bn))
                tp = tpos[bn]
                assert (tp >= 0).all(), "ảnh ngoài tập train lọt vào batch tăng cường"
                vh = vis[b].float()
                m = view > 0
                if m.any():
                    vh[torch.from_numpy(m)] = torch.from_numpy(
                        np.ascontiguousarray(aug_mm[view[m] - 1, tp[m]])).float()
                vh = vh.to(device)
            else:
                vh = vis[b].to(device).float()
            with amp:
                img = F.normalize(split_m.visual_suffix(vh), dim=-1)
                tx = F.normalize(split_m.text_suffix(txt[b].to(device).float(), eot[b].to(device)), dim=-1)
            logits = model.logit_scale.exp() * img.float() @ tx.float().T
            loss = (contrastive_ce(logits) + contrastive_ce(logits.T)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(decay + no_decay, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            losses.append(loss.item())
            print(f"\r  epoch {epoch} bước {s + 1}/{steps_per_epoch} loss {np.mean(losses):.4f}", end="", flush=True)
        model.eval()
        img_all, txt_all = encode_all(split_m, vis, txt, eot, device)
        val = evaluate(img_all, txt_all, idx["val"], labels)
        val["loss"] = contrastive_loss(img_all, txt_all, idx["val"], model.logit_scale, args.batch_size)
        fit = fit_report(img_all, txt_all, idx, sub_train, model.logit_scale, args.batch_size)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val, "fit": fit,
                        "seconds": time.time() - t0})
        mark = ""
        if score(val) > best:
            best, best_epoch, bad, mark = score(val), epoch, 0, "  * lưu checkpoint"
            save_checkpoint(model, names, ckpt_path, args, {"epoch": epoch, "val": val})
        else:
            bad += 1
        print(f"\r  epoch {epoch} loss {np.mean(losses):.4f} val loss {val['loss']:.4f} "
              f"({time.time() - t0:.0f}s) | val: {fmt(val)}{mark}")
        print(f"           {fmt_fit(fit)}")
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
            "n_dropped_dup": dropped,
            "args": {k: v for k, v in vars(args).items()}}
    save_checkpoint(model, names, ckpt_path, args, meta)
    (out_dir / "metrics.json").write_text(json.dumps({
        "base_model": args.base_model, "unfreeze_blocks": args.unfreeze_blocks, "best_epoch": best_epoch,
        "pretrained": {"val": base_val, "test": base_test, "train_loss": base_train_loss, "fit": base_fit},
        "finetuned": {"test": ft_test},
        "val_pair_cosine": pos_stats, "history": history, "args": vars(args),
        "data": {"n_pairs": len(rows), "n_dropped_dup": dropped, "dedup": dedup is not None,
                 "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"])},
        "augmentation": {"mode": "online" if args.augment else ("offline" if args.aug_views else "none"),
                         "views": args.aug_views, "policy": describe_policy(args.aug_policy) if (args.augment or args.aug_views)
                         else "", "applied_to": "train"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCheckpoint: {ckpt_path} (epoch {best_epoch})")
    print(f"Kết quả  : {out_dir / 'metrics.json'}")
    print("Dùng trong backend: đặt CLIP_FINETUNED_PATH trong backend/.env rồi chạy python scripts/reembed_images.py")


if __name__ == "__main__":
    main()
