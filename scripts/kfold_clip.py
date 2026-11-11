"""
Đánh giá chéo k-fold cho cấu hình fine-tune hiện tại của finetune_clip.py.

Cách chia: các nhóm (sau khử trùng bằng dedup_images.py mỗi ảnh là một nhóm riêng, nên thực chất là chia theo ảnh)
được phân đều vào 10 phần theo số ảnh. Fold i (i = 0..k-1): test = phần 2i, val = phần 2i+1, train = 8 phần còn lại
-> mỗi fold có tỉ lệ 80/10/10 giống lần chạy chính và test của các fold KHÔNG chồng nhau
   (finetune_clip.py vẫn kiểm tra không nhóm nào nằm ở hai tập khi chạy).
Siêu tham số giữ cố định (không chỉnh theo kết quả k-fold), epoch được chọn trên val của từng fold.

Đầu ra trong --out-dir:
  * fold{i}/ (split.json, metrics.json, checkpoint, train.log)
  * kfold_summary.json : chỉ số test từng fold + trung bình ± độ lệch chuẩn (CLIP gốc và fine-tuned)

Ví dụ:
    python scripts/kfold_clip.py -- --device cuda --augment --grad-checkpoint --workers 12
    (các tham số sau "--" được chuyển nguyên cho finetune_clip.py; fold đã có metrics.json thì bỏ qua)
"""
import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_clip import ROOT, load_dedup, load_pairs, split_leaks  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KEYS = ["t2i_R@1", "i2t_R@1", "t2i_R@5", "i2t_R@5", "t2i_full_R@1", "i2i_P@10", "i2i_mAP@10", "rsum"]


def make_parts(rows, n_parts: int, seed: int):
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["name"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))  # nhóm lớn xếp trước để các phần cân bằng
    parts = [[] for _ in range(n_parts)]
    for g in keys:
        min(parts, key=len).extend(groups[g])
    return parts


def summarize(out_dir: Path, k: int):
    folds = []
    for i in range(k):
        p = out_dir / f"fold{i}" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            be = m["best_epoch"]
            fit = next((h["fit"] for h in m["history"] if h.get("epoch") == be and "fit" in h), None)
            folds.append({"fold": i, "best_epoch": be, "base": m["pretrained"]["test"], "ft": m["finetuned"]["test"],
                          "gap_t2i_R@1": None if fit is None else fit["train"]["t2i_R@1"] - fit["val"]["t2i_R@1"]})
    if not folds:
        return None

    def ms(vals):
        return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    summ = {"n_folds_done": len(folds), "folds": folds,
            "base": {key: ms([f["base"][key] for f in folds]) for key in KEYS if key in folds[0]["base"]},
            "ft": {key: ms([f["ft"][key] for f in folds]) for key in KEYS},
            "gap_t2i_R@1": ms([f["gap_t2i_R@1"] for f in folds if f["gap_t2i_R@1"] is not None])}
    print(f"\n{'chỉ số':<14}" + "".join(f"{'fold' + str(f['fold']):>8}" for f in folds) +
          f"{'TB ± std (FT)':>17}{'CLIP gốc':>15}")
    for key in KEYS:
        b = summ["base"].get(key)
        print(f"{key:<14}" + "".join(f"{f['ft'][key]:>8.1f}" for f in folds) +
              f"{summ['ft'][key]['mean']:>10.1f} ± {summ['ft'][key]['std']:<4.1f}" +
              (f"{b['mean']:>9.1f} ± {b['std']:.1f}" if b else ""))
    g = summ["gap_t2i_R@1"]
    print(f"{'gap train-val':<14}" + "".join(f"{f['gap_t2i_R@1']:>8.1f}" for f in folds) +
          f"{g['mean']:>10.1f} ± {g['std']:<4.1f}")
    (out_dir / "kfold_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    return summ


def main():
    argv = sys.argv[1:]
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description="Group k-fold cho fine-tune CLIP")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "kfold"))
    ap.add_argument("--captions", default=str(ROOT / "scripts" / "compare_output" / "captions_final.csv"))
    ap.add_argument("--image-dir", default=str(ROOT / "data" / "vietnam_architecture"))
    ap.add_argument("--dedup", default=str(ROOT / "data" / "vietnam_architecture" / "dedup" / "dedup_groups.csv"))
    ap.add_argument("--prefix-cache", default=str(ROOT / "models" / "clip_finetuned" / "cache" /
                                                  "prefix_ViT-B-32_k4.pt"),
                    help="Cache phần đóng băng dùng lại cho mọi fold (không phụ thuộc cách chia)")
    ap.add_argument("--keep-cache", action="store_true", help="Giữ cache trong từng fold sau khi chạy xong")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--folds", default="", help="Chỉ chạy các fold này, vd. 0,1,2 (để chia fold cho nhiều máy)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)

    if args.summary_only:
        summarize(out_dir, args.k)
        return

    rows, _, _ = load_pairs(Path(args.captions), Path(args.image_dir), load_dedup(Path(args.dedup)))
    parts = make_parts(rows, 2 * args.k, args.seed)
    print(f"{len(rows)} ảnh -> {2 * args.k} phần: {[len(p) for p in parts]}")
    tests = set()
    run_folds = [int(x) for x in args.folds.split(",")] if args.folds else list(range(args.k))
    for i in range(args.k):
        fold_dir = out_dir / f"fold{i}"
        if i not in run_folds:
            continue
        if (fold_dir / "metrics.json").exists():
            print(f"fold {i}: đã xong, bỏ qua")
            continue
        split = {"test": parts[2 * i], "val": parts[2 * i + 1],
                 "train": [n for j, p in enumerate(parts) if j not in (2 * i, 2 * i + 1) for n in p]}
        assert not split_leaks(rows, split), "Có nhóm nằm ở nhiều tập"
        assert not (tests & set(split["test"])), "Test các fold bị chồng nhau"
        tests |= set(split["test"])
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
        src = Path(args.prefix_cache)
        dst = fold_dir / "cache" / src.name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
        cmd = [sys.executable, str(ROOT / "scripts" / "finetune_clip.py"), "--captions", args.captions,
               "--image-dir", args.image_dir, "--dedup", args.dedup, "--seed", str(args.seed),
               "--out-dir", str(fold_dir), *passthrough]
        print(f"\n===== fold {i}/{args.k - 1}: train {len(split['train'])} | val {len(split['val'])} | "
              f"test {len(split['test'])}", flush=True)
        with open(fold_dir / "train.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            sys.exit(f"fold {i} lỗi (exit {rc}), xem {fold_dir / 'train.log'}")
        if not args.keep_cache:
            shutil.rmtree(fold_dir / "cache", ignore_errors=True)
        summarize(out_dir, args.k)
    summarize(out_dir, args.k)


if __name__ == "__main__":
    main()
