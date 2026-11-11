"""
So sánh nhiều lần fine-tune (vd. cấu hình chính và thí nghiệm B) từ metrics.json.
Chọn cấu hình theo VAL (rsum tại epoch được chọn); số liệu TEST chỉ để báo cáo, không dùng để chọn.

    python scripts/compare_runs.py models/clip_finetuned_v2 models/clip_finetuned_v2_B
"""
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

KEYS = ["t2i_R@1", "t2i_R@5", "i2t_R@1", "t2i_full_R@1", "t2i_full_R@10", "i2i_P@10", "i2i_mAP@10", "rsum"]


def main():
    runs = []
    for d in sys.argv[1:]:
        m = json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))
        be = m["best_epoch"]
        h = next(x for x in m["history"] if x["epoch"] == be)
        gap = h["fit"]["train"]["t2i_R@1"] - h["fit"]["val"]["t2i_R@1"] if "fit" in h else float("nan")
        runs.append({"name": Path(d).name, "policy": m["args"].get("aug_policy", "default"), "epoch": be,
                     "val": h["val"], "test": m["finetuned"]["test"], "base": m["pretrained"]["test"], "gap": gap,
                     "split": m["data"]})
    sizes = {json.dumps(r["split"], sort_keys=True) for r in runs}
    if len(sizes) > 1:
        print("CẢNH BÁO: các lần chạy có số ảnh train/val/test khác nhau, so sánh không công bằng")
    w = max(len(r["name"]) for r in runs) + 2
    print("\n=== Chọn theo VAL ===")
    print(f"{'lần chạy':<{w}}{'policy':>10}{'epoch':>7}{'val rsum':>10}{'val t2i R@1':>13}{'gap train-val':>15}")
    for r in runs:
        print(f"{r['name']:<{w}}{r['policy']:>10}{r['epoch']:>7}{r['val']['rsum']:>10.1f}{r['val']['t2i_R@1']:>13.1f}"
              f"{r['gap']:>15.1f}")
    best = max(runs, key=lambda r: r["val"]["rsum"])
    print(f"-> chọn: {best['name']} ({best['policy']})")
    print("\n=== TEST (chỉ báo cáo) ===")
    print(f"{'chỉ số':<15}{'CLIP gốc':>10}" + "".join(f"{r['name']:>{w}}" for r in runs))
    for k in KEYS:
        print(f"{k:<15}{runs[0]['base'][k]:>10.1f}" + "".join(f"{r['test'][k]:>{w}.1f}" for r in runs))


if __name__ == "__main__":
    main()
