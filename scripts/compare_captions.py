r"""
So sanh caption cua 2 VLM (Qwen3.6 va Gemini) bang embedding de chon 1 caption chuan cho CLIP.
  - tuong dong cao  -> tu dong lay caption cua model uu tien (--prefer, mac dinh Gemini)
  - tuong dong thap, hoac 2 caption noi 2 loai cong trinh khac nhau -> check thu cong

Cau truc thu muc:
    compare_captions.py
    caption_project_Qwen3.6/captions_clip.csv
    caption_project_Gemini/captions_clip.csv

Cai dat:   pip install sentence-transformers pandas

Cac buoc:
  1) python compare_captions.py
        Tinh embedding (lan dau tai model ~420MB), in bang phan bo tuong dong va tao trong compare_output/:
          similarity_all.csv      moi anh + do tuong dong, sap xep tu thap len cao
          manual_review.csv       cac anh can check thu cong
          calibration_sample.csv  ~100 anh rai deu cac muc tuong dong, de ban cham nhan
  2) (nen lam) Mo calibration_sample.csv, xem tung anh, dien cot ok:
          1 = caption_chon mo ta DUNG anh, 0 = co chi tiet sai
     roi chay:  python compare_captions.py --calibrate
        -> goi y nguong theo ti le loi ban chap nhan (--max-error, mac dinh 5%)
     Chay lai voi nguong moi:  python compare_captions.py --threshold 0.78
        (embedding da luu cache, chay lai chi vai giay; lua chon da dien trong manual_review.csv duoc giu nguyen)
  3) Mo manual_review.csv, voi moi dong dien cot pick:
          q = lay caption Qwen,  g = lay caption Gemini,  x = bo anh nay
     hoac tu viet caption vao cot final_caption (uu tien hon pick).
  4) python compare_captions.py --merge
        -> compare_output/captions_final.csv (filepath, caption) de train CLIP
     Neu 2 file CSV chay tren 2 may khac nhau (Windows D:\... va Mac /Users/...), chi dinh thu muc anh
     tren may se train:  python compare_captions.py --merge --image-dir "D:\caption_project\vietnam_architecture"
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ================== CAU HINH ==================
BASE_DIR = Path(__file__).resolve().parent
CSV_Q = BASE_DIR / "caption_project_Qwen3.6" / "captions_clip.csv"
CSV_G = BASE_DIR / "caption_project_Gemini" / "captions_clip.csv"
OUT_DIR = BASE_DIR / "compare_output"
MODEL = "sentence-transformers/all-mpnet-base-v2"   # tieng Anh, thang diem cosine trai rong, de dat nguong
N_RANDOM_ROUNDS = 5       # so vong ghep caption cua 2 anh KHAC nhau, de biet muc "khong lien quan"
CALIB_BINS, CALIB_PER_BIN = 10, 10
# ==============================================

SIM_ALL = OUT_DIR / "similarity_all.csv"
MANUAL = OUT_DIR / "manual_review.csv"
CALIB = OUT_DIR / "calibration_sample.csv"
FINAL = OUT_DIR / "captions_final.csv"
CACHE = OUT_DIR / "embeddings_cache.npz"
NAME = {"q": "Qwen", "g": "Gemini"}

# Nhom loai cong trinh. Loai chinh = tu khoa xuat hien DAU TIEN trong caption (prompt bat dau bang chu the chinh).
# Hai caption bi coi la mau thuan khi loai chinh khac nhau VA caption nay khong nhac toi loai cua caption kia.
TYPE_GROUPS = {
    "pagoda/temple": ["communal house", "pagoda", "temple", "shrine", "stupa", "cham tower"],
    "church": ["cathedral", "church", "chapel", "basilica", "bell tower"],
    "tomb": ["mausoleum", "tomb", "grave", "cemetery"],
    "house": ["shophouse", "stilt house", "house", "villa", "residence"],
    "bridge": ["bridge"],
    "statue": ["statue", "sculpture", "bust"],
    "drawing/map": ["map", "drawing", "painting", "poster", "illustration", "sketch", "diagram", "blueprint"],
}
_KW = sorted(((k, g) for g, ks in TYPE_GROUPS.items() for k in ks), key=lambda x: -len(x[0]))
_KW_GROUP = dict(_KW)
_KW_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _KW) + r")(?:es|s)?\b", re.I)


def type_groups(text: str):
    """Tra ve (loai chinh, tap cac loai duoc nhac toi)."""
    hits = [_KW_GROUP[m.group(1).lower()] for m in _KW_RE.finditer(text)]
    return (hits[0] if hits else ""), set(hits)


# ---------- doc du lieu ----------
def load_csv(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Khong thay {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).dropna(subset=["caption"])
    df["caption"] = df["caption"].str.strip()
    df = df[df["caption"] != ""]
    # Ghep 2 file theo TEN anh (duong dan 2 project co the khac nhau)
    df["file"] = df["filepath"].map(lambda p: re.split(r"[\\/]", str(p))[-1])
    df = df.drop_duplicates("file", keep="last")
    print(f"{NAME[tag]:<7}: {len(df)} caption  ({path})")
    return df.rename(columns={"filepath": f"path_{tag}", "caption": f"cap_{tag}"})[
        ["file", f"path_{tag}", f"cap_{tag}"]]


def embed(texts, model_name):
    """Embedding da chuan hoa (do dai 1) -> tich vo huong = cosine. Co cache de chay lai nhanh."""
    key = hashlib.md5((model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=False)
        if str(c["key"]) == key:
            print("Dung embedding da cache.")
            return c["emb"]
    from sentence_transformers import SentenceTransformer
    print(f"Tinh embedding {len(texts)} caption bang {model_name} ...")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.savez(CACHE, emb=emb, key=np.array(key))
    return emb


def write_csv(df, path, excel=True):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig" if excel else "utf-8")
    except PermissionError:
        sys.exit(f"Khong ghi duoc {path.name} (dang mo bang Excel?). Dong file roi chay lai.")


def read_labels(path, cols):
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)[cols]


# ---------- buoc 1: so sanh ----------
def compare(args):
    OUT_DIR.mkdir(exist_ok=True)
    df = load_csv(Path(args.qwen), "q").merge(load_csv(Path(args.gemini), "g"), on="file", how="outer")
    both = df.dropna(subset=["cap_q", "cap_g"]).reset_index(drop=True)
    single = df[df["cap_q"].isna() | df["cap_g"].isna()]
    n = len(both)
    print(f"Co ca 2 caption: {n} anh | chi 1 caption: {len(single)} anh")
    if n < 2:
        sys.exit("Qua it anh chung giua 2 file, kiem tra lai ten file anh.")

    emb = embed(both["cap_q"].tolist() + both["cap_g"].tolist(), args.model)
    eq, eg = emb[:n], emb[n:]
    both["similarity"] = (eq * eg).sum(1).round(4)

    # Muc "khong lien quan": caption Qwen cua anh i so voi caption Gemini cua anh j khac
    rng = np.random.default_rng(0)
    neg = []
    for _ in range(N_RANDOM_ROUNDS):
        perm = rng.permutation(n)
        keep = perm != np.arange(n)
        neg.append((eq[keep] * eg[perm[keep]]).sum(1))
    neg = np.concatenate(neg)

    sim = both["similarity"].to_numpy()
    q = lambda a, p: np.quantile(a, p)
    print("\n=== PHAN BO DO TUONG DONG (cosine) ===")
    print(f"Cung 1 anh      : min {sim.min():.2f} | p5 {q(sim, .05):.2f} | p25 {q(sim, .25):.2f} "
          f"| median {np.median(sim):.2f} | p75 {q(sim, .75):.2f}")
    print(f"2 anh khac nhau : median {np.median(neg):.2f} | p95 {q(neg, .95):.2f} "
          f"| p99 {q(neg, .99):.2f} | p99.9 {q(neg, .999):.2f}")
    below_random = int((sim < q(neg, .99)).sum())
    print(f"-> {below_random} anh co 2 caption KHONG giong nhau hon 2 anh ngau nhien (duoi p99): gan nhu chac chan co loi.")

    if args.threshold == "auto":
        thr = round(float(q(neg, .999)), 2)
        print(f"\nNguong tu dong = {thr:.2f} (p99.9 cua cap 2 anh khac nhau). "
              "Nen hieu chinh lai bang --calibrate.")
    else:
        thr = float(args.threshold)
        print(f"\nNguong = {thr:.2f}")

    print("\nNguong | so anh phai check (do tuong dong thap)")
    for t in sorted(set(np.round(np.arange(0.50, 0.96, 0.05), 2)) | {round(thr, 2)}):
        k = int((sim < t).sum())
        print(f"  {t:.2f} | {k:5d} ({100 * k / n:4.1f}%)" + ("   <- dang dung" if abs(t - thr) < 1e-9 else ""))

    # Mau thuan loai cong trinh
    tq = both["cap_q"].map(type_groups)
    tg = both["cap_g"].map(type_groups)
    both["type_q"] = [m for m, _ in tq]
    both["type_g"] = [m for m, _ in tg]
    both["type_conflict"] = [bool(a and b and a != b and a not in sb and b not in sa)
                             for (a, sa), (b, sb) in zip(tq, tg)]
    # Caption bi lot ten truong JSON (vd: "a sign displays visible_text")
    leak = lambda col: both[col].str.contains(r"visible_text|image_type", case=False)
    both["leak"] = leak("cap_q") | leak("cap_g")
    low = both["similarity"] < thr
    both["status"] = np.where(low | both["type_conflict"] | both["leak"], "manual", "auto")
    n_conf_only = int((both["type_conflict"] & ~low).sum())
    n_manual = int((both["status"] == "manual").sum())
    print(f"\nKhac loai cong trinh (du tuong dong cao): them {n_conf_only} anh")
    print(f"Caption lot chu 'visible_text'/'image_type': {int(both['leak'].sum())} anh")
    print(f"TONG: tu dong {n - n_manual} anh | check thu cong {n_manual} + {len(single)} anh chi co 1 caption")

    cols = ["file", "similarity", "status", "type_conflict", "type_q", "type_g", "cap_q", "cap_g", "path_q", "path_g"]
    write_csv(both.sort_values("similarity")[cols], SIM_ALL)

    # Danh sach check thu cong
    m = both[both["status"] == "manual"].copy()
    m["reason"] = [
        "; ".join(filter(None, [f"tuong dong thap {s:.2f}" if s < thr else "",
                                f"khac loai: {a} vs {b}" if c else "",
                                "caption lot chu visible_text" if lk else ""]))
        for s, c, a, b, lk in zip(m["similarity"], m["type_conflict"], m["type_q"], m["type_g"], m["leak"])]
    s = single.copy()
    s["similarity"] = np.nan
    s["reason"] = np.where(s["cap_q"].isna(), "chi Gemini co caption", "chi Qwen co caption")
    rev = pd.concat([m, s], ignore_index=True)
    rev["filepath"] = rev["path_g"].fillna(rev["path_q"])
    rev = rev.rename(columns={"cap_q": "caption_qwen", "cap_g": "caption_gemini"})
    rev["pick"], rev["final_caption"] = "", ""
    old = read_labels(MANUAL, ["file", "pick", "final_caption"])
    if old is not None:        # giu lai nhung gi ban da dien
        old = old.drop_duplicates("file").set_index("file")
        for c in ("pick", "final_caption"):
            rev[c] = rev["file"].map(old[c]).fillna("")
    rev = rev.sort_values("similarity", na_position="last")
    write_csv(rev[["file", "similarity", "reason", "caption_qwen", "caption_gemini",
                   "pick", "final_caption", "filepath"]], MANUAL)

    # Mau hieu chinh nguong (chi tao 1 lan de khong mat nhan da cham)
    if not CALIB.exists():
        both["bin"] = pd.qcut(both["similarity"].rank(method="first"), CALIB_BINS, labels=False)
        parts = []
        for _, g in both.groupby("bin"):
            k = min(CALIB_PER_BIN, len(g))
            p = g.sample(k, random_state=0).copy()
            p["bin_weight"] = round(len(g) / k, 3)
            parts.append(p)
        c = pd.concat(parts).sort_values("similarity")
        pref, other = args.prefer, "q" if args.prefer == "g" else "g"
        c = c.assign(caption_chon=c[f"cap_{pref}"], caption_kia=c[f"cap_{other}"],
                     filepath=c[f"path_{pref}"], ok="")
        write_csv(c[["file", "similarity", "bin_weight", "caption_chon", "caption_kia", "filepath", "ok"]], CALIB)
        print(f"\nDa tao {CALIB.name}: caption_chon = caption cua {NAME[pref]} (se duoc lay tu dong).")

    print(f"\nKet qua trong {OUT_DIR}")


# ---------- buoc 2: hieu chinh nguong ----------
def calibrate(args):
    c = read_labels(CALIB, ["similarity", "bin_weight", "ok"])
    if c is None:
        sys.exit("Chua co calibration_sample.csv, hay chay buoc 1 truoc.")
    c = c[c["ok"].str.strip().isin(["0", "1"])].astype(float)
    if len(c) < 30:
        sys.exit(f"Moi cham {len(c)} dong. Can it nhat ~30 dong (tot nhat ca 100) de uoc luong.")
    all_sim = pd.read_csv(SIM_ALL, encoding="utf-8-sig")["similarity"].to_numpy()
    w, ok, s = c["bin_weight"].to_numpy(), c["ok"].to_numpy(), c["similarity"].to_numpy()
    print(f"Da cham {len(c)} anh, {int((ok == 0).sum())} anh co caption sai.")
    print("\nNguong | % loi trong nhom tu dong | so anh phai check | so mau tren nguong")
    best = None
    for t in np.round(np.arange(0.30, 0.96, 0.01), 2):
        above = s >= t
        if above.sum() < 10:
            break
        err = float((w * (1 - ok))[above].sum() / w[above].sum())
        k = int((all_sim < t).sum())
        if best is None and err <= args.max_error:
            best = t
        if round(t * 100) % 5 == 0 or t == best:
            print(f"  {t:.2f} | {100 * err:5.1f}% | {k:5d} ({100 * k / len(all_sim):4.1f}%) | {int(above.sum())}"
                  + ("   <- goi y" if t == best else ""))
    if best is None:
        print(f"\nKhong nguong nao dat loi <= {100 * args.max_error:.0f}%: loi khong phu thuoc do tuong dong. "
              "Nen doi --prefer hoac check thu cong nhieu hon.")
    else:
        print(f"\nGoi y: python compare_captions.py --threshold {best:.2f}"
              f"   (loi uoc tinh <= {100 * args.max_error:.0f}% trong nhom tu dong)")


# ---------- buoc 4: gop ket qua ----------
def merge(args):
    if not SIM_ALL.exists():
        sys.exit("Chua co similarity_all.csv, hay chay buoc 1 truoc.")
    a = pd.read_csv(SIM_ALL, encoding="utf-8-sig", dtype=str)
    a = a[a["status"] == "auto"]
    p = args.prefer
    rows = [(r["file"], r[f"path_{p}"] if isinstance(r[f"path_{p}"], str) else r[f"path_{'q' if p == 'g' else 'g'}"],
             r[f"cap_{p}"]) for _, r in a.iterrows()]
    n_auto = len(rows)

    m = read_labels(MANUAL, ["file", "caption_qwen", "caption_gemini", "pick", "final_caption", "filepath"])
    n_drop = n_todo = 0
    for _, r in (m.iterrows() if m is not None else []):
        pick, own = r["pick"].strip().lower(), r["final_caption"].strip()
        cap = own or {"q": r["caption_qwen"], "g": r["caption_gemini"]}.get(pick, "")
        if pick == "x" and not own:
            n_drop += 1
        elif cap.strip():
            rows.append((r["file"], r["filepath"], cap.strip()))
        else:
            n_todo += 1
    out = pd.DataFrame(rows, columns=["file", "filepath", "caption"]).drop_duplicates("file", keep="last")
    if args.image_dir:   # doi duong dan sang thu muc anh tren may se train
        out["filepath"] = [str(Path(args.image_dir) / f) for f in out["file"]]
    out = out[["filepath", "caption"]]
    write_csv(out, FINAL, excel=False)
    print(f"Tu dong ({NAME[p]}): {n_auto} | chon tay: {len(out) - n_auto} | bo: {n_drop} "
          f"| CHUA review (khong dua vao): {n_todo}")
    print(f"=> {len(out)} anh trong {FINAL}")


def main():
    ap = argparse.ArgumentParser(description="So sanh caption 2 VLM bang embedding")
    ap.add_argument("--qwen", default=str(CSV_Q))
    ap.add_argument("--gemini", default=str(CSV_G))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threshold", default="auto", help="nguong cosine, hoac 'auto'")
    ap.add_argument("--prefer", choices=["q", "g"], default="g", help="caption lay khi tu dong (mac dinh Gemini)")
    ap.add_argument("--calibrate", action="store_true", help="goi y nguong tu calibration_sample.csv da cham")
    ap.add_argument("--max-error", type=float, default=0.05, help="ti le loi chap nhan khi --calibrate")
    ap.add_argument("--merge", action="store_true", help="gop ket qua thanh captions_final.csv")
    ap.add_argument("--image-dir", default="", help="dung voi --merge: thu muc anh tren may train, "
                                                    "filepath = image-dir + ten file")
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args)
    elif args.merge:
        merge(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()