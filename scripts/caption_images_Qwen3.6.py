"""
Sinh caption anh kien truc bang Qwen3.6-35B-A3B (UD-Q3_K_S) chay tren llama-server local.

Cau truc thu muc (dat script cung cho voi):
    caption_images.py
    clean_list.txt            <- moi dong 1 ten file anh
    vietnam_architecture/     <- thu muc chua anh

Cach chay:
    python caption_images.py --test 3            thu 3 anh dau danh sach
    python caption_images.py --test 200 --random thu 200 anh ngau nhien -> test_captions.csv
    python caption_images.py                     chay tat ca anh con lai
    python caption_images.py --limit 2000        chi chay 2000 anh trong lan nay
    python caption_images.py --save-every 200    cap nhat file CSV moi 200 anh (mac dinh 500)
    python caption_images.py --export            chi xuat lai CSV tu ket qua da co

CACH LUU KET QUA:
    captions.jsonl        ghi NGAY sau moi anh -> mat dien/tat may cung chi mat toi da 1 anh.
                          Chay lai se tu bo qua anh da xong va lam tiep.
    captions_full.csv     }  cap nhat moi --save-every anh, khi xong, va khi bam Ctrl+C.
    captions_clip.csv     }  Luon chua TOAN BO ket qua tu truoc toi gio (ca cac lan chay truoc).
    captions_backup.jsonl ban sao cua captions.jsonl, tao moi lan cap nhat CSV.
    missing_files.txt     ten co trong clean_list.txt nhung khong tim thay anh.
"""
import argparse
import base64
import csv
import io
import json
import locale
import os
import re
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import requests
from PIL import Image, ImageOps

# ================== CAU HINH ==================
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "vietnam_architecture"
LIST_FILE = BASE_DIR / "clean_list.txt"
OUT_JSONL = BASE_DIR / "captions.jsonl"
BACKUP_JSONL = BASE_DIR / "captions_backup.jsonl"
OUT_FULL_CSV = BASE_DIR / "captions_full.csv"
OUT_CLIP_CSV = BASE_DIR / "captions_clip.csv"
MISSING_FILE = BASE_DIR / "missing_files.txt"

SERVER = "http://127.0.0.1:8080"
SAVE_EVERY = 500     # cap nhat CSV moi N anh
MAX_SIDE = 768       # canh dai anh gui len (768 ~ 260 token anh). 640 nhanh hon, it chi tiet hon
MAX_RETRIES = 4      # so lan sinh toi da moi anh neu caption vi pham rule.
                     # Sinh lai rat re (~3s, vi anh da nam san trong cache cua server).
TIMEOUT = 180        # giay cho 1 request. Binh thuong ~10s.
                     # Qua moc nay coi nhu server bi treo -> cho server hoi phuc roi gui lai anh do.
# ==============================================

Image.MAX_IMAGE_PIXELS = None

CAPTION_PROMPT = """You are writing training captions for a CLIP image-text model. The images are photographs related to architecture in Vietnam.
Describe ONLY what is clearly visible in the image.
Rules:

* Write one caption in English, 15 to 40 words, one or two sentences.
* Start directly with the main subject. Do not write "This image shows", "A photo of", "A view of", "An exterior view of", "An interior view of", or similar phrases.
* Mention, when visible: type of structure (pagoda, temple, communal house, tomb, gate, colonial building, shophouse, house, church, bridge, etc.), roof shape, main materials, dominant colors, notable architectural details, and the surroundings.
* If the image shows an interior, a statue, a drawing, or a map instead of a building exterior, say so plainly.
* Do NOT guess the name, city, or date of the building. Only include a name if it is written clearly and legibly in the image.
* Do not describe mood, beauty, or history. No words like "beautiful", "majestic", "ancient", "historic", "lush", "vibrant", "grand", "solemn", "unique", "festive", "picturesque", "serene", "charming", "impressive", "elegant", "stunning", "magnificent".
* Quote text in the caption only if it is short (up to 5 words) and fully legible. Otherwise just mention "a sign" or "an inscription". Put all legible text in visible_text.
* If something is uncertain, leave it out rather than guess.

image_type meanings: exterior = outside of a building or structure; interior = inside a building; detail = close-up of one part of a structure (roof ridge, carving, door, wall, pillar); statue = a statue or sculpture is the main subject; drawing_or_map = drawing, painting, map, poster, or printed page; other = the main subject is not architecture (animals, fish, aquariums, plants, landscapes, food, objects, people, vehicles).

Return ONLY valid JSON, with no markdown and no extra text: { "caption": "...", "image_type": "exterior | interior | detail | statue | drawing_or_map | other", "visible_text": "any legible text in the image, or empty string" }"""

IMAGE_TYPES = ["exterior", "interior", "detail", "statue", "drawing_or_map", "other"]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "image_type": {"type": "string", "enum": IMAGE_TYPES},
        "visible_text": {"type": "string"},
    },
    "required": ["caption", "image_type", "visible_text"],
}

# ===== Bo loc chat luong caption: vi pham -> sinh lai (toi da MAX_RETRIES lan) =====
# Tu cam (khong phan biet hoa thuong, chi bat ca tu tron ven)
BANNED_WORDS = ["beautiful", "majestic", "ancient", "historic", "stunning", "magnificent",
                "vibrant", "lush", "picturesque", "serene", "charming", "impressive", "elegant",
                "solemn", "grand", "unique", "festive"]
# Cach mo dau bi cam (so sanh voi phan dau caption, chu thuong)
BAD_STARTS = ("this image", "this photo", "a photo of", "an image of", "the image",
              "the photo", "a picture of", "a photograph of",
              "a close-up view of", "a view of", "view of")
# Bat them cac kieu "An exterior view of", "Interior view of", "The view through", "A wide view of"...
# ("Interior of ..." va "Close-up of ..." van hop le vi noi ro loai anh.)
BAD_START_RE = re.compile(
    r"^(a |an |the )?((close-up|wide|exterior|interior|indoor|aerial|side|front|top) )*view (of|through)\b")

session = requests.Session()   # giu ket noi mo, bot tre moi request


# ---------- doc danh sach anh ----------
def read_text_any_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def resolve_image(name: str):
    p = Path(name)
    candidates = [p] if p.is_absolute() else [IMAGE_DIR / p, BASE_DIR / p, IMAGE_DIR / p.name]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_list():
    if not LIST_FILE.exists():
        sys.exit(f"Khong thay {LIST_FILE}")
    if not IMAGE_DIR.is_dir():
        sys.exit(f"Khong thay thu muc anh {IMAGE_DIR}")
    names, seen = [], set()
    for line in read_text_any_encoding(LIST_FILE).splitlines():
        name = line.strip().strip('"').strip()
        if name and not name.startswith("#") and name not in seen:
            seen.add(name)
            names.append(name)
    found, missing = [], []
    for n in names:
        p = resolve_image(n)
        (found if p else missing).append((n, p))
    if missing:
        MISSING_FILE.write_text("\n".join(n for n, _ in missing), encoding="utf-8")
    print(f"clean_list.txt: {len(names)} ten, tim thay {len(found)} anh, thieu {len(missing)}"
          + (f" (xem {MISSING_FILE.name})" if missing else ""))
    return found


# ---------- goi model ----------
def encode_image(path: Path) -> str:
    with Image.open(path) as im:
        im.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))   # JPEG lon: giai ma nhanh o do phan giai thap
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"Khong thay JSON trong output: {text[:200]}")
    return json.loads(m.group(0))


def validate(d: dict) -> list:
    problems = []
    cap = str(d.get("caption", "")).strip()
    n_words = len(cap.split())
    if not 15 <= n_words <= 40:
        problems.append(f"caption {n_words} tu")
    low = cap.lower()
    if low.startswith(BAD_STARTS) or BAD_START_RE.search(low):
        problems.append("mo dau sai")
    hits = [w for w in BANNED_WORDS if re.search(rf"\b{w}\b", low)]
    if hits:
        problems.append(f"tu bi cam {hits}")
    if d.get("image_type") not in IMAGE_TYPES:
        problems.append(f"image_type la: {d.get('image_type')}")
    return problems


def caption_one(path: Path):
    b64 = encode_image(path)
    payload = {
        "model": "qwen3.6",
        # Prompt co dinh nam o system (giong nhau moi anh) de server co the tai dung cache
        "messages": [
            {"role": "system", "content": CAPTION_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Caption this image."},
            ]},
        ],
        "cache_prompt": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "caption", "schema": JSON_SCHEMA},
        },
    }
    best, problems = None, ["chua chay"]
    attempts, stalls = 0, 0
    while attempts < MAX_RETRIES:
        try:
            r = session.post(f"{SERVER}/v1/chat/completions", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
        except (requests.Timeout, requests.ConnectionError) as e:
            # Server treo / tat: KHONG tinh la 1 lan thu. Cho server hoi phuc roi gui lai.
            stalls += 1
            if stalls >= 3:   # cung 1 anh lam treo 3 lan -> bo qua, lan chay sau se thu lai
                return None, [f"server treo 3 lan voi anh nay: {type(e).__name__}"]
            wait_for_server(e)
            continue
        except Exception as e:
            problems = [f"loi: {e!r}"[:300]]
            attempts += 1
            time.sleep(2)
            continue
        attempts += 1
        try:
            d = extract_json(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            problems = [f"loi JSON: {e!r}"[:300]]
            continue
        problems = validate(d)
        if not problems:
            return d, []
        best = d
    return best, problems


def wait_for_server(err):
    """Server khong tra loi: bao nguoi dung va cho toi khi server song lai."""
    print("\n" + "!" * 70)
    print(f"  Server khong phan hoi ({type(err).__name__}) luc {time.strftime('%H:%M:%S')}.")
    print("  Kiem tra cua so llama-server:")
    print("   - Neu tieu de cua so bat dau bang 'Select' -> bam Esc trong cua so do.")
    print("     (Click/boi den vao cua so CMD se lam Windows TAM DUNG chuong trinh.)")
    print("   - Neu server da tat -> chay lai start_server.bat.")
    print("  Script se tu dong chay tiep khi server hoat dong lai.")
    print("!" * 70)
    while True:
        try:
            if session.get(f"{SERVER}/health", timeout=10).status_code == 200:
                print(f"  >> Server da hoat dong lai ({time.strftime('%H:%M:%S')}), gui lai anh.\n")
                return
        except requests.RequestException:
            pass
        time.sleep(15)


def check_server():
    try:
        r = session.get(f"{SERVER}/health", timeout=10)
        if r.status_code != 200:
            sys.exit(f"Server chua san sang (HTTP {r.status_code}). Doi model load xong roi chay lai.")
    except requests.ConnectionError:
        sys.exit("Khong ket noi duoc llama-server. Hay chay start_server.bat truoc.")


# ---------- luu ket qua ----------
def load_records() -> dict:
    """Doc captions.jsonl, ban ghi sau cua cung 1 anh ghi de ban truoc."""
    latest = {}
    if OUT_JSONL.exists():
        with OUT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    latest[rec["file"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass    # dong cuoi bi ghi do dang khi mat dien -> bo qua
    return latest


def export_csv():
    """Viet lai 2 file CSV tu TOAN BO captions.jsonl (gom ca cac lan chay truoc)."""
    if not OUT_JSONL.exists():
        return
    recs = load_records()
    try:
        shutil.copyfile(OUT_JSONL, BACKUP_JSONL)
        n_ok = 0
        with OUT_FULL_CSV.open("w", newline="", encoding="utf-8-sig") as ff, \
             OUT_CLIP_CSV.open("w", newline="", encoding="utf-8") as fc:
            wf, wc = csv.writer(ff), csv.writer(fc)
            wf.writerow(["file", "caption", "image_type", "visible_text", "issues"])
            wc.writerow(["filepath", "caption"])
            for rec in recs.values():
                wf.writerow([rec["file"], rec.get("caption", ""), rec.get("image_type", ""),
                             rec.get("visible_text", ""), "; ".join(rec.get("issues", []))])
                if rec.get("caption") and not rec.get("issues"):
                    wc.writerow([rec.get("path", ""), rec["caption"]])
                    n_ok += 1
        print(f"  >> Da cap nhat CSV: {n_ok} anh dat chuan / {len(recs)} anh da xu ly "
              f"({time.strftime('%H:%M:%S')})")
    except PermissionError:
        print("  >> [CANH BAO] Khong ghi duoc CSV (dang mo bang Excel?). "
              "captions.jsonl van an toan, se thu lai o lan cap nhat sau.")


def run_test(items, n, use_random):
    """Chay thu N anh, luu vao test_captions.csv. KHONG dung toi captions.jsonl,
    nen khong anh huong tien do khi chay that."""
    import random
    from collections import Counter
    sample = random.Random(42).sample(items, min(n, len(items))) if use_random else items[:n]
    rows, times, types = [], [], Counter()
    try:
        for i, (name, path) in enumerate(sample, 1):
            t = time.time()
            try:
                d, problems = caption_one(path)
            except Exception as e:
                d, problems = None, [f"loi anh: {e!r}"[:300]]
            dt = time.time() - t
            times.append(dt)
            d = d or {}
            types[d.get("image_type", "?")] += 1
            rows.append([name, f"{dt:.1f}", d.get("caption", ""), len(d.get("caption", "").split()),
                         d.get("image_type", ""), d.get("visible_text", ""), "; ".join(problems)])
            flag = "  <-- " + "; ".join(problems) if problems else ""
            print(f"[{i}/{len(sample)}] {dt:4.1f}s | {d.get('image_type', '?'):<14} | "
                  f"{d.get('caption', '')}{flag}")
    except KeyboardInterrupt:
        print("\nDa dung, luu cac anh da chay.")
    out = BASE_DIR / "test_captions.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "seconds", "caption", "words", "image_type", "visible_text", "issues"])
        w.writerows(rows)
    if rows:
        n_bad = sum(1 for r in rows if r[6])
        avg = sum(times[1:]) / max(len(times) - 1, 1)     # bo anh dau (khoi dong cache)
        print(f"\n===== TONG KET {len(rows)} anh =====")
        print(f"Toc do trung binh: {avg:.1f} s/anh -> 7762 anh ~ {7762 * avg / 3600:.1f} gio")
        print(f"Anh con vi pham sau khi sinh lai: {n_bad} ({100 * n_bad / len(rows):.0f}%)")
        print("image_type:", dict(types))
        print(f"Chi tiet: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="chi thu N anh, in ra man hinh")
    ap.add_argument("--limit", type=int, default=0, help="chi chay toi da N anh trong lan nay")
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY, help="cap nhat CSV moi N anh")
    ap.add_argument("--export", action="store_true", help="chi xuat lai CSV roi thoat")
    ap.add_argument("--random", action="store_true", help="dung voi --test: chon anh ngau nhien")
    args = ap.parse_args()

    if args.export:
        export_csv()
        return

    check_server()
    items = load_list()

    if args.test:
        run_test(items, args.test, args.random)
        return

    done = {k for k, r in load_records().items() if r.get("caption")}
    todo = [(n, p) for n, p in items if n not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Da xong {len(done)}, lan nay se chay {len(todo)} anh. "
          f"CSV cap nhat moi {args.save_every} anh. Ctrl+C de dung an toan.")
    if not todo:
        export_csv()
        return

    t0 = time.time()
    recent = deque(maxlen=20)       # toc do trung binh 20 anh gan nhat
    with OUT_JSONL.open("a", encoding="utf-8") as out:
        try:
            for i, (name, path) in enumerate(todo, 1):
                t = time.time()
                try:
                    d, problems = caption_one(path)
                except Exception as e:                    # anh hong, khong mo duoc...
                    d, problems = None, [f"loi anh: {e!r}"[:300]]
                rec = {"file": name, "path": str(path), **(d or {}), "issues": problems}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()                               # luu ngay tung anh
                recent.append(time.time() - t)

                if i % 10 == 0 or i == len(todo):
                    spi = sum(recent) / len(recent)
                    eta_h = (len(todo) - i) * spi / 3600
                    print(f"[{i}/{len(todo)}] {spi:.1f} s/anh, con khoang {eta_h:.1f} gio "
                          f"(da chay {(time.time() - t0) / 3600:.1f} gio)")
                if i % args.save_every == 0:
                    os.fsync(out.fileno())
                    export_csv()
        except KeyboardInterrupt:
            print("\nDa dung. Ket qua da luu, chay lai se lam tiep.")
        finally:
            out.flush()
            os.fsync(out.fileno())

    export_csv()


if __name__ == "__main__":
    main()