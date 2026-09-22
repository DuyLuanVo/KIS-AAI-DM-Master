"""
Tải ảnh và metadata kiến trúc Việt Nam từ Wikimedia Commons qua PetScan,
đồng thời lọc và kiểm tra chất lượng ảnh (trùng lặp, kích thước, tỉ lệ).

Cách dùng:
    python download_commons_images.py               # Tải theo LIMIT mặc định
    python download_commons_images.py --limit 200   # Tải số lượng ảnh chỉ định
    python download_commons_images.py --full        # Tải toàn bộ danh sách
    python download_commons_images.py --check-only  # Chỉ kiểm tra chất lượng ảnh hiện có
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time

import imagehash
import requests
from PIL import Image

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# --- Cấu hình ---

PETSCAN_URL = (
    "https://petscan.wmcloud.org/?language=commons&project=wikimedia&depth=3"
    "&categories=Architecture+of+Vietnam&ns%5B6%5D=1&format=json&doit=1"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "vietnam_architecture",
)

USER_AGENT = "KIS-Architect/1.0 (student research project)"

LIMIT = None              # Số lượng ảnh tải mặc định (None: tải toàn bộ)
BATCH_SIZE = 50         # Tối đa số tiêu đề trong mỗi request API
TARGET_SIZE = 640       # Cạnh dài nhất sau khi resize
THUMB_WIDTH = 960       # Chiều rộng thumbnail lấy từ Wikimedia
DOWNLOAD_DELAY = 3      # Giây nghỉ giữa các lần tải ảnh
MAX_RETRIES = 5
API_DELAY = 1.0         # Giây nghỉ giữa các lượt gọi API metadata

# Ngưỡng kiểm tra chất lượng ảnh
MIN_SHORT_SIDE = 224    # Cạnh ngắn tối thiểu
MAX_ASPECT = 2.5        # Tỉ lệ cạnh tối đa
DUP_DISTANCE = 4        # Ngưỡng khoảng cách Hamming của pHash để tính trùng lặp

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}

# Lọc bỏ các category kỹ thuật hoặc bản quyền không mô tả nội dung
NOISE_CATEGORY = re.compile(
    r"(taken with|photographs by|photos by|uploaded|wiki loves|quality images|"
    r"featured pictures|valued images|pages with|files with|files by|media needing|"
    r"media with|self-published|license|cc-by|cc-zero|pd-|gfdl|"
    r"images by|images from|flickr|panoramio|author|user:)",
    re.IGNORECASE,
)

RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://commons.wikimedia.org/",
}


# --- Tiện ích ---

def strip_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def clean_html(raw_html):
    """Loại bỏ thẻ HTML và chuẩn hóa khoảng trắng."""
    if not raw_html:
        return ""
    text = re.sub(r"<.*?>", " ", str(raw_html))
    return " ".join(text.split())


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_basename(title, pageid):
    """Tạo tên tệp an toàn gắn pageid để tránh trùng lặp."""
    base = os.path.splitext(title)[0]
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", base).replace(" ", "_")
    return f"{base[:120]}_{pageid}"


def is_noise_category(name):
    return bool(NOISE_CATEGORY.search(name))


def request_with_retry(url, params=None, timeout=30, headers=None):
    """GET kèm thử lại khi lỗi mạng, rate limit (429) hoặc lỗi 5xx."""
    req_headers = headers if headers is not None else session.headers
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers=req_headers)
        except requests.RequestException as e:
            wait = 5 * attempt
            print(f"[Cảnh báo] Lỗi mạng ({e.__class__.__name__}), thử lại sau {wait}s...", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(int(retry_after), 5)
                print(f"[Cảnh báo] HTTP {resp.status_code} (Server yêu cầu Retry-After: {retry_after}s), chờ {wait}s...", flush=True)
            else:
                wait = 5 * attempt
                print(f"[Cảnh báo] HTTP {resp.status_code}, chờ {wait}s rồi thử lại...", flush=True)
            time.sleep(wait)
            continue

        return resp
    return None


def get_json(url, params):
    resp = request_with_retry(url, params)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        print(f"[Cảnh báo] Phản hồi không phải JSON từ {url}")
        return None


# --- PetScan ---

def get_petscan_results():
    print("Đang tải danh sách ảnh từ PetScan...")
    resp = request_with_retry(PETSCAN_URL, timeout=300)
    if resp is None or resp.status_code != 200:
        print("[Lỗi] Không tải được kết quả PetScan.")
        return []

    try:
        pages = resp.json()["*"][0]["a"]["*"]
    except (ValueError, KeyError, IndexError) as e:
        print(f"[Lỗi] Phân tích JSON từ PetScan thất bại: {e}")
        return []

    titles = [p["title"].replace("_", " ") for p in pages if p.get("nstext") == "File"]
    images = [t for t in titles if os.path.splitext(t)[1].lower() in ALLOWED_EXT]
    print(f"PetScan: Tìm thấy {len(titles)} tệp ({len(images)} ảnh raster).")
    return images


# --- Wikimedia Commons API ---

def fetch_file_info(titles):
    """Lấy URL ảnh, extmetadata và categories (loại trừ category ẩn) theo nhóm."""
    base = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "titles": "|".join(f"File:{t}" for t in titles),
        "prop": "imageinfo|categories",
        "iiprop": "url|extmetadata",
        "iiurlwidth": THUMB_WIDTH,
        "iiextmetadatalanguage": "en",
        "cllimit": "max",
        "clshow": "!hidden",
    }
    params = dict(base)
    result = {}

    while True:
        data = get_json(COMMONS_API, params)
        if data is None:
            break

        for page in data.get("query", {}).get("pages", []):
            title = strip_prefix(page.get("title", ""), "File:")
            entry = result.setdefault(title, {"pageid": page.get("pageid"), "categories": []})

            if page.get("missing"):
                entry["missing"] = True

            if page.get("imageinfo"):
                info = page["imageinfo"][0]
                entry["url"] = info.get("thumburl") or info.get("url")
                entry["source_url"] = info.get("descriptionurl")
                entry["ext"] = info.get("extmetadata", {})

            for cat in page.get("categories", []):
                name = strip_prefix(cat.get("title", ""), "Category:")
                if name and not is_noise_category(name) and name not in entry["categories"]:
                    entry["categories"].append(name)

        if "continue" not in data:
            break
        params = {**base, **data["continue"]}

    return result


def fetch_structured_data(pageids):
    """Lấy structured data của Commons: caption và depicts (P180)."""
    ids = [f"M{pid}" for pid in pageids if pid]
    out = {}

    for i in range(0, len(ids), 50):
        data = get_json(COMMONS_API, {
            "action": "wbgetentities",
            "ids": "|".join(ids[i:i + 50]),
            "format": "json",
        })
        time.sleep(API_DELAY)
        if data is None:
            continue

        for mid, entity in data.get("entities", {}).items():
            if "missing" in entity:
                continue

            labels = entity.get("labels") or {}
            statements = entity.get("statements") or {}
            if not isinstance(labels, dict):
                labels = {}
            if not isinstance(statements, dict):
                statements = {}

            depicts = []
            for st in statements.get("P180", []):
                dv = st.get("mainsnak", {}).get("datavalue")
                if dv and isinstance(dv.get("value"), dict) and "id" in dv["value"]:
                    depicts.append(dv["value"]["id"])

            out[int(mid[1:])] = {
                "caption_en": labels.get("en", {}).get("value", ""),
                "caption_vi": labels.get("vi", {}).get("value", ""),
                "depicts_qids": depicts,
            }

    return out


_qid_cache = {}


def resolve_qids(qids):
    """Đổi Q-id Wikidata thành tên tiếng Anh / tiếng Việt (có cache)."""
    todo = [q for q in set(qids) if q not in _qid_cache]

    for i in range(0, len(todo), 50):
        data = get_json(WIKIDATA_API, {
            "action": "wbgetentities",
            "ids": "|".join(todo[i:i + 50]),
            "props": "labels",
            "languages": "en|vi",
            "format": "json",
        })
        time.sleep(API_DELAY)
        if data is None:
            continue
        for q, entity in data.get("entities", {}).items():
            labels = entity.get("labels") or {}
            _qid_cache[q] = {
                "en": labels.get("en", {}).get("value", ""),
                "vi": labels.get("vi", {}).get("value", ""),
            }

    return {q: _qid_cache.get(q, {"en": "", "vi": ""}) for q in qids}


# --- Xử lý Metadata ---

def build_metadata(title, info, sd, qlabels):
    ext = info.get("ext", {})

    def ev(key):
        return clean_html(ext.get(key, {}).get("value", ""))

    depicts_qids = sd.get("depicts_qids", [])
    depicts = []
    for q in depicts_qids:
        lab = qlabels.get(q, {})
        depicts.append(lab.get("en") or lab.get("vi") or q)

    return {
        # Thông tin phục vụ sinh caption & đối chiếu
        "categories": info.get("categories", []),
        "depicts": depicts,
        "commons_caption_en": sd.get("caption_en", ""),
        "commons_caption_vi": sd.get("caption_vi", ""),
        "description": ev("ImageDescription"),
        "monument_name": ev("ObjectName") or os.path.splitext(title)[0],

        # Thông tin lưu trữ và bản quyền
        "file_title": title,
        "pageid": info.get("pageid"),
        "depicts_qids": depicts_qids,
        "date": ev("DateTimeOriginal") or ev("DateTime"),
        "artist": ev("Artist"),
        "license": ev("LicenseShortName"),
        "license_url": ev("LicenseUrl"),
        "latitude": safe_float(ev("GPSLatitude")),
        "longitude": safe_float(ev("GPSLongitude")),
        "source_url": info.get("source_url")
        or f"https://commons.wikimedia.org/wiki/File:{title.replace(' ', '_')}",
    }


def meta_path_for(record):
    base = os.path.splitext(record["image_file"])[0]
    return os.path.join(OUTPUT_DIR, f"{base}.json")


def save_metadata(record):
    with open(meta_path_for(record), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# --- Xử lý Ảnh ---

def is_valid_image(path):
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def save_resized_png(content, path):
    """Mở ảnh, chuyển RGB (xử lý nền trong suốt nếu có), resize và lưu PNG."""
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception:
        return False

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    img.thumbnail((TARGET_SIZE, TARGET_SIZE), RESAMPLE)
    img.save(path, "PNG", optimize=True)
    return True


# --- Tải Ảnh & Metadata ---

def download_all(titles, limit=LIMIT):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    limit = len(titles) if limit is None else min(limit, len(titles))
    mode = "toàn bộ" if limit == len(titles) else "giới hạn"
    print(f"Thư mục lưu: {OUTPUT_DIR}")
    print(f"Chế độ {mode}: {limit}/{len(titles)} ảnh")

    done, failed = 0, 0
    records = []

    for i in range(0, len(titles), BATCH_SIZE):
        if done >= limit:
            break

        chunk = titles[i:i + BATCH_SIZE]
        infos = fetch_file_info(chunk)
        pageids = [v["pageid"] for v in infos.values() if v.get("pageid")]
        sd_map = fetch_structured_data(pageids)
        qlabels = resolve_qids({q for sd in sd_map.values() for q in sd["depicts_qids"]})

        for title in chunk:
            if done >= limit:
                break

            info = infos.get(title)
            if not info or info.get("missing") or not info.get("url"):
                print(f"[Bỏ qua] {title} (không có thông tin ảnh)")
                continue

            pageid = info["pageid"]
            meta = build_metadata(title, info, sd_map.get(pageid, {}), qlabels)

            base = safe_basename(title, pageid)
            img_path = os.path.join(OUTPUT_DIR, f"{base}.png")
            meta["image_file"] = os.path.basename(img_path)

            if is_valid_image(img_path):
                print(f"[Đã có] {base}.png")
            else:
                print(f"[{done + 1}/{limit}] Đang tải {title}...", flush=True)
                resp = request_with_retry(info["url"], timeout=60, headers=DOWNLOAD_HEADERS)
                if resp is None or resp.status_code != 200:
                    code = resp.status_code if resp is not None else "hết lượt thử"
                    print(f"[Lỗi] Không tải được {title} ({code})")
                    failed += 1
                    continue
                if not save_resized_png(resp.content, img_path):
                    print(f"[Lỗi] {title} không phải ảnh hợp lệ")
                    failed += 1
                    continue
                time.sleep(DOWNLOAD_DELAY)

            save_metadata(meta)
            records.append(meta)
            done += 1

    n = len(records) or 1
    has_cat = sum(1 for r in records if r["categories"])
    has_dep = sum(1 for r in records if r["depicts"])
    has_cap = sum(1 for r in records if r["commons_caption_en"] or r["commons_caption_vi"])
    has_desc = sum(1 for r in records if r["description"])

    print("\n--- Thống kê tải ---")
    print(f"   Tải thành công : {done}")
    print(f"   Thất bại       : {failed}")
    print(f"   Có categories  : {has_cat} ({has_cat / n:.0%})")
    print(f"   Có depicts     : {has_dep} ({has_dep / n:.0%})")
    print(f"   Có caption     : {has_cap} ({has_cap / n:.0%})")
    print(f"   Có description : {has_desc} ({has_desc / n:.0%})")

    return records


def load_existing_records():
    """Đọc metadata của các ảnh đã tải (dùng cho --check-only)."""
    records = []
    if not os.path.isdir(OUTPUT_DIR):
        return records
    for name in sorted(os.listdir(OUTPUT_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, name), encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError):
            continue
        if "image_file" in record and os.path.exists(os.path.join(OUTPUT_DIR, record["image_file"])):
            records.append(record)
    return records


# --- Kiểm Tra Chất Lượng ---

def check_quality(records):
    """Kiểm tra từng ảnh: kích thước, tỉ lệ, trùng lặp và gắn cờ quality_flags."""
    print(f"\nĐang kiểm tra chất lượng {len(records)} ảnh...")
    hashes = {}

    for i, r in enumerate(records, 1):
        r["quality_flags"] = []
        r["dup_of"] = ""
        path = os.path.join(OUTPUT_DIR, r["image_file"])
        try:
            with Image.open(path) as im:
                w, h = im.size
                phash = imagehash.phash(im)
        except Exception as e:
            r["quality_flags"].append(f"loi_doc_file ({e.__class__.__name__})")
            continue

        r["width"], r["height"] = w, h
        r["aspect"] = round(max(w, h) / min(w, h), 2)
        r["phash"] = str(phash)
        hashes[id(r)] = int(str(phash), 16)

        if min(w, h) < MIN_SHORT_SIDE:
            r["quality_flags"].append("qua_nho")
        if r["aspect"] > MAX_ASPECT:
            r["quality_flags"].append("ti_le_dai")

        if i % 500 == 0:
            print(f"   ... {i}/{len(records)}", flush=True)

    # Giữ lại ảnh có độ phân giải cao nhất trong nhóm trùng lặp
    candidates = [r for r in records if id(r) in hashes]
    candidates.sort(key=lambda r: r["width"] * r["height"], reverse=True)
    kept = []
    for r in candidates:
        for k in kept:
            if bin(hashes[id(r)] ^ hashes[id(k)]).count("1") <= DUP_DISTANCE:
                r["dup_of"] = k["image_file"]
                r["quality_flags"].append("trung_lap")
                break
        else:
            kept.append(r)

    for r in records:
        r["is_clean"] = not r["quality_flags"]
        save_metadata(r)


def write_outputs(records):
    """Ghi metadata_all.jsonl, quality_report.csv, clean_list.txt và in thống kê."""
    all_path = os.path.join(OUTPUT_DIR, "metadata_all.jsonl")
    report_path = os.path.join(OUTPUT_DIR, "quality_report.csv")
    clean_path = os.path.join(OUTPUT_DIR, "clean_list.txt")

    with open(all_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fields = ["image_file", "width", "height", "aspect", "phash", "dup_of", "flags"]
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda r: r["image_file"]):
            row = {k: r.get(k, "") for k in fields}
            row["flags"] = ";".join(r.get("quality_flags", []))
            writer.writerow(row)

    clean = sorted(r["image_file"] for r in records if r.get("is_clean"))
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write("\n".join(clean))

    def count(flag):
        return sum(1 for r in records if any(x.startswith(flag) for x in r.get("quality_flags", [])))

    print("\n--- Thống kê chất lượng ---")
    print(f"   Tổng số ảnh    : {len(records)}")
    print(f"   Trùng lặp      : {count('trung_lap')}")
    print(f"   Quá nhỏ        : {count('qua_nho')}")
    print(f"   Tỉ lệ quá dài  : {count('ti_le_dai')}")
    print(f"   Lỗi đọc file   : {count('loi_doc_file')}")
    print(f"   Đạt yêu cầu    : {len(clean)}")
    print(f"\n   Metadata tổng hợp : {all_path}")
    print(f"   Báo cáo chất lượng: {report_path}")
    print(f"   Danh sách sạch    : {clean_path}")


# --- Chương trình chính ---

def parse_args():
    parser = argparse.ArgumentParser(description="Tải ảnh và metadata từ Wikimedia Commons")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Tải toàn bộ danh sách PetScan")
    group.add_argument("--limit", type=int, help="Số lượng ảnh muốn tải")
    group.add_argument("--check-only", action="store_true",
                       help="Không tải, chỉ kiểm tra chất lượng ảnh hiện có")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit phải lớn hơn 0")
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.check_only:
        records = load_existing_records()
        if not records:
            print(f"[Lỗi] Không tìm thấy ảnh nào trong {OUTPUT_DIR}")
            sys.exit(1)
    else:
        limit = None if args.full else (args.limit if args.limit is not None else LIMIT)
        titles = get_petscan_results()
        if not titles:
            sys.exit(1)
        records = download_all(titles, limit=limit)

    if records:
        check_quality(records)
        write_outputs(records)
        print(f"\nHoàn tất! Dữ liệu được lưu tại: {OUTPUT_DIR}")