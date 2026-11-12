"""
Bước 1 của pipeline - thu thập: tải ảnh (thumbnail rộng 960 px) và metadata kiến trúc Việt Nam
từ Wikimedia Commons qua PetScan + MediaWiki API. Ảnh được lưu nguyên dạng (PNG, giữ EXIF);
mọi xử lý ảnh nằm ở bước 2: scripts/preprocess_images.py (chuẩn hóa, lọc chất lượng) và scripts/dedup_images.py.

Cách dùng:
    python download_commons_images.py               # Tải theo LIMIT mặc định
    python download_commons_images.py --limit 200   # Tải số lượng ảnh chỉ định
    python download_commons_images.py --full        # Tải toàn bộ danh sách
"""

import argparse
import io
import json
import os
import re
import sys
import time

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
THUMB_WIDTH = 960       # Chiều rộng thumbnail lấy từ Wikimedia
DOWNLOAD_DELAY = 3      # Giây nghỉ giữa các lần tải ảnh
MAX_RETRIES = 5
API_DELAY = 1.0         # Giây nghỉ giữa các lượt gọi API metadata


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


def save_png(content, path):
    """Lưu ảnh tải về thành PNG, không xử lý gì thêm (giữ EXIF để bước tiền xử lý xoay đúng chiều)."""
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception:
        return False
    if img.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):  # vd. CMYK: PNG không hỗ trợ
        img = img.convert("RGB")
    exif = img.info.get("exif")
    img.save(path, "PNG", **({"exif": exif} if exif else {}))
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
                if not save_png(resp.content, img_path):
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


def write_outputs(records):
    """Ghi metadata_all.jsonl (metadata của mọi ảnh đã tải)."""
    all_path = os.path.join(OUTPUT_DIR, "metadata_all.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n   Metadata tổng hợp : {all_path}")
    print("   Bước tiếp theo    : python scripts/preprocess_images.py")


# --- Chương trình chính ---

def parse_args():
    parser = argparse.ArgumentParser(description="Tải ảnh và metadata từ Wikimedia Commons")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Tải toàn bộ danh sách PetScan")
    group.add_argument("--limit", type=int, help="Số lượng ảnh muốn tải")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit phải lớn hơn 0")
    return args


if __name__ == "__main__":
    args = parse_args()

    limit = None if args.full else (args.limit if args.limit is not None else LIMIT)
    titles = get_petscan_results()
    if not titles:
        sys.exit(1)
    records = download_all(titles, limit=limit)

    if records:
        write_outputs(records)
        print(f"\nHoàn tất! Dữ liệu được lưu tại: {OUTPUT_DIR}")