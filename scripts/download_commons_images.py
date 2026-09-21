import os
import requests
import json
import time
import sys
from urllib.parse import unquote

# Đảm bảo in tiếng Việt và emoji không bị lỗi trên Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# URL của bạn (đã thêm format=json)
PETSCAN_URL = "https://petscan.wmcloud.org/?language=commons&project=wikimedia&depth=3&categories=Architecture+of+Vietnam&ns%5B6%5D=1&format=json&doit=1"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vietnam_architecture")

def get_petscan_results():
    print("⏳ Đang tải danh sách ảnh từ PetScan...")
    headers = {'User-Agent': 'KIS-Architect/1.0 (Contact: myemail@example.com)'}
    response = requests.get(PETSCAN_URL, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    try:
        # Lấy danh sách các trang (pages) từ JSON của PetScan
        pages = data['*'][0]['a']['*']
        titles = [page['title'] for page in pages if page.get('nstext') == 'File']
        print(f"✅ Đã tìm thấy {len(titles)} tệp ảnh kiến trúc Việt Nam.")
        return titles
    except Exception as e:
        print(f"❌ Lỗi khi phân tích JSON từ PetScan: {e}")
        return []

def is_valid_image(filepath):
    if not os.path.exists(filepath):
        return False
    # Kiểm tra kích thước và nội dung xem có phải trang lỗi HTML không
    if os.path.getsize(filepath) < 4096:
        with open(filepath, 'rb') as f:
            header = f.read(50)
            if b'<!DOCTYPE' in header or b'<html' in header or b'Error' in header:
                return False
    return True

import re

def clean_html(raw_html):
    """Loại bỏ thẻ HTML và chuẩn hóa khoảng trắng trong văn bản mô tả"""
    if not raw_html:
        return ""
    cleantext = re.sub(r'<.*?>', ' ', str(raw_html))
    return ' '.join(cleantext.split())

def download_images(titles, limit=5000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Ảnh và Metadata sẽ được lưu vào: {OUTPUT_DIR}")
    
    api_url = "https://commons.wikimedia.org/w/api.php"
    api_headers = {'User-Agent': 'KIS-Architect/1.0 (Contact: student@university.edu)'}
    download_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://commons.wikimedia.org/'
    }
    
    batch_size = 50
    downloaded = 0
    
    for i in range(0, min(len(titles), limit * 3), batch_size):
        if downloaded >= limit:
            break
            
        chunk_titles = titles[i:i + batch_size]
        query_titles = "|".join([f"File:{t}" for t in chunk_titles])
        
        params = {
            "action": "query",
            "titles": query_titles,
            "prop": "imageinfo|categories",
            "iiprop": "url|extmetadata",
            "cllimit": 50,
            "iiurlwidth": 800,  # Wikimedia khuyến nghị tải qua thumburl (800px)
            "format": "json"
        }
        
        try:
            res = requests.get(api_url, params=params, headers=api_headers, timeout=20)
            res.raise_for_status()
            pages = res.json().get('query', {}).get('pages', {})
            
            # Lưu trữ thông tin từng tệp: URL & Metadata
            file_data_map = {}
            for pid, pinfo in pages.items():
                raw_title = pinfo.get('title', '').replace('File:', '').strip()
                
                # Trích xuất categories
                categories = []
                for cat in pinfo.get('categories', []):
                    cat_name = cat.get('title', '').replace('Category:', '').strip()
                    # Bỏ qua các danh mục kỹ thuật nội bộ của Wikimedia
                    if not any(cat_name.startswith(p) for p in ['CC-', 'Files', 'Self-published', 'Assumed', 'PD-']):
                        categories.append(cat_name)
                
                # Trích xuất imageinfo & extmetadata
                img_url = None
                metadata = {}
                if 'imageinfo' in pinfo and len(pinfo['imageinfo']) > 0:
                    info = pinfo['imageinfo'][0]
                    img_url = info.get('thumburl') or info.get('url')
                    ext = info.get('extmetadata', {})
                    
                    # Bóc tách các trường giá trị cốt lõi
                    title_val = ext.get('ObjectName', {}).get('value') or raw_title.replace('_', ' ')
                    desc_val = clean_html(ext.get('ImageDescription', {}).get('value') or '')
                    date_val = clean_html(ext.get('DateTimeOriginal', {}).get('value') or ext.get('DateTime', {}).get('value') or '')
                    artist_val = clean_html(ext.get('Artist', {}).get('value') or '')
                    lat_val = ext.get('GPSLatitude', {}).get('value')
                    lon_val = ext.get('GPSLongitude', {}).get('value')
                    
                    metadata = {
                        "monument_name": title_val,
                        "description": desc_val,
                        "categories": categories,
                        "date": date_val,
                        "artist": artist_val,
                        "latitude": float(lat_val) if lat_val else None,
                        "longitude": float(lon_val) if lon_val else None,
                        "source_url": info.get('descriptionurl') or f"https://commons.wikimedia.org/wiki/File:{raw_title}"
                    }
                
                file_data_map[raw_title] = {
                    "url": img_url,
                    "metadata": metadata
                }
            
            for title in chunk_titles:
                if downloaded >= limit:
                    break
                    
                img_name = unquote(title).replace(" ", "_").replace("/", "-")
                base_name = os.path.splitext(img_name)[0]
                img_filepath = os.path.join(OUTPUT_DIR, img_name)
                meta_filepath = os.path.join(OUTPUT_DIR, f"{base_name}.json")
                
                finfo = file_data_map.get(title.replace('_', ' ')) or file_data_map.get(title) or {}
                img_url = finfo.get('url')
                metadata = finfo.get('metadata') or {}
                
                # Ghi tệp metadata JSON nếu có thông tin
                if metadata:
                    try:
                        with open(meta_filepath, 'w', encoding='utf-8') as mf:
                            json.dump(metadata, mf, ensure_ascii=False, indent=2)
                    except Exception as me:
                        print(f"⚠️ Không thể ghi metadata cho {img_name}: {me}")
                
                # Bỏ qua nếu ảnh đã tồn tại và hợp lệ
                if is_valid_image(img_filepath):
                    print(f"⏭️ {img_name} (Ảnh & Metadata đã sẵn sàng)")
                    downloaded += 1
                    continue
                elif os.path.exists(img_filepath):
                    os.remove(img_filepath)
                
                if not img_url:
                    continue
                    
                try:
                    print(f"[{downloaded+1}/{limit}] ⬇️ Đang tải {img_name}...")
                    img_resp = requests.get(img_url, headers=download_headers, timeout=30)
                    if img_resp.status_code == 200 and not img_resp.content.startswith(b'<!DOCTYPE') and not img_resp.content.startswith(b'<html'):
                        with open(img_filepath, 'wb') as f:
                            f.write(img_resp.content)
                        downloaded += 1
                        time.sleep(0.2)
                    elif img_resp.status_code == 429:
                        print(f"⚠️ Wikimedia 429 (Rate limited) khi tải {img_name}, chờ 2 giây...")
                        time.sleep(2)
                    else:
                        print(f"⚠️ Không tải được {img_name} (HTTP {img_resp.status_code})")
                except Exception as e:
                    print(f"❌ Lỗi khi tải dữ liệu ảnh {img_name}: {e}")
                    
        except Exception as e:
            print(f"❌ Lỗi khi truy vấn thông tin ảnh từ API: {e}")
            time.sleep(2)
            
    print(f"🎯 Đã hoàn thành xử lý {downloaded}/{limit} ảnh và metadata.")

if __name__ == "__main__":
    titles = get_petscan_results()
    if titles:
        # Tải thử 50 tấm đầu tiên để làm dữ liệu mẫu
        # Bạn có thể tăng biến limit=50 lên thành limit=500 hoặc len(titles) để tải toàn bộ
        download_images(titles, limit=50)
        print(f"🎉 Hoàn tất! Bạn có thể vào Web UI, chọn mục Nạp Kho Ảnh và chọn các tệp từ thư mục: {OUTPUT_DIR}")
