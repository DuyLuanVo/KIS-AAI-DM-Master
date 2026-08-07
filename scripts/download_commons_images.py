import os
import requests
import json
import time
from urllib.parse import unquote

# URL của bạn (đã thêm format=json)
PETSCAN_URL = "https://petscan.wmcloud.org/?language=commons&project=wikimedia&depth=4&categories=Architecture+of+Vietnam&ns%5B6%5D=1&format=json&doit=1"
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

def download_images(titles, limit=50):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Ảnh sẽ được lưu vào: {OUTPUT_DIR}")
    
    api_url = "https://commons.wikimedia.org/w/api.php"
    headers = {'User-Agent': 'KIS-Architect/1.0 (Contact: myemail@example.com)'}
    
    downloaded = 0
    for title in titles:
        if downloaded >= limit:
            print(f"🎯 Đã đạt giới hạn tải xuống ({limit} ảnh). Dừng lại.")
            break
            
        file_title = f"File:{title}"
        params = {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        
        try:
            res = requests.get(api_url, params=params, headers=headers)
            res.raise_for_status()
            pages = res.json().get('query', {}).get('pages', {})
            
            for page_id, page_info in pages.items():
                if 'imageinfo' in page_info:
                    img_url = page_info['imageinfo'][0]['url']
                    # Xử lý tên file an toàn
                    img_name = unquote(title)
                    img_name = img_name.replace(" ", "_").replace("/", "-")
                    
                    filepath = os.path.join(OUTPUT_DIR, img_name)
                    if not os.path.exists(filepath):
                        print(f"[{downloaded+1}/{limit}] ⬇️ Đang tải {img_name}...")
                        img_data = requests.get(img_url, headers={'User-Agent': 'KIS-Architect/1.0'}).content
                        with open(filepath, 'wb') as f:
                            f.write(img_data)
                        downloaded += 1
                        time.sleep(0.5)  # Tránh gửi quá nhiều request cùng lúc (Rate limit)
                    else:
                        print(f"⏭️ Bỏ qua {img_name} (đã tồn tại)")
                        
        except Exception as e:
            print(f"❌ Lỗi khi tải {title}: {e}")

if __name__ == "__main__":
    titles = get_petscan_results()
    if titles:
        # Tải thử 50 tấm đầu tiên để làm dữ liệu mẫu
        # Bạn có thể tăng biến limit=50 lên thành limit=500 hoặc len(titles) để tải toàn bộ
        download_images(titles, limit=50)
        print(f"🎉 Hoàn tất! Bạn có thể vào Web UI, chọn mục Nạp Kho Ảnh và chọn các tệp từ thư mục: {OUTPUT_DIR}")
