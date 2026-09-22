"""
Hai prompt cho pipeline sinh và kiểm tra caption.

    VLM 1 (sinh caption)  : chỉ nhận ẢNH + CAPTION_PROMPT
    VLM 2 (đối chiếu)     : nhận ẢNH + build_verify_prompt(caption, metadata)

Nên dùng hai mô hình KHÁC NHAU cho VLM 1 và VLM 2 để lỗi không trùng nhau.
Dùng chung với metadata do crawl_commons.py tạo ra.
"""

import json
import re


# ============================ PROMPT 1: SINH CAPTION ============================
# Văn bản gửi cho VLM 1 (kèm ảnh). Dùng trực tiếp, không cần điền gì thêm.

CAPTION_PROMPT = """You are writing training captions for a CLIP image-text model.
The images are photographs related to architecture in Vietnam.

Describe ONLY what is clearly visible in the image.

Rules:
- Write one caption in English, 15 to 40 words, one or two sentences.
- Start directly with the main subject. Do not write "This image shows",
  "A photo of", or similar phrases.
- Mention, when visible: type of structure (pagoda, temple, communal house,
  tomb, gate, colonial building, shophouse, house, church, bridge, etc.),
  roof shape, main materials, dominant colors, notable architectural details,
  and the surroundings.
- If the image shows an interior, a statue, a drawing, or a map instead of a
  building exterior, say so plainly.
- Do NOT guess the name, city, or date of the building. Only include a name
  if it is written clearly and legibly in the image.
- Do not describe mood, beauty, or history. No words like "beautiful",
  "majestic", "ancient", "historic".
- If something is uncertain, leave it out rather than guess.

Return ONLY valid JSON, with no markdown and no extra text:
{
  "caption": "...",
  "image_type": "exterior | interior | detail | statue | drawing_or_map | other",
  "visible_text": "any legible text in the image, or empty string"
}"""
# ---------------------------- HẾT PROMPT 1 ----------------------------


# ============================ PROMPT 2: ĐỐI CHIẾU ============================
# Đây chỉ là KHUÔN MẪU: <<CAPTION>> và <<METADATA_ITEMS>> là ô trống.
# KHÔNG gửi trực tiếp biến này. Hãy gọi build_verify_prompt() để có prompt hoàn chỉnh.

VERIFY_PROMPT_TEMPLATE = """You are verifying an image caption against the image and its metadata
from Wikimedia Commons. Metadata was written by humans and may be
incomplete or partly wrong.

CAPTION:
<<CAPTION>>

METADATA ITEMS:
<<METADATA_ITEMS>>

Task 1: For EACH metadata item, choose exactly one label:
- "match": the caption expresses this item, and it is visible in the image.
- "not_mentioned": the item is visible in the image, but the caption does
  not mention it.
- "contradiction": the caption states something that conflicts with this item.
- "not_verifiable": the item cannot be judged from the image
  (names, locations, dates, photographers, historical facts).

Task 2: List any statements in the caption that are wrong compared to
what is actually in the image, regardless of the metadata.

Be strict about contradictions and hallucinations, but do not penalize the
caption for omitting minor details. Compare meaning, not exact wording:
"Buddhist temple" matches "pagoda"; "tiled roof" matches "terracotta roof".

Return ONLY valid JSON, with no markdown and no extra text:
{
  "items": [
    {"item": "...", "label": "match | not_mentioned | contradiction | not_verifiable",
     "reason": "short reason"}
  ],
  "caption_errors": ["..."],
  "overall": "good | minor_issues | wrong"
}"""
# ---------------------------- HẾT PROMPT 2 ----------------------------


# ===================== HÀM ĐIỀN DỮ LIỆU VÀO PROMPT 2 =====================
# Code Python chạy trên máy, KHÔNG gửi cho VLM.

# Bước phụ: metadata JSON -> danh sách mục (dùng bên trong build_verify_prompt).
def build_metadata_items(meta, max_desc_chars=300):
    """Chuyển metadata thành danh sách đánh số để đưa vào prompt 2."""
    items = []
    items += [f"Category: {c}" for c in meta.get("categories", [])]
    items += [f"Depicts: {d}" for d in meta.get("depicts", [])]
    if meta.get("commons_caption_en"):
        items.append(f"Commons caption: {meta['commons_caption_en']}")
    elif meta.get("commons_caption_vi"):
        items.append(f"Commons caption (Vietnamese): {meta['commons_caption_vi']}")
    if meta.get("description"):
        items.append(f"Description: {meta['description'][:max_desc_chars]}")
    return items


# Dùng hàm này để lấy prompt 2 hoàn chỉnh, rồi gửi kết quả cho VLM 2 (kèm ảnh).
def build_verify_prompt(caption, meta):
    """
    Trả về prompt 2 đã điền caption và metadata.
    Trả về None nếu ảnh không có metadata nào để đối chiếu
    (đưa thẳng ảnh đó vào nhóm kiểm tra thủ công).
    """
    items = build_metadata_items(meta)
    if not items:
        return None
    numbered = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))
    return (VERIFY_PROMPT_TEMPLATE
            .replace("<<CAPTION>>", caption)
            .replace("<<METADATA_ITEMS>>", numbered))


# ============================ XỬ LÝ KẾT QUẢ ============================
# Code Python chạy trên máy, KHÔNG gửi cho VLM.
# Dùng SAU KHI đã nhận câu trả lời từ VLM.

# Dùng cho cả VLM 1 và VLM 2: đổi câu trả lời dạng chữ thành dict Python.
def parse_json_response(text):
    """Đọc JSON từ phản hồi VLM, kể cả khi mô hình lỡ bọc trong ```json```."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# Chỉ dùng cho VLM 2: tính tỉ lệ khớp, số mâu thuẫn, số lỗi caption.
def score_verification(result):
    """
    Tính điểm từ kết quả VLM 2.
    ti_le_khop bỏ qua các mục 'not_verifiable'; None nghĩa là không có mục nào
    kiểm chứng được -> nên đưa vào nhóm kiểm tra thủ công.
    """
    labels = [x.get("label") for x in result.get("items", [])]
    usable = [l for l in labels if l != "not_verifiable"]
    return {
        "ti_le_khop": labels.count("match") / len(usable) if usable else None,
        "so_khop": labels.count("match"),
        "so_khong_nhac_toi": labels.count("not_mentioned"),
        "so_mau_thuan": labels.count("contradiction"),
        "so_khong_kiem_chung": labels.count("not_verifiable"),
        "so_loi_caption": len(result.get("caption_errors", [])),
        "overall": result.get("overall"),
    }


# Dùng kết quả của score_verification để chia nhóm kiểm tra.
def needs_manual_review(score, min_ratio=0.6):
    """Quy tắc chia nhóm: True -> kiểm tra thủ công, False -> nhóm lấy mẫu ngẫu nhiên."""
    return (
        score["ti_le_khop"] is None
        or score["so_mau_thuan"] > 0
        or score["so_loi_caption"] > 0
        or score["overall"] != "good"
        or score["ti_le_khop"] < min_ratio
    )