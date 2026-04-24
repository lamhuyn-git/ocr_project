"""
keyword_matcher.py — Lớp 1: Xác định text block nào là nhãn field (label)

Input : một đoạn text từ OCR
Output: tên field (vd: "ho_chu_dem_va_ten") hoặc None
"""

import re
import unicodedata
from typing import Dict, List, Optional

try:
    from rapidfuzz import fuzz
    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False
    print("⚠️  rapidfuzz chưa cài — dùng exact match. Chạy: pip install rapidfuzz")


# ── Keyword map: field → danh sách keyword có thể gặp trong OCR ──
# Mỗi field có nhiều biến thể vì OCR hay mất dấu, sai ký tự
KEYWORD_MAP: Dict[str, List[str]] = {
    "title": [
        "mẫu ct01 ban hành kèm theo thông tư số 66/2023/tt-bca ngày 17/11/2023 của bộ trưởng bộ công an",
        "ban hành kèm theo thông tư số 66/2023/tt-bca",
        "thông tư số 66/2023/tt-bca ngày 17/11/2023",
        "66/2023/tt-bca",
        "mẫu ct01", "mau ct01", "ct01",
    ],
    "kinh_gui": [
        "kính gửi", "kinh gui", "kính gui", "kinh gửi",
    ],
    "ho_chu_dem_va_ten": [
        "họ, chữ đệm và tên", "ho chu dem va ten",
        "họ và tên", "ho va ten",
        "họ, chữ đệm và tên:",
    ],
    "ngay_thang_nam_sinh": [
        "ngày, tháng, năm sinh", "ngay thang nam sinh",
        "ngày sinh", "ngay sinh",
        "ngày/tháng/năm sinh",
    ],
    "gioi_tinh": [
        "giới tính", "gioi tinh",
    ],
    "so_dinh_dan_ca_nhan": [
        # Keyword ngắn ("số định danh") để sau keyword dài hơn
        # tránh nhầm với so_dinh_dan_ca_nhan_cua_chu_ho
        "số định danh cá nhân", "so dinh danh ca nhan",
        "số định danh", "so dinh danh",
        "cccd", "số cccd", "so cccd",
    ],
    "so_dien_thoai": [
        "số điện thoại", "so dien thoai",
        "điện thoại", "dien thoai", "đt",
    ],
    "email": [
        "email", "e-mail", "thư điện tử",
    ],
    "ho_chu_dem_va_ten_chu_ho": [
        "họ, chữ đệm và tên chủ hộ", "ho chu dem va ten chu ho",
        "họ tên chủ hộ", "ho ten chu ho",
        "tên chủ hộ", "ten chu ho",
    ],
    "moi_quan_he_voi_chu_ho": [
        "mối quan hệ với chủ hộ", "moi quan he voi chu ho",
        "quan hệ với chủ hộ", "quan he voi chu ho",
        "quan hệ chủ hộ",
    ],
    "so_dinh_dan_ca_nhan_cua_chu_ho": [
        "số định danh cá nhân của chủ hộ",
        "so dinh danh ca nhan cua chu ho",
        "số định danh chủ hộ", "so dinh danh chu ho",
        "cccd chủ hộ",
    ],
    "noi_dung_de_nghi": [
        "nội dung đề nghị", "noi dung de nghi",
        "đề nghị", "de nghi",
        "nội dung", "noi dung",
    ],
}

FUZZY_THRESHOLD = 75


def normalize(text: str) -> str:
    """Lowercase, bỏ dấu câu cuối, chuẩn hóa khoảng trắng, NFC."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r'[:\(\)\[\]\.]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def match_keyword(text: str) -> Optional[str]:
    """
    Nhận 1 đoạn text, trả về tên field nếu là nhãn, None nếu không phải.

    Thứ tự ưu tiên:
    1. Exact match (nhanh)
    2. Fuzzy match (bắt OCR errors)

    Lưu ý: field có keyword dài (so_dinh_dan_ca_nhan_cua_chu_ho) được
    kiểm tra trước field keyword ngắn (so_dinh_dan_ca_nhan) để tránh nhầm.
    """
    norm = normalize(text)

    # Sắp xếp field theo độ dài keyword dài nhất (keyword dài → ít nhầm hơn)
    sorted_fields = sorted(
        KEYWORD_MAP.items(),
        key=lambda item: max(len(kw) for kw in item[1]),
        reverse=True
    )

    # Bước 1: Exact / substring match
    for field, keywords in sorted_fields:
        for kw in sorted(keywords, key=len, reverse=True):
            if normalize(kw) in norm:
                return field

    # Bước 2: Fuzzy match
    if _FUZZY_AVAILABLE:
        best_score, best_field = 0, None
        for field, keywords in sorted_fields:
            for kw in keywords:
                score = fuzz.partial_ratio(norm, normalize(kw))
                if score > best_score:
                    best_score = score
                    best_field = field
        if best_score >= FUZZY_THRESHOLD:
            return best_field

    return None
