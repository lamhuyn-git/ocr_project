"""
validator.py — Lớp 3: Validate và clean giá trị từng field

Input : tên field + raw text từ OCR
Output: giá trị đã chuẩn hóa, hoặc None nếu không hợp lệ

Mỗi field có hàm clean riêng phù hợp với đặc thù dữ liệu.
"""

import re
from typing import Callable, Dict, Optional

from .keyword_matcher import normalize


# ── Hàm clean cho từng loại field ───────────────────────────────

def clean_title(text: str) -> Optional[str]:
    """Tiêu đề biểu mẫu: strip + chuẩn hóa khoảng trắng."""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned if cleaned else None


def clean_cccd(text: str) -> Optional[str]:
    """CCCD/CMND: đúng 12 chữ số."""
    digits = re.sub(r'\D', '', text)
    return digits if len(digits) == 12 else None


def clean_phone(text: str) -> Optional[str]:
    """Số điện thoại: 10–11 chữ số, bắt đầu bằng 0."""
    digits = re.sub(r'\D', '', text)
    if len(digits) in (10, 11) and digits.startswith('0'):
        return digits
    return None


def clean_date(text: str) -> Optional[str]:
    """Ngày sinh: dd/mm/yyyy — chấp nhận nhiều ký tự phân cách."""
    m = re.search(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', text)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
    return None


def clean_gender(text: str) -> Optional[str]:
    """Giới tính: chuẩn hóa về 'Nam' hoặc 'Nữ'."""
    t = normalize(text)
    if any(w in t for w in ['nữ', 'nu', 'female']):
        return 'Nữ'
    if any(w in t for w in ['nam', 'male']):
        return 'Nam'
    return None


def clean_email(text: str) -> Optional[str]:
    """Email: trích xuất địa chỉ email hợp lệ, fallback giữ nguyên."""
    m = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    if m:
        return m.group(0)
    return text.strip() if text.strip() else None


def clean_noi_dung(text: str) -> Optional[str]:
    """
    Nội dung đề nghị: chuẩn hóa về pattern cố định của CT01.
    'Đăng ký tạm trú/thường trú X nhân khẩu tại [địa chỉ]'
    Fallback: giữ nguyên text nếu không khớp pattern.
    """
    if not text:
        return None
    m = re.search(
        r'[Đđ]ăng\s+ký\s+(tạm\s+trú|thường\s+trú|tam\s+tru|thuong\s+tru)'
        r'\s+(\d+)\s+nhân\s+khẩu\s+tại\s+(.+)',
        text, re.IGNORECASE
    )
    if m:
        loai     = m.group(1).strip()
        so_nguoi = m.group(2)
        dia_chi  = m.group(3).strip().rstrip('.')
        return f"Đăng ký {loai} {so_nguoi} nhân khẩu tại {dia_chi}"
    return text.strip() if text.strip() else None


def clean_text(text: str) -> Optional[str]:
    """Field text tự do (họ tên, địa chỉ...): chỉ strip, không validate."""
    cleaned = text.strip()
    return cleaned if cleaned else None


# ── Map field → hàm clean ────────────────────────────────────────
CLEAN_FUNCTIONS: Dict[str, Callable] = {
    "title":                          clean_title,
    "so_dinh_dan_ca_nhan":            clean_cccd,
    "so_dinh_dan_ca_nhan_cua_chu_ho": clean_cccd,
    "so_dien_thoai":                  clean_phone,
    "ngay_thang_nam_sinh":            clean_date,
    "gioi_tinh":                      clean_gender,
    "email":                          clean_email,
    "noi_dung_de_nghi":               clean_noi_dung,
}


def validate_and_clean(field: str, raw_value: str) -> Optional[str]:
    """
    Lớp 3 entry point: áp dụng hàm clean phù hợp cho field.
    Field không có rule đặc biệt → chỉ strip whitespace.
    """
    if not raw_value:
        return None
    clean_fn = CLEAN_FUNCTIONS.get(field, clean_text)
    return clean_fn(raw_value)
