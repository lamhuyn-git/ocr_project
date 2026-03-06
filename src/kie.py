"""
kie.py — Key Information Extraction (Trích xuất thông tin quan trọng)
Dùng Rule-based (Regex + Keyword Matching) để tìm các trường thông tin
Ví dụ: Họ tên, Ngày sinh, Số CCCD, Địa chỉ, v.v.
"""

import re
from typing import List, Dict, Optional, Tuple


# ================================================
#  REGEX PATTERNS
# ================================================

# Ngày tháng năm: DD/MM/YYYY hoặc DD-MM-YYYY hoặc DD.MM.YYYY
DATE_PATTERN = re.compile(r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b')

# Số CCCD (12 chữ số) hoặc CMND (9 chữ số)
ID_NUMBER_PATTERN = re.compile(r'\b(\d{9}|\d{12})\b')

# Số điện thoại Việt Nam (các đầu số phổ biến)
PHONE_PATTERN = re.compile(
    r'\b(0|\+84)(3[2-9]|5[6-9]|7[06-9]|8[0-9]|9[0-9])\d{7}\b'
)

# Địa chỉ email
EMAIL_PATTERN = re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b')


# ================================================
#  TỪ ĐIỂN KEYWORD → TÊN TRƯỜNG
# ================================================

FIELD_KEYWORDS = {
    # Họ tên
    'họ và tên': 'ho_ten',
    'họ tên': 'ho_ten',
    'full name': 'ho_ten',
    'name': 'ho_ten',

    # Ngày sinh
    'ngày sinh': 'ngay_sinh',
    'sinh ngày': 'ngay_sinh',
    'date of birth': 'ngay_sinh',
    'dob': 'ngay_sinh',

    # Giới tính
    'giới tính': 'gioi_tinh',
    'gender': 'gioi_tinh',
    'sex': 'gioi_tinh',

    # Số CCCD/CMND
    'số cccd': 'so_id',
    'số cmnd': 'so_id',
    'số căn cước': 'so_id',
    'id number': 'so_id',
    'số:': 'so_id',

    # Quê quán
    'quê quán': 'que_quan',
    'place of origin': 'que_quan',

    # Quốc tịch
    'quốc tịch': 'quoc_tich',
    'nationality': 'quoc_tich',

    # Địa chỉ thường trú
    'nơi thường trú': 'dia_chi',
    'địa chỉ': 'dia_chi',
    'address': 'dia_chi',
    'place of residence': 'dia_chi',

    # Số điện thoại
    'điện thoại': 'so_dien_thoai',
    'phone': 'so_dien_thoai',
    'tel': 'so_dien_thoai',
    'mobile': 'so_dien_thoai',
    'sdt': 'so_dien_thoai',

    # Ngày cấp
    'ngày cấp': 'ngay_cap',
    'date of issue': 'ngay_cap',

    # Ngày hết hạn
    'có giá trị đến': 'ngay_het_han',
    'date of expiry': 'ngay_het_han',
    'hết hạn': 'ngay_het_han',

    # Nơi cấp
    'nơi cấp': 'noi_cap',
    'place of issue': 'noi_cap',

    # Dân tộc
    'dân tộc': 'dan_toc',
    'ethnicity': 'dan_toc',

    # Tôn giáo
    'tôn giáo': 'ton_giao',
    'religion': 'ton_giao',
}

# Các từ đồng nghĩa với giới tính
GENDER_MAP = {
    'nam': 'Nam', 'nữ': 'Nữ', 'nu': 'Nữ',
    'male': 'Nam', 'female': 'Nữ',
    'm': 'Nam', 'f': 'Nữ',
}


# ================================================
#  HÀM TIỆN ÍCH
# ================================================

def normalize(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ khoảng trắng thừa"""
    return ' '.join(text.lower().strip().split())


def extract_value_after_keyword(line: str, keyword: str) -> str:
    """
    Lấy phần text sau keyword trong cùng một dòng
    Ví dụ: "Họ và tên: Nguyễn Văn A" → "Nguyễn Văn A"
    """
    normalized_line = normalize(line)
    if keyword not in normalized_line:
        return ""

    # Vị trí kết thúc của keyword trong bản gốc (giữ nguyên case)
    idx = normalized_line.find(keyword) + len(keyword)
    value = line[idx:].strip()

    # Loại bỏ dấu phân cách ở đầu (: ; - —)
    value = re.sub(r'^[:;\-–—\s]+', '', value).strip()
    return value


def find_field_in_line(line: str, all_lines: List[str], idx: int) -> Optional[Tuple[str, str]]:
    """
    Tìm trường thông tin từ một dòng text

    Returns:
        (tên_trường, giá_trị) hoặc None nếu không tìm thấy
    """
    normalized = normalize(line)

    for keyword, field_name in FIELD_KEYWORDS.items():
        if keyword not in normalized:
            continue

        # Thử lấy giá trị từ cùng dòng
        value = extract_value_after_keyword(line, keyword)

        # Nếu không có, lấy dòng kế tiếp
        if not value and idx + 1 < len(all_lines):
            next_line = all_lines[idx + 1].strip()
            next_normalized = normalize(next_line)

            # Chỉ lấy nếu dòng kế không phải keyword khác
            is_another_field = any(kw in next_normalized for kw in FIELD_KEYWORDS)
            if not is_another_field and next_line:
                value = next_line

        if value:
            return (field_name, value)

    return None


# ================================================
#  PIPELINE CHÍNH
# ================================================

def extract_information(text_lines: List[str]) -> Dict:
    """
    Trích xuất tất cả thông tin từ danh sách dòng OCR

    Args:
        text_lines: Danh sách dòng text sau OCR

    Returns:
        Dict các trường thông tin đã tìm được
    """
    print("\n" + "="*50)
    print("🔎 BẮT ĐẦU TRÍCH XUẤT THÔNG TIN (KIE)")
    print("="*50)

    result = {}

    # ── Phương pháp 1: Keyword Matching ──────────────
    for i, line in enumerate(text_lines):
        field_info = find_field_in_line(line, text_lines, i)
        if field_info:
            field_name, value = field_info
            if field_name not in result:  # Chỉ lưu lần đầu tìm thấy
                result[field_name] = value
                print(f"  🏷️  {field_name:<20} = {value}")

    # ── Phương pháp 2: Regex Pattern Matching ────────
    full_text = ' '.join(text_lines)

    # Tìm số CCCD/CMND nếu chưa có
    if 'so_id' not in result:
        id_matches = ID_NUMBER_PATTERN.findall(full_text)
        if id_matches:
            # Ưu tiên 12 chữ số (CCCD) hơn 9 chữ số (CMND)
            twelve = [n for n in id_matches if len(n) == 12]
            nine = [n for n in id_matches if len(n) == 9]
            best = (twelve or nine)
            if best:
                result['so_id'] = best[0]
                print(f"  🔢 so_id (pattern)      = {best[0]}")

    # Tìm ngày sinh nếu chưa có
    if 'ngay_sinh' not in result:
        dates = DATE_PATTERN.findall(full_text)
        if dates:
            d, m, y = dates[0]
            result['ngay_sinh'] = f"{d}/{m}/{y}"
            print(f"  📅 ngay_sinh (pattern)  = {d}/{m}/{y}")

    # Tìm số điện thoại nếu chưa có
    if 'so_dien_thoai' not in result:
        phones = PHONE_PATTERN.findall(full_text)
        if phones:
            result['so_dien_thoai'] = '0' + phones[0][1] + full_text[
                full_text.find(phones[0][1]) + len(phones[0][1]):
                full_text.find(phones[0][1]) + len(phones[0][1]) + 7
            ]
            # Cách đơn giản hơn: tìm lại toàn bộ match
            phone_full = re.search(PHONE_PATTERN, full_text)
            if phone_full:
                result['so_dien_thoai'] = phone_full.group()
                print(f"  📞 so_dien_thoai        = {result['so_dien_thoai']}")

    # Tìm email nếu chưa có
    if 'email' not in result:
        email_match = EMAIL_PATTERN.search(full_text)
        if email_match:
            result['email'] = email_match.group()
            print(f"  📧 email (pattern)      = {result['email']}")

    # Tìm giới tính nếu chưa có
    if 'gioi_tinh' not in result:
        for line in text_lines:
            normalized = normalize(line)
            for gender_text, gender_value in GENDER_MAP.items():
                if re.search(r'\b' + re.escape(gender_text) + r'\b', normalized):
                    result['gioi_tinh'] = gender_value
                    print(f"  👤 gioi_tinh (pattern) = {gender_value}")
                    break
            if 'gioi_tinh' in result:
                break

    print(f"\n✅ Trích xuất được {len(result)} trường thông tin")
    return result
