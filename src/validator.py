"""
validator.py — Kiểm tra tính hợp lệ của dữ liệu trích xuất
Áp dụng các quy tắc nghiệp vụ (business logic)
"""

import re
from datetime import datetime, date
from typing import Dict, List, Tuple


# ================================================
#  CÁC HÀM VALIDATOR TỪNG TRƯỜNG
# ================================================

def validate_date(date_str: str) -> Tuple[bool, str]:
    """
    Kiểm tra ngày tháng hợp lệ
    - Đúng định dạng DD/MM/YYYY (hoặc dấu - hoặc .)
    - Ngày tháng thực tế tồn tại (VD: không có 30/02)
    - Nằm trong khoảng hợp lý (1900 → hôm nay)
    """
    if not date_str:
        return False, "Không có giá trị"

    patterns = [
        r'^(\d{1,2})/(\d{1,2})/(\d{4})$',
        r'^(\d{1,2})-(\d{1,2})-(\d{4})$',
        r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$',
    ]

    for pattern in patterns:
        match = re.match(pattern, date_str.strip())
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            try:
                dt = datetime(year, month, day)
                if year < 1900:
                    return False, f"Năm {year} quá cũ"
                if dt.date() > date.today():
                    return False, "Ngày trong tương lai"
                return True, f"Hợp lệ: {dt.strftime('%d/%m/%Y')}"
            except ValueError:
                return False, f"Ngày không tồn tại: {day}/{month}/{year}"

    return False, f"Sai định dạng: '{date_str}' (cần DD/MM/YYYY)"


def validate_id_number(id_str: str) -> Tuple[bool, str]:
    """
    Kiểm tra số CCCD/CMND
    - CCCD mới: 12 chữ số
    - CMND cũ: 9 chữ số
    """
    if not id_str:
        return False, "Không có giá trị"

    # Chỉ giữ lại chữ số
    clean = re.sub(r'\D', '', id_str)

    if len(clean) == 12:
        if clean == '0' * 12:
            return False, "Không hợp lệ (toàn số 0)"
        return True, f"CCCD hợp lệ: {clean}"
    elif len(clean) == 9:
        if clean == '0' * 9:
            return False, "Không hợp lệ (toàn số 0)"
        return True, f"CMND hợp lệ: {clean}"
    else:
        return False, f"Độ dài sai: {len(clean)} chữ số (cần 9 hoặc 12)"


def validate_phone(phone_str: str) -> Tuple[bool, str]:
    """
    Kiểm tra số điện thoại Việt Nam
    Các đầu số hợp lệ: 03x, 05x, 07x, 08x, 09x
    """
    if not phone_str:
        return False, "Không có giá trị"

    # Bỏ khoảng trắng và ký tự phân cách
    clean = re.sub(r'[\s\-\.\(\)]', '', phone_str)

    # Chuẩn hóa đầu +84 → 0
    if clean.startswith('+84'):
        clean = '0' + clean[3:]

    pattern = r'^(0)(3[2-9]|5[6-9]|7[06-9]|8[0-9]|9[0-9])\d{7}$'
    if re.match(pattern, clean):
        return True, f"Hợp lệ: {clean}"

    return False, f"Không hợp lệ: '{phone_str}'"


def validate_name(name_str: str) -> Tuple[bool, str]:
    """
    Kiểm tra họ tên tiếng Việt
    - Độ dài 2–100 ký tự
    - Không chứa số hoặc ký tự đặc biệt
    - Có ít nhất 2 từ (họ và tên)
    """
    if not name_str:
        return False, "Không có giá trị"

    stripped = name_str.strip()

    if len(stripped) < 2:
        return False, "Quá ngắn (tối thiểu 2 ký tự)"

    if len(stripped) > 100:
        return False, "Quá dài (tối đa 100 ký tự)"

    # Kiểm tra ký tự không hợp lệ
    if re.search(r'[\d@#$%^&*()+=\[\]{}<>\\|]', stripped):
        return False, "Chứa ký tự không hợp lệ (số hoặc ký tự đặc biệt)"

    words = stripped.split()
    if len(words) < 2:
        return False, "Cần ít nhất 2 từ (họ + tên)"

    return True, f"Hợp lệ: {stripped}"


def calculate_age(birth_date_str: str) -> Tuple[bool, int, str]:
    """
    Tính tuổi từ ngày sinh

    Returns:
        (success, age, message)
    """
    parts = re.split(r'[/\-\.]', birth_date_str.strip())
    if len(parts) != 3:
        return False, 0, "Không thể tính tuổi"

    try:
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        birth = date(year, month, day)
        today = date.today()
        age = (today - birth).days // 365
        return True, age, f"{age} tuổi"
    except Exception as e:
        return False, 0, f"Lỗi: {e}"


# ================================================
#  PIPELINE VALIDATION CHÍNH
# ================================================

def run_validation(extracted_data: Dict) -> Dict:
    """
    Chạy toàn bộ kiểm tra trên dữ liệu đã trích xuất

    Args:
        extracted_data: Kết quả từ kie.extract_information()

    Returns:
        {
            'is_valid': bool,     — Kết quả tổng thể
            'fields': {...},      — Kết quả kiểm tra từng trường
            'errors': [...],      — Danh sách lỗi
            'warnings': [...],    — Danh sách cảnh báo
        }
    """
    print("\n" + "="*50)
    print("✔️  BẮT ĐẦU KIỂM TRA NGHIỆP VỤ")
    print("="*50)

    result = {
        'is_valid': True,
        'fields': {},
        'errors': [],
        'warnings': [],
    }

    # Định nghĩa: (tên hiển thị, hàm validator, bắt buộc không?)
    field_validators = {
        'ho_ten':        ('Họ tên',        validate_name,      True),
        'ngay_sinh':     ('Ngày sinh',     validate_date,      True),
        'so_id':         ('Số CCCD/CMND',  validate_id_number, True),
        'so_dien_thoai': ('Số điện thoại', validate_phone,     False),
        'ngay_cap':      ('Ngày cấp',      validate_date,      False),
        'ngay_het_han':  ('Ngày hết hạn',  validate_date,      False),
    }

    for field_key, (display_name, validator_fn, is_required) in field_validators.items():
        value = extracted_data.get(field_key, '')

        if not value:
            if is_required:
                msg = f"Thiếu trường bắt buộc: {display_name}"
                result['errors'].append(msg)
                result['is_valid'] = False
                print(f"  ❌ {display_name:<20}: KHÔNG TÌM THẤY")
                result['fields'][field_key] = {'valid': False, 'message': 'Không tìm thấy', 'value': None}
            else:
                print(f"  ⚠️  {display_name:<20}: Không có (tùy chọn)")
                result['fields'][field_key] = {'valid': None, 'message': 'Không có dữ liệu', 'value': None}
            continue

        is_valid, message = validator_fn(str(value))
        icon = "✅" if is_valid else "❌"
        print(f"  {icon} {display_name:<20}: '{value}' → {message}")

        result['fields'][field_key] = {
            'valid': is_valid,
            'message': message,
            'value': value,
        }

        if not is_valid:
            result['errors'].append(f"{display_name}: {message}")
            result['is_valid'] = False

    # Kiểm tra bổ sung: tuổi
    birth = extracted_data.get('ngay_sinh', '')
    if birth:
        ok, age, age_msg = calculate_age(birth)
        if ok:
            icon = "✅" if 0 <= age <= 150 else "⚠️ "
            print(f"  {icon} Tuổi tính được          : {age_msg}")
            if age > 150:
                result['warnings'].append(f"Tuổi bất thường: {age_msg}")
        else:
            print(f"  ⚠️  Không tính được tuổi: {age_msg}")

    # ── Tổng kết ──────────────────────────────────
    print()
    if result['is_valid']:
        print("🎉 KẾT QUẢ TỔNG THỂ: ✅ HỢP LỆ")
    else:
        print(f"❗ KẾT QUẢ TỔNG THỂ: ❌ CÓ {len(result['errors'])} LỖI")
        for err in result['errors']:
            print(f"   → {err}")

    if result['warnings']:
        print(f"   ⚠️  {len(result['warnings'])} cảnh báo:")
        for w in result['warnings']:
            print(f"   → {w}")

    return result
