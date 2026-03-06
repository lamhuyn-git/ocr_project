# Hướng Dẫn Build Dự Án OCR Hồ Sơ Tiếng Việt

---

## Tổng Quan Dự Án

Dự án này xây dựng một pipeline tự động đọc và trích xuất thông tin từ ảnh hồ sơ/giấy tờ đăng ký tạm trú tiếng Việt

```
Ảnh hồ sơ → Tiền xử lý → OCR (PaddleOCR) → KIE (Rule-based) → Validation → Kết quả
```

---

## Cấu Trúc Thư Mục Dự Án

```
ocr_project/
│
├── images/                  ← Thư mục chứa ảnh đầu vào
│   └── sample.jpg
│
├── outputs/                 ← Kết quả xuất ra
│   └── result.json
│
├── src/                     ← Code nguồn
│   ├── preprocess.py        ← Bước 1: Tiền xử lý ảnh
│   ├── ocr_engine.py        ← Bước 2: OCR với PaddleOCR
│   ├── kie.py               ← Bước 3: Trích xuất trường thông tin
│   └── validator.py         ← Bước 4: Kiểm tra nghiệp vụ
│
├── main.py                  ← File chạy chính
├── requirements.txt         ← Danh sách thư viện cần cài
└── README.md
```

---

## BƯỚC 1 — Cài Đặt Môi Trường

### 1.1 Cài Python (nếu chưa có)

1. Vào https://www.python.org/downloads/
2. Tải Python **3.9 hoặc 3.10** (khuyến nghị — PaddleOCR tương thích tốt nhất)
3. Khi cài, **tích vào ô "Add Python to PATH"** rồi nhấn Install
4. Mở Terminal, kiểm tra: `python --version` → phải hiện `Python 3.9.x`

### 1.2 Cài VSCode

1. Vào https://code.visualstudio.com/ → tải và cài
2. Mở VSCode → Vào **Extensions** (Ctrl+Shift+X)
3. Tìm và cài:
   - `Python` (của Microsoft)
   - `Pylance`

### 1.3 Tạo Thư Mục Dự Án

Mở Terminal trong VSCode (`Ctrl + `` ` ``):

```bash
# Tạo thư mục dự án
mkdir ocr_project
cd ocr_project

# Tạo các thư mục con
mkdir images outputs src
```

### 1.4 Tạo Virtual Environment (Môi Trường Ảo)

> **Tại sao cần venv?** Để tránh xung đột thư viện giữa các dự án khác nhau.

```bash
# Tạo môi trường ảo tên là "venv"
python -m venv venv

# Kích hoạt môi trường ảo:
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

Sau khi kích hoạt, terminal sẽ hiện `(venv)` ở đầu dòng — nghĩa là thành công

---

## BƯỚC 2 — Cài Thư Viện

Tạo file `requirements.txt` với nội dung:

```
paddlepaddle==2.6.1
paddleocr==2.7.3
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
numpy==1.24.4
Pillow==10.3.0
```

Chạy lệnh cài đặt:

```bash
pip install -r requirements.txt
```

> ⏳ Lần đầu cài sẽ mất **5–15 phút** do PaddleOCR cần tải model về.

> ⚠️ **Lưu ý Windows:** Nếu báo lỗi về `paddlepaddle`, thử:
>
> ```bash
> pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
> ```

---

## 💻 BƯỚC 3 — Viết Code

### 3.1 File `src/preprocess.py` — Tiền Xử Lý Ảnh

```python
"""
preprocess.py — Tiền xử lý ảnh trước khi đưa vào OCR
Mục tiêu: làm ảnh rõ hơn, thẳng hơn, ít nhiễu hơn
"""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Đọc ảnh từ đường dẫn file"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    print(f"✅ Đọc ảnh thành công: {image_path} | Kích thước: {img.shape}")
    return img


def resize_image(img: np.ndarray, max_width: int = 1500) -> np.ndarray:
    """
    Resize ảnh nếu quá lớn — giúp OCR xử lý nhanh hơn
    Giữ nguyên tỉ lệ khung hình
    """
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
        print(f"🔄 Resize: {w}x{h} → {max_width}x{new_h}")
    return img


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Chỉnh thẳng ảnh bị nghiêng (deskew)
    Dùng Hough Line Transform để phát hiện góc nghiêng
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Tìm các đường thẳng
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        print("⚠️ Không phát hiện được đường thẳng để deskew")
        return img

    # Tính góc nghiêng trung bình
    angles = []
    for line in lines[:20]:  # Chỉ dùng 20 đường đầu
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:  # Bỏ qua góc quá lớn
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)

    if abs(median_angle) < 0.5:  # Góc nhỏ hơn 0.5 độ — không cần xoay
        return img

    print(f"📐 Phát hiện góc nghiêng: {median_angle:.2f}°, đang chỉnh...")

    # Xoay ảnh
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def denoise(img: np.ndarray) -> np.ndarray:
    """Giảm nhiễu ảnh bằng Non-local Means Denoising"""
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    print("🔇 Đã giảm nhiễu ảnh")
    return denoised


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Tăng độ tương phản bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Giúp chữ hiện rõ hơn trên nền không đều
    """
    # Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Áp dụng CLAHE cho kênh L (độ sáng)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Ghép lại và chuyển về BGR
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    print("✨ Đã tăng độ tương phản (CLAHE)")
    return result


def preprocess_pipeline(image_path: str, save_debug: bool = False) -> np.ndarray:
    """
    Pipeline tiền xử lý đầy đủ

    Args:
        image_path: Đường dẫn tới ảnh gốc
        save_debug: Nếu True, lưu ảnh đã xử lý ra file để kiểm tra

    Returns:
        Ảnh đã được xử lý (numpy array)
    """
    print("\n" + "="*50)
    print("🖼️  BẮT ĐẦU TIỀN XỬ LÝ ẢNH")
    print("="*50)

    # Bước 1: Đọc ảnh
    img = load_image(image_path)

    # Bước 2: Resize nếu cần
    img = resize_image(img)

    # Bước 3: Chỉnh thẳng
    img = deskew(img)

    # Bước 4: Giảm nhiễu
    img = denoise(img)

    # Bước 5: Tăng độ tương phản
    img = apply_clahe(img)

    # Lưu ảnh debug nếu cần
    if save_debug:
        debug_path = image_path.replace("images/", "outputs/debug_")
        cv2.imwrite(debug_path, img)
        print(f"💾 Lưu ảnh debug: {debug_path}")

    print("✅ Tiền xử lý hoàn tất!\n")
    return img
```

---

### 3.2 File `src/ocr_engine.py` — OCR với PaddleOCR

```python
"""
ocr_engine.py — Nhận diện chữ trong ảnh dùng PaddleOCR
PaddleOCR hỗ trợ tiếng Việt với độ chính xác cao
"""

import numpy as np
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict


# Khởi tạo PaddleOCR một lần duy nhất (tránh tải model nhiều lần)
# lang='vi' — tiếng Việt
# use_angle_cls=True — phát hiện chữ bị xoay
# use_gpu=False — dùng CPU (nếu có GPU, đặt True để nhanh hơn)
_ocr_instance = None


def get_ocr_instance() -> PaddleOCR:
    """Lấy instance OCR (Singleton pattern — chỉ tạo 1 lần)"""
    global _ocr_instance
    if _ocr_instance is None:
        print("🔄 Đang khởi tạo PaddleOCR (lần đầu sẽ tải model ~100MB)...")
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang='vi',              # Tiếng Việt
            use_gpu=False,          # Đổi thành True nếu có GPU
            show_log=False,         # Tắt log thừa
            rec_model_dir=None,     # Dùng model mặc định PP-OCRv4
            det_model_dir=None,
        )
        print("✅ PaddleOCR sẵn sàng!")
    return _ocr_instance


def run_ocr(img: np.ndarray) -> List[Dict]:
    """
    Chạy OCR trên ảnh đã tiền xử lý

    Args:
        img: Ảnh (numpy array từ OpenCV)

    Returns:
        Danh sách các dòng text với vị trí và độ tin cậy
        Mỗi phần tử: {
            'text': str,        — nội dung chữ
            'confidence': float, — độ tin cậy (0.0 → 1.0)
            'bbox': list        — tọa độ hộp bao [top-left, top-right, bottom-right, bottom-left]
            'center_y': float   — tọa độ y trung tâm (dùng để sắp xếp theo dòng)
        }
    """
    print("\n" + "="*50)
    print("🔍 BẮT ĐẦU OCR")
    print("="*50)

    ocr = get_ocr_instance()

    # Chạy OCR
    raw_results = ocr.ocr(img, cls=True)

    if not raw_results or raw_results[0] is None:
        print("⚠️ Không tìm thấy chữ trong ảnh!")
        return []

    # Xử lý kết quả thô từ PaddleOCR
    parsed = []
    for line in raw_results[0]:
        bbox = line[0]          # 4 điểm góc: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]       # Nội dung chữ
        confidence = line[1][1] # Độ tin cậy

        # Tính tọa độ y trung tâm (để sắp xếp theo thứ tự dòng)
        center_y = sum(pt[1] for pt in bbox) / 4

        parsed.append({
            'text': text.strip(),
            'confidence': round(confidence, 4),
            'bbox': bbox,
            'center_y': center_y,
            'x_left': min(pt[0] for pt in bbox),  # Cạnh trái nhất
        })

    # Sắp xếp theo thứ tự từ trên xuống dưới, trái sang phải
    parsed.sort(key=lambda x: (round(x['center_y'] / 15) * 15, x['x_left']))

    print(f"📝 Tìm thấy {len(parsed)} dòng text")

    # In preview kết quả
    for i, item in enumerate(parsed):
        conf_emoji = "✅" if item['confidence'] > 0.85 else "⚠️"
        print(f"  {conf_emoji} [{i+1:02d}] ({item['confidence']:.2%}) {item['text']}")

    print()
    return parsed


def filter_by_confidence(ocr_results: List[Dict], min_confidence: float = 0.70) -> List[Dict]:
    """Lọc bỏ các dòng có độ tin cậy thấp"""
    filtered = [r for r in ocr_results if r['confidence'] >= min_confidence]
    removed = len(ocr_results) - len(filtered)
    if removed > 0:
        print(f"🔽 Lọc bỏ {removed} dòng có confidence < {min_confidence:.0%}")
    return filtered


def get_text_lines(ocr_results: List[Dict]) -> List[str]:
    """Trích xuất danh sách dòng text thuần (không kèm metadata)"""
    return [r['text'] for r in ocr_results]
```

---

### 3.3 File `src/kie.py` — Trích Xuất Thông Tin (KIE)

```python
"""
kie.py — Key Information Extraction (Trích xuất thông tin quan trọng)
Dùng Rule-based (Regex + Logic) để xác định các trường thông tin
Ví dụ: Họ tên, Ngày sinh, Số CCCD, Địa chỉ, v.v.
"""

import re
from typing import List, Dict, Optional


# =============================================
# ĐỊNH NGHĨA CÁC PATTERN REGEX TIẾNG VIỆT
# =============================================

# Pattern ngày tháng năm: DD/MM/YYYY hoặc DD-MM-YYYY
DATE_PATTERN = re.compile(
    r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b'
)

# Pattern số CCCD/CMND: 9 hoặc 12 chữ số
ID_NUMBER_PATTERN = re.compile(
    r'\b(\d{9}|\d{12})\b'
)

# Pattern số điện thoại Việt Nam
PHONE_PATTERN = re.compile(
    r'\b(0|\+84)(3[2-9]|5[6-9]|7[06-9]|8[0-9]|9[0-9])\d{7}\b'
)

# Pattern email
EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'
)

# Các từ khóa nhận diện trường thông tin (keyword → tên trường)
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
    'no.': 'so_id',

    # Quê quán / Quốc tịch
    'quê quán': 'que_quan',
    'place of origin': 'que_quan',
    'quốc tịch': 'quoc_tich',
    'nationality': 'quoc_tich',

    # Địa chỉ
    'nơi thường trú': 'dia_chi',
    'địa chỉ': 'dia_chi',
    'address': 'dia_chi',
    'place of residence': 'dia_chi',

    # Số điện thoại
    'điện thoại': 'so_dien_thoai',
    'phone': 'so_dien_thoai',
    'tel': 'so_dien_thoai',
    'mobile': 'so_dien_thoai',

    # Ngày cấp / Có giá trị đến
    'ngày cấp': 'ngay_cap',
    'date of issue': 'ngay_cap',
    'có giá trị đến': 'ngay_het_han',
    'date of expiry': 'ngay_het_han',

    # Nơi cấp
    'nơi cấp': 'noi_cap',
    'place of issue': 'noi_cap',
}

# Các giá trị giới tính có thể có
GENDER_VALUES = {
    'nam': 'Nam',
    'nữ': 'Nữ',
    'nu': 'Nữ',
    'male': 'Nam',
    'female': 'Nữ',
    'm': 'Nam',
    'f': 'Nữ',
}


def normalize_text(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ khoảng trắng thừa"""
    return ' '.join(text.lower().strip().split())


def find_field_from_keyword(line: str, all_lines: List[str], line_idx: int) -> Optional[tuple]:
    """
    Tìm tên trường từ từ khóa trong dòng text
    Trả về (tên_trường, giá_trị) hoặc None
    """
    normalized = normalize_text(line)

    for keyword, field_name in FIELD_KEYWORDS.items():
        if keyword in normalized:
            # Thử lấy giá trị từ cùng dòng (sau dấu : hoặc sau keyword)
            value = extract_value_from_line(line, keyword)

            # Nếu không có giá trị ở cùng dòng, lấy dòng tiếp theo
            if not value and line_idx + 1 < len(all_lines):
                next_line = all_lines[line_idx + 1]
                # Kiểm tra dòng tiếp theo không phải keyword khác
                next_normalized = normalize_text(next_line)
                is_another_keyword = any(kw in next_normalized for kw in FIELD_KEYWORDS)
                if not is_another_keyword:
                    value = next_line.strip()

            if value:
                return (field_name, value)

    return None


def extract_value_from_line(line: str, keyword: str) -> str:
    """
    Lấy phần giá trị từ dòng text sau keyword
    Ví dụ: "Họ và tên: Nguyễn Văn A" → "Nguyễn Văn A"
    """
    normalized = normalize_text(line)

    if keyword not in normalized:
        return ""

    # Tìm vị trí keyword trong text gốc (case-insensitive)
    idx = normalized.find(keyword) + len(keyword)
    value_part = line[idx:].strip()

    # Loại bỏ dấu câu ở đầu
    value_part = re.sub(r'^[:|\-–\s]+', '', value_part).strip()

    return value_part


def extract_with_patterns(text_lines: List[str]) -> Dict:
    """
    Trích xuất thông tin dùng Regex pattern
    (Bổ sung cho phương pháp keyword)
    """
    full_text = ' '.join(text_lines)

    extracted = {}

    # Tìm ngày tháng
    dates = DATE_PATTERN.findall(full_text)
    if dates:
        extracted['dates_found'] = [f"{d}/{m}/{y}" for d, m, y in dates]

    # Tìm số CCCD/CMND
    id_numbers = ID_NUMBER_PATTERN.findall(full_text)
    if id_numbers:
        # Ưu tiên số 12 chữ số (CCCD mới) trước số 9 chữ số (CMND cũ)
        twelve_digit = [n for n in id_numbers if len(n) == 12]
        nine_digit = [n for n in id_numbers if len(n) == 9]
        extracted['so_id_candidates'] = twelve_digit or nine_digit

    # Tìm số điện thoại
    phones = PHONE_PATTERN.findall(full_text)
    if phones:
        extracted['so_dien_thoai_candidates'] = list(set(phones))

    # Tìm email
    emails = EMAIL_PATTERN.findall(full_text)
    if emails:
        extracted['email'] = emails[0]

    # Tìm giới tính
    for line in text_lines:
        normalized = normalize_text(line)
        for gender_text, gender_value in GENDER_VALUES.items():
            if re.search(r'\b' + re.escape(gender_text) + r'\b', normalized):
                extracted['gioi_tinh'] = gender_value
                break

    return extracted


def extract_information(text_lines: List[str]) -> Dict:
    """
    Pipeline KIE chính — trích xuất tất cả thông tin

    Args:
        text_lines: Danh sách dòng text từ OCR

    Returns:
        Dict chứa các trường thông tin đã trích xuất
    """
    print("\n" + "="*50)
    print("🔎 BẮT ĐẦU TRÍCH XUẤT THÔNG TIN (KIE)")
    print("="*50)

    result = {}

    # Phương pháp 1: Keyword matching
    for i, line in enumerate(text_lines):
        field_info = find_field_from_keyword(line, text_lines, i)
        if field_info:
            field_name, value = field_info
            if field_name not in result:  # Chỉ lưu lần đầu tìm thấy
                result[field_name] = value
                print(f"  🏷️  {field_name}: {value}")

    # Phương pháp 2: Regex pattern matching (bổ sung)
    pattern_results = extract_with_patterns(text_lines)

    # Merge kết quả — ưu tiên keyword matching
    for key, value in pattern_results.items():
        if key.endswith('_candidates'):
            base_key = key.replace('_candidates', '')
            if base_key not in result and value:
                result[base_key] = value[0]
                print(f"  🔢 {base_key} (pattern): {value[0]}")
        elif key not in result:
            result[key] = value
            print(f"  📅 {key} (pattern): {value}")

    # Xử lý đặc biệt: ngày tháng
    if 'dates_found' in result and 'ngay_sinh' not in result:
        # Ngày đầu tiên tìm thấy thường là ngày sinh
        if result['dates_found']:
            result['ngay_sinh'] = result.pop('dates_found')[0]
    elif 'dates_found' in result:
        del result['dates_found']  # Xóa nếu đã có ngày sinh

    print(f"\n✅ Trích xuất được {len(result)} trường thông tin")
    return result
```

---

### 3.4 File `src/validator.py` — Kiểm Tra Nghiệp Vụ

```python
"""
validator.py — Kiểm tra tính hợp lệ của thông tin trích xuất
Áp dụng các quy tắc nghiệp vụ (business logic)
"""

import re
from datetime import datetime, date
from typing import Dict, List, Tuple


def validate_date(date_str: str) -> Tuple[bool, str]:
    """
    Kiểm tra ngày tháng hợp lệ
    - Đúng định dạng DD/MM/YYYY
    - Ngày tháng thực tế tồn tại
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
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            try:
                dt = datetime(year, month, day)

                # Kiểm tra năm hợp lý (1900 → ngày hiện tại)
                if year < 1900 or dt.date() > date.today():
                    return False, f"Năm không hợp lệ: {year}"

                return True, f"Hợp lệ: {dt.strftime('%d/%m/%Y')}"
            except ValueError:
                return False, f"Ngày không tồn tại: {day}/{month}/{year}"

    return False, f"Sai định dạng: '{date_str}' (cần DD/MM/YYYY)"


def validate_id_number(id_str: str) -> Tuple[bool, str]:
    """
    Kiểm tra số CCCD/CMND
    - CCCD mới: 12 chữ số
    - CMND cũ: 9 chữ số
    - Không được toàn số 0
    """
    if not id_str:
        return False, "Không có giá trị"

    clean = re.sub(r'\D', '', id_str)  # Chỉ giữ lại chữ số

    if len(clean) == 12:
        if clean == '0' * 12:
            return False, "Số không hợp lệ (toàn số 0)"
        return True, f"CCCD hợp lệ: {clean}"
    elif len(clean) == 9:
        if clean == '0' * 9:
            return False, "Số không hợp lệ (toàn số 0)"
        return True, f"CMND hợp lệ: {clean}"
    else:
        return False, f"Độ dài không hợp lệ: {len(clean)} chữ số (cần 9 hoặc 12)"


def validate_phone(phone_str: str) -> Tuple[bool, str]:
    """Kiểm tra số điện thoại Việt Nam"""
    if not phone_str:
        return False, "Không có giá trị"

    clean = re.sub(r'[\s\-\.\(\)]', '', phone_str)

    # Chuẩn hóa đầu số
    if clean.startswith('+84'):
        clean = '0' + clean[3:]

    pattern = r'^(0)(3[2-9]|5[6-9]|7[06-9]|8[0-9]|9[0-9])\d{7}$'
    if re.match(pattern, clean):
        return True, f"Hợp lệ: {clean}"

    return False, f"Số điện thoại không hợp lệ: '{phone_str}'"


def validate_name(name_str: str) -> Tuple[bool, str]:
    """Kiểm tra họ tên (tiếng Việt)"""
    if not name_str:
        return False, "Không có giá trị"

    # Độ dài tối thiểu 2, tối đa 100 ký tự
    if len(name_str) < 2:
        return False, "Họ tên quá ngắn"
    if len(name_str) > 100:
        return False, "Họ tên quá dài"

    # Chỉ chứa chữ cái (kể cả tiếng Việt có dấu) và khoảng trắng
    # Loại bỏ số và ký tự đặc biệt
    if re.search(r'[\d@#$%^&*()+=\[\]{}<>]', name_str):
        return False, "Họ tên chứa ký tự không hợp lệ"

    # Phải có ít nhất 2 từ (họ và tên)
    words = name_str.split()
    if len(words) < 2:
        return False, "Cần ít nhất họ và tên (2 từ)"

    return True, f"Hợp lệ: {name_str}"


def validate_age(birth_date_str: str, min_age: int = 0, max_age: int = 150) -> Tuple[bool, str]:
    """Tính và kiểm tra tuổi hợp lệ"""
    is_valid, msg = validate_date(birth_date_str)
    if not is_valid:
        return False, f"Ngày sinh không hợp lệ: {msg}"

    try:
        parts = re.split(r'[/\-\.]', birth_date_str)
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        birth = date(year, month, day)
        today = date.today()
        age = (today - birth).days // 365

        if age < min_age:
            return False, f"Tuổi quá nhỏ: {age} tuổi (tối thiểu {min_age})"
        if age > max_age:
            return False, f"Tuổi không hợp lệ: {age} tuổi"

        return True, f"Hợp lệ: {age} tuổi"
    except Exception as e:
        return False, f"Lỗi tính tuổi: {e}"


def run_validation(extracted_data: Dict) -> Dict:
    """
    Chạy tất cả validation trên dữ liệu đã trích xuất

    Returns:
        Dict với cấu trúc:
        {
            'is_valid': bool,             — Kết quả tổng thể
            'fields': {                   — Kết quả từng trường
                'ho_ten': {'valid': bool, 'message': str, 'value': str},
                ...
            },
            'errors': [str],             — Danh sách lỗi
            'warnings': [str],           — Danh sách cảnh báo
        }
    """
    print("\n" + "="*50)
    print("✔️  BẮT ĐẦU KIỂM TRA NGHIỆP VỤ")
    print("="*50)

    validation_result = {
        'is_valid': True,
        'fields': {},
        'errors': [],
        'warnings': [],
    }

    # Danh sách các validator cần chạy
    validators = {
        'ho_ten': ('Họ tên', validate_name, True),           # (tên hiển thị, hàm, bắt buộc)
        'ngay_sinh': ('Ngày sinh', validate_date, True),
        'so_id': ('Số CCCD/CMND', validate_id_number, True),
        'so_dien_thoai': ('Số điện thoại', validate_phone, False),
        'ngay_cap': ('Ngày cấp', validate_date, False),
        'ngay_het_han': ('Ngày hết hạn', validate_date, False),
    }

    for field_key, (display_name, validator_func, is_required) in validators.items():
        value = extracted_data.get(field_key, '')

        if not value:
            if is_required:
                msg = f"❌ Thiếu trường bắt buộc: {display_name}"
                validation_result['errors'].append(msg)
                validation_result['is_valid'] = False
                print(f"  {msg}")
                validation_result['fields'][field_key] = {
                    'valid': False, 'message': 'Không tìm thấy', 'value': None
                }
            else:
                print(f"  ⚠️  {display_name}: Không có (không bắt buộc)")
                validation_result['fields'][field_key] = {
                    'valid': None, 'message': 'Không có dữ liệu', 'value': None
                }
            continue

        is_valid, message = validator_func(str(value))

        status = "✅" if is_valid else "❌"
        print(f"  {status} {display_name}: '{value}' — {message}")

        validation_result['fields'][field_key] = {
            'valid': is_valid,
            'message': message,
            'value': value,
        }

        if not is_valid:
            validation_result['errors'].append(f"{display_name}: {message}")
            validation_result['is_valid'] = False

    # Kiểm tra thêm: tuổi nếu có ngày sinh
    if 'ngay_sinh' in extracted_data and extracted_data['ngay_sinh']:
        is_valid_age, age_msg = validate_age(extracted_data['ngay_sinh'])
        print(f"  {'✅' if is_valid_age else '⚠️ '} Tuổi: {age_msg}")
        if not is_valid_age:
            validation_result['warnings'].append(f"Tuổi: {age_msg}")

    # Tổng kết
    print()
    if validation_result['is_valid']:
        print("🎉 KẾT QUẢ: Hợp lệ!")
    else:
        print(f"❗ KẾT QUẢ: Có {len(validation_result['errors'])} lỗi")
        for err in validation_result['errors']:
            print(f"   → {err}")

    return validation_result
```

---

### 3.5 File `main.py` — Chạy Pipeline Chính

```python
"""
main.py — Pipeline OCR chính
Chạy file này để xử lý ảnh hồ sơ
"""

import json
import os
import sys
from datetime import datetime

# Thêm thư mục src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_pipeline
from ocr_engine import run_ocr, filter_by_confidence, get_text_lines
from kie import extract_information
from validator import run_validation


def process_document(image_path: str, save_result: bool = True) -> dict:
    """
    Xử lý một ảnh hồ sơ qua toàn bộ pipeline

    Args:
        image_path: Đường dẫn tới ảnh đầu vào
        save_result: Lưu kết quả ra file JSON

    Returns:
        Dict kết quả đầy đủ
    """
    start_time = datetime.now()

    print("\n" + "🚀 " + "="*48)
    print(f"  OCR PIPELINE — Bắt đầu xử lý")
    print(f"  File: {image_path}")
    print("="*50 + "\n")

    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file '{image_path}'")
        return {}

    # ── BƯỚC 1: Tiền xử lý ảnh ──────────────────
    preprocessed_img = preprocess_pipeline(image_path, save_debug=True)

    # ── BƯỚC 2: OCR ─────────────────────────────
    ocr_results = run_ocr(preprocessed_img)

    # Lọc kết quả OCR chất lượng thấp
    filtered_results = filter_by_confidence(ocr_results, min_confidence=0.70)
    text_lines = get_text_lines(filtered_results)

    # ── BƯỚC 3: Trích xuất thông tin (KIE) ──────
    extracted_data = extract_information(text_lines)

    # ── BƯỚC 4: Kiểm tra nghiệp vụ ──────────────
    validation_result = run_validation(extracted_data)

    # ── TỔNG HỢP KẾT QUẢ ────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()

    final_result = {
        'metadata': {
            'file': image_path,
            'processed_at': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 2),
            'total_lines_detected': len(ocr_results),
            'lines_after_filter': len(filtered_results),
        },
        'raw_text': text_lines,
        'extracted_data': extracted_data,
        'validation': validation_result,
        'status': 'SUCCESS' if validation_result['is_valid'] else 'FAILED',
    }

    # In tổng kết
    print("\n" + "="*50)
    print("📊 TỔNG KẾT")
    print("="*50)
    print(f"  ⏱️  Thời gian xử lý : {elapsed:.2f} giây")
    print(f"  📝 Số dòng OCR      : {len(ocr_results)}")
    print(f"  🏷️  Trường trích xuất: {len(extracted_data)}")
    print(f"  📋 Trạng thái       : {final_result['status']}")

    # In dữ liệu đã trích xuất
    print("\n📋 DỮ LIỆU TRÍCH XUẤT:")
    for key, value in extracted_data.items():
        print(f"   {key:<20} : {value}")

    # Lưu kết quả ra JSON
    if save_result:
        os.makedirs('outputs', exist_ok=True)
        output_filename = f"outputs/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã lưu kết quả: {output_filename}")

    print("\n" + "✅ " + "="*48 + "\n")
    return final_result


# ── CHẠY CHƯƠNG TRÌNH ────────────────────────────
if __name__ == '__main__':
    # Đặt đường dẫn tới ảnh của bạn ở đây
    # Thay 'images/sample.jpg' bằng ảnh thực của bạn

    IMAGE_PATH = 'images/sample.jpg'

    # Nếu truyền đường dẫn qua dòng lệnh: python main.py images/my_doc.jpg
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    result = process_document(IMAGE_PATH)
```

---

## 🚀 BƯỚC 4 — Chạy Dự Án

### 4.1 Chuẩn bị ảnh mẫu

Bỏ ảnh hồ sơ (CCCD, chứng minh nhân dân, v.v.) vào thư mục `images/`.

Đổi tên thành `sample.jpg` hoặc sửa đường dẫn trong `main.py`.

### 4.2 Chạy chương trình

```bash
# Đảm bảo virtual environment đã được kích hoạt (thấy (venv) ở đầu)
# Windows:
venv\Scripts\activate

# Chạy chương trình
python main.py

# Hoặc chỉ định file ảnh cụ thể:
python main.py images/cccd_front.jpg
```

### 4.3 Kết quả mong đợi

```
🚀 ==================================================
  OCR PIPELINE — Bắt đầu xử lý
  File: images/sample.jpg
==================================================

==================================================
🖼️  BẮT ĐẦU TIỀN XỬ LÝ ẢNH
==================================================
✅ Đọc ảnh thành công | Kích thước: (1080, 1920, 3)
📐 Phát hiện góc nghiêng: 1.23°, đang chỉnh...
🔇 Đã giảm nhiễu ảnh
✨ Đã tăng độ tương phản (CLAHE)

==================================================
🔍 BẮT ĐẦU OCR
==================================================
📝 Tìm thấy 24 dòng text
  ✅ [01] (98.21%) CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
  ✅ [02] (95.43%) Họ và tên: NGUYỄN VĂN AN
  ...

==================================================
🔎 BẮT ĐẦU TRÍCH XUẤT THÔNG TIN (KIE)
==================================================
  🏷️  ho_ten: NGUYỄN VĂN AN
  🏷️  ngay_sinh: 15/08/1990
  ...

==================================================
✔️  BẮT ĐẦU KIỂM TRA NGHIỆP VỤ
==================================================
  ✅ Họ tên: 'NGUYỄN VĂN AN' — Hợp lệ
  ✅ Ngày sinh: '15/08/1990' — Hợp lệ: 15/08/1990
  ...
🎉 KẾT QUẢ: Hợp lệ!

💾 Đã lưu kết quả: outputs/result_20260303_142530.json
```

---

## 🔧 BƯỚC 5 — Xử Lý Lỗi Thường Gặp

| Lỗi                              | Nguyên nhân       | Cách sửa                                                                         |
| -------------------------------- | ----------------- | -------------------------------------------------------------------------------- |
| `ModuleNotFoundError: paddleocr` | Chưa cài thư viện | Chạy lại `pip install -r requirements.txt`                                       |
| `FileNotFoundError`              | Sai đường dẫn ảnh | Kiểm tra lại tên file trong thư mục `images/`                                    |
| `OSError: [WinError 126]`        | Thiếu Visual C++  | Cài [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| OCR ra kết quả sai               | Ảnh mờ/tối        | Thêm `save_debug=True` để xem ảnh sau tiền xử lý                                 |
| Chạy rất chậm                    | Dùng CPU          | Đổi `use_gpu=True` nếu có GPU NVIDIA                                             |

---

## 📈 BƯỚC 6 — Nâng Cấp Tiếp Theo

Sau khi chạy thành công, bạn có thể:

1. **Xử lý nhiều ảnh cùng lúc:** Dùng vòng lặp qua thư mục `images/`
2. **Thêm loại giấy tờ mới:** Mở rộng từ điển `FIELD_KEYWORDS` trong `kie.py`
3. **Xuất ra Excel:** Dùng `openpyxl` để tạo bảng tổng hợp
4. **Tạo giao diện web:** Dùng `Gradio` (`pip install gradio`) để kéo thả ảnh

---

## 📁 Tóm Tắt Lệnh

```bash
# 1. Vào thư mục dự án
cd ocr_project

# 2. Kích hoạt môi trường ảo
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# 4. Chạy chương trình
python main.py

# 5. Xem kết quả
# Mở file trong thư mục outputs/
```
