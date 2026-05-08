"""
cell-row-extractor.py — Trích xuất dãy ô bảng bằng OpenCV, bypass text detector.

Dùng cho các field dạng bảng ô (CCCD, số định danh) mà text detector bỏ sót.
Algorithm:
  1. Crop vùng hàng bên phải label block
  2. Vertical projection → tìm đường kẻ dọc (cell borders)
  3. Crop từng ô giữa 2 đường kẻ
  4. ocr.ocr(cell, det=False) → recognition-only trên từng ô
  5. Nối kết quả thành chuỗi
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from paddleocr import PaddleOCR


# Ô hợp lệ phải rộng ít nhất ngưỡng này (px)
_MIN_CELL_WIDTH_PX = 15
# Cột được coi là cell border khi >= tỉ lệ này chiều cao chứa pixel đen
_BORDER_RATIO = 0.30
# Upscale ô có chiều cao nhỏ hơn ngưỡng này để recognition ổn định hơn
_MIN_CELL_HEIGHT_PX = 48
# Confidence tối thiểu khi recognition từng ô
_MIN_CELL_CONFIDENCE = 0.10


def _crop_row_region(img: np.ndarray, label_block: Dict) -> tuple:
    """
    Crop vùng hàng bên phải label_block từ ảnh gốc.
    Trả về (crop_img, x_offset, y_offset).
    """
    ys = [pt[1] for pt in label_block['bbox']]
    xs = [pt[0] for pt in label_block['bbox']]
    row_h = max(ys) - min(ys)
    pad   = int(row_h * 0.5)

    y1 = max(0,              int(min(ys)) - pad)
    y2 = min(img.shape[0],  int(max(ys)) + pad)
    # Bắt đầu ngay sau label, lùi 10px để bắt đường kẻ đầu tiên
    x1 = max(0,              int(max(xs)) - 10)
    x2 = img.shape[1]

    return img[y1:y2, x1:x2], x1, y1


def _find_cell_bounds(row_crop: np.ndarray) -> List[tuple]:
    """
    Dùng vertical projection để tìm ranh giới ô.

    Tính tổng pixel đen theo từng cột dọc:
      - Cột nhiều pixel đen = đường kẻ bảng (border)
      - Khoảng giữa 2 border = 1 ô

    Trả về list (x_start, x_end) tính theo tọa độ row_crop.
    """
    gray = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tổng pixel đen theo từng cột (axis=0)
    v_proj = np.sum(binary, axis=0).astype(np.float32)
    border_threshold = binary.shape[0] * _BORDER_RATIO * 255

    is_border = v_proj > border_threshold

    cells = []
    in_cell   = False
    cell_start = 0

    for x, border in enumerate(is_border):
        if not border and not in_cell:
            in_cell    = True
            cell_start = x
        elif border and in_cell:
            in_cell = False
            if x - cell_start >= _MIN_CELL_WIDTH_PX:
                cells.append((cell_start, x))

    # Xử lý ô cuối cùng (không có border đóng)
    if in_cell and len(is_border) - cell_start >= _MIN_CELL_WIDTH_PX:
        cells.append((cell_start, len(is_border)))

    # Lọc ô quá rộng so với median — thường là artifact từ vùng label
    if len(cells) >= 3:
        widths     = sorted(e - s for s, e in cells)
        median_w   = widths[len(widths) // 2]
        max_allowed = median_w * 2.5
        cells = [(s, e) for s, e in cells if (e - s) <= max_allowed]

    return cells


def _get_rec_model(ocr: PaddleOCR):
    """Lấy text recognition model từ bên trong PaddleOCR pipeline."""
    return ocr.paddlex_pipeline._pipeline.text_rec_model


def _recognize_cell(cell_img: np.ndarray, ocr: PaddleOCR) -> str:
    """
    Chạy recognition-only trực tiếp qua text_rec_model, bypass text detector.
    Upscale nếu ô quá nhỏ để recognition ổn định hơn.
    """
    h, w = cell_img.shape[:2]
    if h < _MIN_CELL_HEIGHT_PX and h > 0:
        scale    = _MIN_CELL_HEIGHT_PX / h
        cell_img = cv2.resize(cell_img,
                              (int(w * scale), _MIN_CELL_HEIGHT_PX),
                              interpolation=cv2.INTER_CUBIC)

    rec_model = _get_rec_model(ocr)
    results   = list(rec_model.predict([cell_img]))
    if not results:
        return ''

    rec   = results[0]
    text  = rec.get('rec_text', '').strip()
    score = rec.get('rec_score', 0)
    if score < _MIN_CELL_CONFIDENCE:
        print(f"      (ô trắng: score={score:.2f}, text='{text}') → '_'")
        return '_'
    return text


def extract_cell_row(img: np.ndarray,
                     label_block: Dict,
                     ocr: PaddleOCR) -> Optional[str]:
    """
    Entry point: tìm và nhận dạng dãy ô bảng bên phải label_block.

    Args:
        img:         ảnh gốc (BGR numpy array)
        label_block: block chứa nhãn field (có 'bbox', 'center_y', 'x_left')
        ocr:         PaddleOCR instance đã khởi tạo

    Returns:
        Chuỗi ký tự ghép từ các ô, hoặc None nếu không tìm được ô nào.
    """
    row_crop, _, _ = _crop_row_region(img, label_block)
    if row_crop.size == 0:
        return None

    cells = _find_cell_bounds(row_crop)
    if not cells:
        print("  [cell-extractor] Không tìm thấy ô nào trong vùng label")
        return None

    print(f"  [cell-extractor] Tìm thấy {len(cells)} ô, đang nhận dạng...")
    digits = []
    for x_start, x_end in cells:
        cell_img = row_crop[:, x_start:x_end]
        digit    = _recognize_cell(cell_img, ocr)
        if digit:
            digits.append(digit)
            print(f"    ô [{x_start}:{x_end}] → '{digit}'")

    if not digits:
        return None

    result = ''.join(digits)
    print(f"  [cell-extractor] Kết quả: '{result}'")
    return result
