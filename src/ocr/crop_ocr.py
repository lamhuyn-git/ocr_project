from typing import Dict, List, Tuple
from PIL.ImageOps import crop
import cv2
import numpy as np
from .engine import get_ocr_instance, run_ocr, reset_instance


def crop_roi(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]      # numpy slicing: [hàng y, cột x]


def optional_preprocess(crop: np.ndarray, min_height: int = 48) -> np.ndarray:
    if crop.size == 0:
        return crop
    h = crop.shape[0]
    if 0 < h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _map_point_to_canvas(point, scale_x, scale_y, offset_x, offset_y):
    x, y = point
    return [x * scale_x + offset_x, y * scale_y + offset_y]


def ocr_crop(crop, box_offset=(0, 0), preprocess=True, model_version=None):
    if crop is None or crop.size == 0:
        return []

    # Nếu ảnh crop quá nhỏ, scale lên cho OCR dễ đọc hơn.
    if preprocess:
        image_to_ocr = optional_preprocess(crop)
    else:
        image_to_ocr = crop

    blocks = run_ocr(get_ocr_instance(model_version), image_to_ocr)

    # Tính tỉ lệ đưa toạ độ từ ảnh-đã-OCR về crop gốc.
    scale_x = crop.shape[1] / image_to_ocr.shape[1]
    scale_y = crop.shape[0] / image_to_ocr.shape[0]
    offset_x, offset_y = box_offset
    for b in blocks:
        b["bbox"] = [
            _map_point_to_canvas(pt, scale_x, scale_y, offset_x, offset_y)
            for pt in b["bbox"]
        ]
        b["center_y"] = b["center_y"] * scale_y + offset_y
        b["x_left"] = b["x_left"] * scale_x + offset_x

    return blocks


def _group_lines(blocks: List[Dict], y_tol: int = 14) -> List[List[Dict]]:
    lines: List[List[Dict]] = []
    for b in blocks:
        same_line = lines and abs(b["center_y"] - lines[-1][-1]["center_y"]) <= y_tol
        if same_line:
            lines[-1].append(b)         
        else:
            lines.append([b])           

    for line in lines:
        line.sort(key=lambda b: b["x_left"])
    return lines


def join_blocks(blocks: List[Dict], line_sep: str = "\n", token_sep: str = " ") -> Tuple[str, float]:
    if not blocks:
        return "", 0.0

    lines = _group_lines(blocks)
    text = line_sep.join(token_sep.join(b["text"] for b in line) for line in lines)

    total_chars = sum(len(b["text"]) for b in blocks) or 1   # 'or 1' tránh chia 0
    weighted_sum = sum(b["confidence"] * len(b["text"]) for b in blocks)
    avg_conf = weighted_sum / total_chars

    return text, round(avg_conf, 4)
