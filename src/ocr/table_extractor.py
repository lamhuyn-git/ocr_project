"""
table_extractor.py — Trích xuất bảng "thành viên cùng thay đổi" (Phase 03b).

BƯỚC A (hiện tại):
  - remove_rulings(): khử đường kẻ ngang+dọc của bảng → còn lại "mực chữ".
  - has_ink():        kiểm tra vùng (đã khử kẻ) có nội dung không.
  - extract_table():  bảng TRỐNG → trả empty NGAY (fast path, không OCR);
                      bảng CÓ chữ → tạm gom block theo hàng (rows).

BƯỚC B (để dành — cần mẫu bảng đã điền để verify):
  - tách lưới ô theo config (columns × rows) → OCR từng ô → map mỗi hàng thành object thành viên.
"""
from typing import Dict, Optional

import cv2
import numpy as np

from config_detection.roi_calculator import field_roi_pixels

from .crop_ocr import crop_roi, ocr_crop, _group_lines

# Ngưỡng số pixel "mực" tối thiểu để coi 1 vùng (đã khử kẻ + lọc nhiễu) là có nội dung.
DEFAULT_MIN_INK = 60
# Diện tích tối thiểu (px) của 1 blob để coi là chữ thật.
DEFAULT_MIN_BLOB_AREA = 20
# Chiều cao tối thiểu (px) của 1 blob — gạch/chấm hướng dẫn trong ô rất THẤP (~2-5px),
# chữ thật cao hơn nhiều (~15-35px) → lọc theo chiều cao loại bỏ được dấu chấm in sẵn.
DEFAULT_MIN_BLOB_HEIGHT = 10
# Tỉ lệ chiều cao phần header (hàng tiêu đề cột) — bỏ khi kiểm tra nội dung thân bảng.
DEFAULT_HEADER_H_FRAC = 0.20


def _drop_small_blobs(
    mask: np.ndarray,
    min_area: int = DEFAULT_MIN_BLOB_AREA,
    min_height: int = DEFAULT_MIN_BLOB_HEIGHT,
) -> np.ndarray:
    """
    Giữ blob trông giống NÉT CHỮ; bỏ blob nhỏ hoặc thấp-dẹt.

    Bỏ nếu: diện tích < min_area  HOẶC  chiều cao < min_height.
    → loại mảnh kẻ sót, nhiễu, và DẤU CHẤM/GẠCH hướng dẫn in sẵn trong ô (rất thấp).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):                       # bỏ label 0 = nền
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= min_area and height >= min_height:
            out[labels == i] = 255
    return out


def remove_rulings(crop: np.ndarray) -> np.ndarray:
    """
    Khử đường kẻ ngang + dọc của bảng, trả ảnh nhị phân chỉ còn 'mực chữ' (foreground=255).

    Ý tưởng: đường kẻ là đoạn liền DÀI → morphology OPEN với kernel dài tách riêng được;
    nét chữ ngắn nên không bị bắt → trừ kẻ khỏi ảnh nhị phân là còn lại chữ.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Nhị phân hoá: mực (chữ + kẻ) thành trắng (255) trên nền đen
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape

    # Tách kẻ NGANG: kernel ngang dài (~1/15 bề ngang bảng)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 15), 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

    # Tách kẻ DỌC: kernel dọc dài (~1/15 chiều cao bảng)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 15)))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    rulings = cv2.bitwise_or(horizontal, vertical)
    text_only = cv2.subtract(bw, rulings)       # bỏ kẻ, còn lại chữ
    text_only = cv2.medianBlur(text_only, 3)    # dọn nhiễu hạt nhỏ còn sót
    text_only = _drop_small_blobs(text_only)    # bỏ mảnh kẻ sót / blob nhỏ
    return text_only


def has_ink(binary_region: np.ndarray, min_pixels: int = DEFAULT_MIN_INK) -> bool:
    """True nếu vùng nhị phân (đã khử kẻ) có đủ pixel mực → coi như có nội dung."""
    if binary_region.size == 0:
        return False
    return int((binary_region > 0).sum()) >= min_pixels


def extract_table(
    warped: np.ndarray,
    config: dict,
    name: str,
    threshold: float,
    quality: Optional[str],
    preprocess: bool,
) -> Dict:
    h, w = warped.shape[:2]
    box = field_roi_pixels(config, name, w, h, quality)
    crop = crop_roi(warped, box)

    table_cfg = config["fields"][name].get("table", {})
    header_h_frac = table_cfg.get("header_h_frac", DEFAULT_HEADER_H_FRAC)
    min_ink = table_cfg.get("min_ink", DEFAULT_MIN_INK)

    # Khử kẻ rồi xét phần THÂN bảng (bỏ hàng tiêu đề) có nội dung không
    text_only = remove_rulings(crop)
    body = text_only[int(header_h_frac * text_only.shape[0]):, :]

    if not has_ink(body, min_ink):
        # FAST PATH: bảng trống → trả empty, KHÔNG chạy OCR
        return {
            "type": "table",
            "rows": [],
            "confidence": 0.0,
            "low_confidence": False,
            "empty": True,
            "n_blocks": 0,
            "bbox": list(box),
        }

    # Bảng CÓ nội dung: (Bước A) gom block theo hàng — tách ô/object để Bước B.
    blocks = ocr_crop(crop, box_offset=(box[0], box[1]), preprocess=preprocess)
    rows = []
    for line in _group_lines(blocks):
        line_conf = sum(b["confidence"] for b in line) / len(line)
        rows.append({
            "text": " ".join(b["text"] for b in line),
            "confidence": round(line_conf, 4),
            "cells": [
                {"text": b["text"], "confidence": b["confidence"], "bbox": b["bbox"]}
                for b in line
            ],
        })

    avg_conf = round(sum(b["confidence"] for b in blocks) / len(blocks), 4) if blocks else 0.0
    return {
        "type": "table",
        "rows": rows,
        "confidence": avg_conf,
        "low_confidence": (avg_conf < threshold) if blocks else False,
        "empty": not blocks,
        "n_blocks": len(blocks),
        "bbox": list(box),
    }
