"""
digit_grid.py — Đọc field "số định danh" in trong LƯỚI Ô.

Vạch ngăn ô là "mực" giống nét chữ → OCR đọc vạch thành ký tự (thừa/nhầm) hoặc làm vỡ chữ số.
Cách xử lý: XOÁ vạch lưới (giữ chữ số trên nền trắng) rồi OCR cả dải (whole-strip).
"""
import re

import cv2
import numpy as np

from .crop_ocr import crop_roi, ocr_crop, join_blocks


def remove_grid_lines(crop):
    """Xoá vạch lưới (dọc + ngang) khỏi crop, trả ảnh BGR đã làm sạch (vạch → nền trắng)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 3)))      # vạch dọc dài
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 15), 1))    # vạch ngang dài
    lines = cv2.bitwise_or(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk),
    )
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)        # phủ hết bề rộng vạch
    cleaned = gray.copy()
    cleaned[lines > 0] = 255                                                  # xoá vạch → trắng
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _paddle_strip_reader(img):
    """Đọc cả dải bằng PaddleOCR → (text, conf)."""
    blocks = ocr_crop(img, box_offset=(0, 0))
    return join_blocks(blocks)


def recognize_digit_grid(warped, box, n_cells=12, reader=_paddle_strip_reader):
    """
    box: (x1,y1,x2,y2) ROI dải số trên ảnh warped.
    Xoá vạch lưới rồi OCR cả dải. Trả (text, conf). n_cells chỉ để tham chiếu độ dài kỳ vọng.
    """
    crop = crop_roi(warped, box)
    if crop is None or crop.size == 0:
        return "", 0.0
    cleaned = remove_grid_lines(crop)
    return reader(cleaned)
