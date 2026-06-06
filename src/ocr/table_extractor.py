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
from .normalizer import apply_normalizers

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


# Tên 6 cột mặc định (trái → phải) nếu config không khai báo.
DEFAULT_COLUMNS = ["tt", "ho_ten", "ngay_sinh", "gioi_tinh", "so_dinh_danh", "quan_he"]


def _vertical_ruling_xs(crop: np.ndarray) -> list:
    """Trả x-tâm các đường kẻ DỌC của bảng (= ranh giới cột)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape
    # Kẻ dọc = đoạn liền dài (>= ~1/3 chiều cao bảng)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 3)))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    col_sum = (vertical > 0).sum(axis=0)            # mật độ mực dọc theo từng x
    thr = max(5, int(0.3 * h))                       # x là kẻ nếu phủ >= 30% chiều cao
    xs = np.where(col_sum >= thr)[0]
    if len(xs) == 0:
        return []
    # Gom các x liền nhau thành 1 đường kẻ, lấy tâm
    groups, start, prev = [], int(xs[0]), int(xs[0])
    gap = max(3, w // 100)
    for x in xs[1:]:
        x = int(x)
        if x - prev > gap:
            groups.append((start + prev) // 2)
            start = x
        prev = x
    groups.append((start + prev) // 2)
    return groups


def _detect_col_bounds(crop: np.ndarray, n_cols: int, fallback_fracs=None):
    """Ranh cột (n_cols+1 mốc, px crop-relative). Ưu tiên kẻ dọc; thiếu → fallback config/đều."""
    w = crop.shape[1]
    xs = [x for x in _vertical_ruling_xs(crop) if 2 < x < w - 2]
    bounds = sorted(set([0] + xs + [w]))
    if len(bounds) >= n_cols + 1:
        idx = np.linspace(0, len(bounds) - 1, n_cols + 1).round().astype(int)
        return [bounds[i] for i in idx], "rulings"
    if fallback_fracs and len(fallback_fracs) == n_cols + 1:
        return [int(f * w) for f in fallback_fracs], "config"
    return [int(round(i * w / n_cols)) for i in range(n_cols + 1)], "even"


def _assemble_members(rows: list, bounds_abs: list, col_names: list, cell_norm=None) -> list:
    """Gán cell→cột theo center_x (toạ độ warped tuyệt đối); bỏ hàng rỗng; chuẩn hoá theo cột."""
    n = len(col_names)
    cell_norm = cell_norm or {}
    members = []
    for ri, row in enumerate(rows, 1):
        cols = {c: [] for c in col_names}
        for cell in row["cells"]:
            x1, _, x2, _ = cell["bbox"]
            cx = (x1 + x2) / 2
            ci = n - 1
            for k in range(n):
                if cx < bounds_abs[k + 1]:
                    ci = k
                    break
            cols[col_names[ci]].append((x1, cell["text"]))
        merged = {c: " ".join(t for _, t in sorted(v)).strip() for c, v in cols.items()}
        if not any(merged.values()):
            continue                                # hàng rỗng → bỏ
        # chuẩn hoá từng cột (nếu config khai báo cell_normalize)
        merged = {c: apply_normalizers(v, cell_norm.get(c)) for c, v in merged.items()}
        if "tt" in merged and not merged["tt"]:
            merged["tt"] = str(ri)                  # TT không đọc được → số thứ tự
        members.append(merged)
    return members


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
            "members": [],
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

    # Bước B: gán cell → 6 cột (dò kẻ dọc, fallback config) → list thành viên
    col_names = table_cfg.get("columns", DEFAULT_COLUMNS)
    bounds_rel, src = _detect_col_bounds(crop, len(col_names), table_cfg.get("col_x_frac"))
    bounds_abs = [b + box[0] for b in bounds_rel]          # crop-relative → warped tuyệt đối
    members = _assemble_members(rows, bounds_abs, col_names, table_cfg.get("cell_normalize"))

    avg_conf = round(sum(b["confidence"] for b in blocks) / len(blocks), 4) if blocks else 0.0
    return {
        "type": "table",
        "members": members,
        "rows": rows,
        "col_source": src,        # 'rulings' | 'config' | 'even' — debug nguồn ranh cột
        "confidence": avg_conf,
        "low_confidence": (avg_conf < threshold) if blocks else False,
        "empty": not members,
        "n_blocks": len(blocks),
        "bbox": list(box),
    }
