"""
spatial_lookup.py — Lớp 2: Tìm value của label dựa trên vị trí bbox

Input : label_block (block chứa nhãn field) + toàn bộ OCR blocks
Output: chuỗi text là giá trị của field đó, hoặc None

3 chiến lược theo thứ tự ưu tiên:
  1. Tách value từ trong cùng block  →  "Họ và tên: Nguyễn Văn A"
  2. Block cùng dòng, bên phải label →  label | value
  3. Block dòng ngay bên dưới        →  label
                                         value
"""

import re
from typing import Dict, List, Optional

from .keyword_matcher import KEYWORD_MAP, normalize

# ── Ngưỡng không gian — đơn vị pixel ────────────────────────────
SAME_LINE_THRESHOLD  = 15   # chênh lệch center_y ≤ này → cùng dòng
BELOW_LINE_THRESHOLD = 80   # dòng dưới phải cách không quá này


def _bbox_height(block: Dict) -> float:
    ys = [pt[1] for pt in block['bbox']]
    return max(ys) - min(ys)


def _split_inline(text: str, field: str) -> Optional[str]:
    """
    Chiến lược 1: tách value từ block chứa cả label + value.
    Ví dụ: "Họ và tên: Nguyễn Văn A"  →  "Nguyễn Văn A"

    Tìm keyword dài nhất trước để tránh cắt nhầm ở keyword ngắn hơn.
    """
    keywords = KEYWORD_MAP.get(field, [])
    norm_text = normalize(text)

    for kw in sorted(keywords, key=len, reverse=True):
        norm_kw = normalize(kw)
        if norm_kw in norm_text:
            idx   = norm_text.find(norm_kw)
            after = text[idx + len(norm_kw):]
            after = re.sub(r'^[\s:\-\(]+', '', after).strip()
            return after if after else None

    return None


def find_value(label_block: Dict, all_blocks: List[Dict]) -> Optional[str]:
    """
    Lớp 2 entry point: tìm value cho một label_block.
    """
    from .keyword_matcher import match_keyword
    field = match_keyword(label_block['text'])

    # ── Chiến lược 1: value nằm trong cùng block ────────────────
    if field:
        inline = _split_inline(label_block['text'], field)
        if inline:
            return inline

    # ── Chuẩn bị ngưỡng động theo chiều cao ký tự ───────────────
    label_cy   = label_block['center_y']
    label_xmax = max(pt[0] for pt in label_block['bbox'])
    label_xmin = min(pt[0] for pt in label_block['bbox'])
    label_h    = _bbox_height(label_block)

    same_line_thresh = max(SAME_LINE_THRESHOLD, label_h * 0.5)
    below_thresh     = max(BELOW_LINE_THRESHOLD, label_h * 2.5)

    inline_candidates = []
    below_candidates  = []

    for block in all_blocks:
        if block is label_block:
            continue
        dy = block['center_y'] - label_cy
        bx = block['x_left']

        # ── Chiến lược 2: cùng dòng, bên phải ───────────────────
        if abs(dy) <= same_line_thresh and bx > label_xmax - 10:
            inline_candidates.append(block)

        # ── Chiến lược 3: dòng ngay dưới ────────────────────────
        elif 0 < dy <= below_thresh and bx >= label_xmin - 20:
            below_candidates.append(block)

    if inline_candidates:
        inline_candidates.sort(key=lambda b: b['x_left'])
        return ' '.join(b['text'] for b in inline_candidates).strip()

    if below_candidates:
        below_candidates.sort(key=lambda b: b['center_y'])
        nearest_y = below_candidates[0]['center_y']
        same_row  = [b for b in below_candidates
                     if abs(b['center_y'] - nearest_y) <= same_line_thresh]
        same_row.sort(key=lambda b: b['x_left'])
        return ' '.join(b['text'] for b in same_row).strip()

    return None
