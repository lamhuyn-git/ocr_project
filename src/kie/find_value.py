import re
from typing import Dict, List, Optional

from .find_label import find_label, KEYWORD_MAP, FUZZY_THRESHOLD, normalize

try:
    from rapidfuzz.fuzz import partial_ratio_alignment as _pra
    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False

# ── Ngưỡng không gian — đơn vị pixel ────────────────────────────
SAME_LINE_THRESHOLD  = 15   # chênh lệch center_y ≤ này → cùng dòng
BELOW_LINE_THRESHOLD = 80   # dòng dưới phải cách không quá này

# Pattern nhận diện điểm bắt đầu của field kế tiếp: "6. Email", "8. Mối quan hệ"...
_NEXT_FIELD_RE = re.compile(
    r'(?<!\d)\b\d{1,2}\.\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ]'
)


def _trim_at_next_label(text: str) -> str:
    """Cắt text tại vị trí gặp số thứ tự field kế tiếp.

    Ví dụ:
      "0911751863.ệ 6. Email:"   → "0911751863.ệ"
      "26/11/1984. 3. Giới tính" → "26/11/1984."
      "Ngô Mai Phương"           → "Ngô Mai Phương"  (không thay đổi)
    """
    m = _NEXT_FIELD_RE.search(text)
    if m:
        return text[:m.start()].strip().rstrip(',. ')
    return text


def _split_inline(text: str, label: str) -> Optional[str]:
    keywords = KEYWORD_MAP.get(label, [])
    norm_text = normalize(text)

    # Best case
    for kw in sorted(keywords, key=len, reverse=True):
        norm_kw = normalize(kw)
        if norm_kw in norm_text:
            idx   = norm_text.find(norm_kw)
            after = text[idx + len(norm_kw):]
            after = re.sub(r'^[\s:\-\(]+', '', after).strip()
            after = _trim_at_next_label(after)
            return after if after else None

    # Bad case: Use Fuzzy match
    if _FUZZY_AVAILABLE:
        best_score, best_end = 0, None
        for kw in sorted(keywords, key=len, reverse=True):
            norm_kw = normalize(kw)
            alignment = _pra(norm_kw, norm_text)
            if alignment.score > best_score:
                best_score = alignment.score
                best_end   = alignment.dest_end  # vị trí kết thúc keyword trong norm_text
        if best_score >= FUZZY_THRESHOLD and best_end is not None:
            after = text[best_end:]
            after = re.sub(r'^[\s:\-\(]+', '', after).strip()
            after = _trim_at_next_label(after)
            return after if after else None

    return None


def _merge_blocks_to_one(blocks: List[Dict], text: str) -> Dict:
    """Gộp nhiều block thành 1 block tổng hợp với text đã join."""
    all_pts = [pt for b in blocks for pt in b['bbox']]
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    return {
        'text'      : text,
        'confidence': min(b.get('confidence', 1.0) for b in blocks),
        'bbox'      : [[min(xs), min(ys)], [max(xs), min(ys)],
                       [max(xs), max(ys)], [min(xs), max(ys)]],
        'center_y'  : sum(b['center_y'] for b in blocks) / len(blocks),
        'x_left'    : min(b['x_left'] for b in blocks),
    }


def find_value(label_block: Dict, all_blocks: List[Dict], label: str, img_h: int) -> Optional[Dict]:
    """
    Trả về Dict chứa đầy đủ block info của value:
      {text, confidence, bbox, center_y, x_left}
    Trả về None nếu không tìm được value.
    """
    # Case 1: value nằm trong cùng block với label
    value_inline = _split_inline(label_block['text'], label)
    if value_inline:
        # Dùng lại metadata của label_block, chỉ thay text
        return {**label_block, 'text': value_inline}

    # ── Chuẩn bị ngưỡng động theo chiều cao ký tự ───────────────
    cur_block_cy   = label_block['center_y']
    cur_block_xmax = max(pt[0] for pt in label_block['bbox'])
    cur_block_xmin = min(pt[0] for pt in label_block['bbox'])

    same_line_thresh = img_h * 0.02
    below_thresh     = img_h * 0.023

    inline_candidates = []
    below_candidates  = []

    for block in all_blocks:
        if block is label_block:
            continue
        if find_label(block['text']) is not None:
            continue
        dy = block['center_y'] - cur_block_cy
        bx = block['x_left']

        if abs(dy) <= same_line_thresh and bx > cur_block_xmax - 10:
            inline_candidates.append(block)
        elif 0 < dy <= below_thresh and bx >= cur_block_xmin - 20:
            below_candidates.append(block)

    # Case 2: value nằm cùng dòng bên phải
    if inline_candidates:
        inline_candidates.sort(key=lambda b: b['x_left'])
        joined = ' '.join(b['text'] for b in inline_candidates).strip()
        joined = _trim_at_next_label(joined)
        return _merge_blocks_to_one(inline_candidates, joined)

    # Case 3: value nằm dòng dưới
    if below_candidates:
        below_candidates.sort(key=lambda b: b['center_y'])
        nearest_y = below_candidates[0]['center_y']
        same_row  = [b for b in below_candidates
                     if abs(b['center_y'] - nearest_y) <= same_line_thresh]
        same_row.sort(key=lambda b: b['x_left'])
        joined = ' '.join(b['text'] for b in same_row).strip()
        joined = _trim_at_next_label(joined)
        return _merge_blocks_to_one(same_row, joined)

    return None
