import re
from typing import Dict, List
import numpy as np


_NEW_FIELD_PATTERN = re.compile(
    r'^\d{1,2}\.\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ]'
)


def get_xmax(block: Dict) -> float:
    return max(pt[0] for pt in block['bbox'])


def block_height(block: Dict) -> float:
    ys = [pt[1] for pt in block['bbox']]
    return max(ys) - min(ys)


def _starts_new_field(text: str) -> bool:
    """True nếu block bắt đầu bằng số thứ tự field, vd '3. Giới tính', '6. Email'."""
    return bool(_NEW_FIELD_PATTERN.match(text.strip()))


def x_overlaps(a: Dict, b: Dict) -> bool:
    return a['x_left'] <= get_xmax(b) and b['x_left'] <= get_xmax(a)


def get_thresholds_horizontal(
    blocks: List[Dict],
    img_width: int,
    img_height: int
) -> tuple:

    if blocks:
        heights = [block_height(b) for b in blocks]
        median_h = float(np.median(heights))
    else:
        median_h = img_height * 0.02

    same_line = median_h * 0.5

    max_gap = median_h * 1.0

    print(f"Median block height: {median_h:.1f}px")
    print(f"Same line threshold (y-axis): {same_line:.1f}px")
    print(f"Max gap for merging (x-axis): {max_gap:.1f}px")

    return same_line, max_gap


def merge_two(a: Dict, b: Dict) -> Dict:
    all_pts = a['bbox'] + b['bbox']
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]

    merged_bbox = [
        [min(xs), min(ys)],
        [max(xs), min(ys)],
        [max(xs), max(ys)],
        [min(xs), max(ys)],
    ]

    merged_text= a['text'] + ' ' + b['text']

    new_center_y = (a['center_y'] + b['center_y']) / 2

    new_x_left = min(a['x_left'], b['x_left'])

    new_confidence = min(a.get('confidence', 1.0), b.get('confidence', 1.0))

    return {
        'text'    : merged_text.strip(),
        'bbox'    : merged_bbox,
        'center_y': new_center_y,
        'x_left'  : new_x_left,
        'confidence': new_confidence,
    }


def get_threshold_vertical(img_height: int) -> float:
    threshold = img_height * 0.023
    print(f"  Vertical adjacent-line threshold (y-axis): {threshold:.1f}px")
    return threshold


def merge_blocks_vertical(blocks: List[Dict], img_height: int) -> List[Dict]:
    if not blocks:
        return blocks

    same_line_threshold = get_threshold_vertical(img_height)
    changed = True
    result = sorted(blocks, key=lambda b: b['center_y'])

    while changed:
        changed = False
        merged = []
        skip   = set()

        for i in range(len(result)):
            if i in skip:
                continue
            current = result[i]

            for j in range(i + 1, len(result)):
                if j in skip:
                    continue
                nxt = result[j]

                # Dừng khi block tiếp theo quá xa theo Y
                if abs(nxt['center_y'] - current['center_y']) >= same_line_threshold:
                    break

                # Gộp khi X giao nhau
                if x_overlaps(current, nxt):
                    current = merge_two(current, nxt)
                    skip.add(j)
                    changed = True

            merged.append(current)

        result = merged

    return result


def group_into_rows(blocks: List[Dict], same_line_threshold: float) -> List[List[Dict]]:
    sorted_blocks = sorted(blocks, key=lambda b: b['center_y'])
    rows: List[List[Dict]] = []
    current_row: List[Dict] = []

    for block in sorted_blocks:
        if not current_row:
            current_row.append(block)
        elif abs(block['center_y'] - current_row[0]['center_y']) <= same_line_threshold:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b['x_left']))
            current_row = [block]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b['x_left']))

    return rows


def merge_blocks_horizontal(blocks: List[Dict], img_width: int, img_height: int) -> List[Dict]:
    print("Merging blocks horizontally...")
    if not blocks:
        return blocks

    same_line_threshold, max_gap_merge = get_thresholds_horizontal(blocks=blocks, img_width=img_width, img_height=img_height)

    rows = group_into_rows(blocks, same_line_threshold)

    result: List[Dict] = []

    for row in rows:
        merged_row = [row[0]]
        for block in row[1:]:
            current = merged_row[-1]
            gap = block['x_left'] - get_xmax(current)
            dynamic_gap = max(max_gap_merge, block_height(current) * 1.0)
            if gap <= dynamic_gap and not _starts_new_field(block['text']):
                merged_row[-1] = merge_two(current, block)
            else:
                merged_row.append(block)
        result.extend(merged_row)

    return result
