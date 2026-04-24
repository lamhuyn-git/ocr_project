"""
block_merger.py — Gộp các OCR block bị break trên cùng 1 dòng

Vấn đề: PaddleOCR đôi khi tách 1 label dài thành 2–3 block nhỏ cùng dòng.
Ví dụ: "Số định danh cá nhân" + "của chủ hộ:" → 2 box thay vì 1

Giải pháp: gộp các block có center_y gần nhau (cùng dòng) và x_left liền kề
thành 1 block trước khi đưa vào KIE.

Output block gộp kế thừa:
  - text    : join bằng dấu cách
  - bbox    : bounding box bao trùm tất cả
  - center_y: trung bình center_y của các block con
  - x_left  : x_left nhỏ nhất (block bên trái nhất)
"""

from typing import Dict, List

SAME_LINE_THRESHOLD = 15   # pixel — giống spatial_lookup
MAX_GAP_MERGE      = 60    # pixel — khoảng cách x tối đa để gộp 2 block


def _bbox_xmax(block: Dict) -> float:
    return max(pt[0] for pt in block['bbox'])


def _merge_two(a: Dict, b: Dict) -> Dict:
    """Gộp 2 block thành 1, block a ở bên trái."""
    all_pts = a['bbox'] + b['bbox']
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]

    merged_bbox = [
        [min(xs), min(ys)],
        [max(xs), min(ys)],
        [max(xs), max(ys)],
        [min(xs), max(ys)],
    ]
    return {
        'text'    : a['text'] + ' ' + b['text'],
        'bbox'    : merged_bbox,
        'center_y': (a['center_y'] + b['center_y']) / 2,
        'x_left'  : min(a['x_left'], b['x_left']),
        # confidence: lấy min để an toàn
        'confidence': min(
            a.get('confidence', 1.0),
            b.get('confidence', 1.0)
        ),
    }


def merge_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    Gộp các block cùng dòng, liền kề nhau.

    Thuật toán:
      1. Sort block theo (center_y, x_left)
      2. Duyệt, nếu block kế tiếp cùng dòng VÀ khoảng cách x nhỏ
         → gộp vào block hiện tại
      3. Lặp lại đến khi không còn gộp được

    Chạy 2 pass để bắt trường hợp 3+ block liên tiếp.
    """
    if not blocks:
        return blocks

    changed = True
    result = sorted(blocks, key=lambda b: (b['center_y'], b['x_left']))

    while changed:
        changed = False
        merged = []
        skip = set()

        for i, block in enumerate(result):
            if i in skip:
                continue

            current = block
            for j in range(i + 1, len(result)):
                if j in skip:
                    continue
                nxt = result[j]

                # Kiểm tra cùng dòng
                if abs(nxt['center_y'] - current['center_y']) > SAME_LINE_THRESHOLD:
                    break  # các block đã sort theo y, dòng kế tiếp thì dừng

                # Kiểm tra khoảng cách x
                gap = nxt['x_left'] - _bbox_xmax(current)
                if gap <= MAX_GAP_MERGE:
                    current = _merge_two(current, nxt)
                    skip.add(j)
                    changed = True

            merged.append(current)

        result = merged

    return result
