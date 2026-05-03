from typing import Dict, List


def get_xmax(block: Dict) -> float:
    return max(pt[0] for pt in block['bbox'])


def _block_height(block: Dict) -> float:
    ys = [pt[1] for pt in block['bbox']]
    return max(ys) - min(ys)


def x_overlaps(a: Dict, b: Dict) -> bool:
    return a['x_left'] <= get_xmax(b) and b['x_left'] <= get_xmax(a)


def get_thresholds_horizontal(img_width: int,img_height: int ) -> tuple:
    print(f"Calculating thresholds based on image width: {img_width}px")
    same_line = img_height * 0.02
    max_gap   = img_width * 0.02
    print(f"  Same line threshold (y-axis): {same_line:.1f}px")
    print(f"  Max gap for merging (x-axis): {max_gap:.1f}px")
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


def merge_blocks_horizontal(blocks: List[Dict], img_width: int, img_height:int) -> List[Dict]:
    print("Merging blocks horizontally...")
    if not blocks:
        return blocks

    same_line_threshold, max_gap_merge = get_thresholds_horizontal(img_width=img_width, img_height=img_height)
    changed = True
    
    print(f"Initial blocks before horizontal merge: {blocks}")
    result = sorted(blocks, key=lambda b: (b['center_y'], b['x_left']))
    print(f"result after sorting: {result}")

    while changed:
        changed = False
        merged = []
        skip = set()

        for i in range(len(result)):
            block = result[i]
            if i in skip:
                continue
            current = block

            for j in range(i + 1, len(result)):
                if j in skip:
                    continue
                nxt = result[j]

                if abs(nxt['center_y'] - current['center_y']) > same_line_threshold:
                    break

                gap = abs(nxt['x_left'] - get_xmax(current))
                # Scale max_gap by block height so table-cell digits (tall relative
                # to their gap) merge correctly without inflating global threshold.
                dynamic_gap = max(max_gap_merge, _block_height(current) * 1.2)
                if gap > dynamic_gap:
                    continue
                current = merge_two(current, nxt)
                skip.add(j)
                changed = True

            merged.append(current)

        result = merged

    return result
