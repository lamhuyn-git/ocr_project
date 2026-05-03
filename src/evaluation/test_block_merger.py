"""
test_block_merger.py — Test merge pipeline và visualize kết quả

Cách chạy (từ thư mục gốc project):
    # 1 ảnh
    python -m src.evaluation.test_block_merger --image image_test/scan/scan_002.jpg

    # Toàn bộ 1 thư mục
    python -m src.evaluation.test_block_merger --dir image_test/scan

    # Toàn bộ image_test (tất cả subfolder)
    python -m src.evaluation.test_block_merger --dir image_test

Lưu ảnh vào outputs/test_results/<tên_ảnh>_before_merge.jpg và _after_merge.jpg
"""

import sys
import os
import argparse
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocess.preprocess      import preprocess_pipeline
from src.recognition.engine        import get_ocr_instance, run_ocr, filter_by_confidence
from src.kie.block_merger          import merge_blocks_horizontal, merge_blocks_vertical
from src.kie.keyword_matcher       import match_keyword
from src.recognition.visualize     import draw_bounding_boxes

GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
RED   = "\033[91m"; RESET  = "\033[0m";  BOLD = "\033[1m"

OUTPUT_DIR  = "outputs/test_results"
EXTENSIONS  = {'.jpg', '.jpeg', '.png'}


def collect_images(path: str) -> list:
    """Thu thập tất cả ảnh từ file hoặc thư mục (recursive)."""
    if os.path.isfile(path):
        return [path]
    images = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in EXTENSIONS:
                images.append(os.path.join(root, f))
    return images


def print_blocks(blocks, label=""):
    if label:
        print(f"\n{BOLD}{label}{RESET}")
    print(f"  {'#':<4} {'text':<50} {'x_left':>7} {'center_y':>9} {'match'}")
    print(f"  {'-'*4} {'-'*50} {'-'*7} {'-'*9} {'-'*25}")
    for i, b in enumerate(blocks):
        field = match_keyword(b['text']) or ''
        color = GREEN if field else RESET
        print(f"  {i:<4} {color}{b['text'][:50]:<50}{RESET} {b['x_left']:>7.0f} "
              f"{b['center_y']:>9.1f} {CYAN}{'→ ' + field if field else ''}{RESET}")


def print_summary(before, after):
    """In tóm tắt: block nào được gộp."""
    print(f"\n  {BOLD}{len(before)} blocks → {len(after)} blocks "
          f"(gộp {len(before) - len(after)}){RESET}")
    before_texts = {b['text'] for b in before}
    for b in after:
        if b['text'] not in before_texts:
            field = match_keyword(b['text']) or '—'
            print(f"  {GREEN}✓ \"{b['text'][:60]}\"{RESET}")
            print(f"    → field: {CYAN}{field}{RESET}")


def run_image(image_path: str, ocr):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  {image_path}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    # Preprocess
    img = preprocess_pipeline(image_path)
    h, w = img.shape[:2]

    # OCR → raw blocks
    raw = run_ocr(ocr, img)
    if not raw:
        print(f"  {RED}Không tìm thấy text!{RESET}")
        return

    # Filter → merge horizontal → merge vertical
    filtered = filter_by_confidence(raw, min_confidence=0.5)
    h_merged = merge_blocks_horizontal(filtered,  img_width=w)
    merged   = merge_blocks_vertical(h_merged,    img_height=h)

    print_blocks(filtered, "TRƯỚC KHI MERGE:")
    print_blocks(merged,   "SAU KHI MERGE:")
    print_summary(filtered, merged)

    # Lưu ảnh before/after
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base        = os.path.splitext(os.path.basename(image_path))[0]
    before_path = os.path.join(OUTPUT_DIR, f"{base}_before_merge.jpg")
    after_path  = os.path.join(OUTPUT_DIR, f"{base}_after_merge.jpg")

    cv2.imwrite(before_path, draw_bounding_boxes(img.copy(), filtered))
    cv2.imwrite(after_path,  draw_bounding_boxes(img.copy(), merged))

    print(f"\n  {GREEN}Đã lưu:{RESET}")
    print(f"  {CYAN}{before_path}{RESET}")
    print(f"  {CYAN}{after_path}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Path 1 ảnh CT01')
    group.add_argument('--dir',   help='Thư mục chứa ảnh (recursive)')
    args = parser.parse_args()

    path   = args.image or args.dir
    images = collect_images(path)

    if not images:
        print(f"{RED}Không tìm thấy ảnh nào trong: {path}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Tìm thấy {len(images)} ảnh — khởi tạo OCR model...{RESET}")
    # Khởi tạo OCR 1 lần duy nhất, dùng lại cho tất cả ảnh
    ocr = get_ocr_instance()

    for img_path in images:
        try:
            run_image(img_path, ocr)
        except Exception as e:
            print(f"  {RED}Lỗi {img_path}: {e}{RESET}")

    print(f"\n{BOLD}Xong! Kết quả lưu tại: {OUTPUT_DIR}/{RESET}\n")
