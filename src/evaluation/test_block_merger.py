"""
test_block_merger.py — Test merge_blocks() và visualize kết quả

Cách chạy (từ thư mục gốc project):
    python -m src.evaluation.test_block_merger --image image_test/phone_good/phone_good_001.jpg

Lưu 2 ảnh vào outputs/test/:
    before_merge.jpg  — raw OCR blocks
    after_merge.jpg   — sau khi gộp
"""

import sys
import os
import argparse
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.kie.block_merger    import merge_blocks
from src.kie.keyword_matcher import match_keyword
from src.recognition.visualize import draw_bounding_boxes

GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
RED   = "\033[91m"; RESET  = "\033[0m";  BOLD = "\033[1m"

OUTPUT_DIR = "outputs/test"


def print_blocks(blocks, label=""):
    if label:
        print(f"\n{BOLD}{label}{RESET}")
    print(f"  {'#':<4} {'text':<45} {'x_left':>7} {'center_y':>9} {'match'}")
    print(f"  {'-'*4} {'-'*45} {'-'*7} {'-'*9} {'-'*30}")
    for i, b in enumerate(blocks):
        field = match_keyword(b['text']) or ''
        color = GREEN if field else RESET
        print(f"  {i:<4} {color}{b['text']:<45}{RESET} {b['x_left']:>7.0f} "
              f"{b['center_y']:>9.1f} {CYAN}{'→ ' + field if field else ''}{RESET}")


def print_diff(before, after):
    print(f"\n  {BOLD}{len(before)} blocks → {len(after)} blocks "
          f"(gộp {len(before)-len(after)}){RESET}")
    before_texts = set(b['text'] for b in before)
    for b in after:
        if b['text'] not in before_texts:
            field = match_keyword(b['text']) or 'không match'
            print(f"  {GREEN}✓ \"{b['text']}\"{RESET}")
            print(f"    → field: {CYAN}{field}{RESET}")


def save_images(img, raw_blocks, merged_blocks, image_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    before_img = draw_bounding_boxes(img.copy(), raw_blocks)
    after_img  = draw_bounding_boxes(img.copy(), merged_blocks)

    before_path = os.path.join(OUTPUT_DIR, f"{base}_before_merge.jpg")
    after_path  = os.path.join(OUTPUT_DIR, f"{base}_after_merge.jpg")

    cv2.imwrite(before_path, before_img)
    cv2.imwrite(after_path,  after_img)

    print(f"\n  {GREEN}Đã lưu:{RESET}")
    print(f"  {CYAN}{before_path}{RESET}  ← raw OCR blocks")
    print(f"  {CYAN}{after_path}{RESET}   ← sau khi merge")


def run_image(image_path):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  TEST BLOCK MERGER — {image_path}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    from src.preprocess.preprocess import preprocess_pipeline
    from src.recognition.engine    import engine_pipeline

    print("\n  Đang chạy OCR...")
    img        = preprocess_pipeline(image_path)
    raw_blocks = engine_pipeline(img)
    print(f"  OCR xong: {len(raw_blocks)} blocks")

    merged_blocks = merge_blocks(raw_blocks)

    print_blocks(raw_blocks,    "TRƯỚC KHI MERGE:")
    print_blocks(merged_blocks, "SAU KHI MERGE:")
    print_diff(raw_blocks, merged_blocks)

    save_images(img, raw_blocks, merged_blocks, image_path)
    print(f"\n{BOLD}{'='*70}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path ảnh CT01')
    args = parser.parse_args()
    run_image(args.image)
