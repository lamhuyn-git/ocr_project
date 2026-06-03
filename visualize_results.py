"""
visualize_results.py — Vẽ kết quả OCR (box ROI + text) lên ảnh đã warp để kiểm bằng mắt.

Đọc JSON đã sinh trong --json-dir + ảnh gốc trong --img-dir → align → overlay → lưu *_viz.jpg.
Màu: xanh = ok, cam = low-confidence, xám = field rỗng.

Chạy:
  python visualize_results.py --img-dir test_image --json-dir debug_output/outputs/test_results
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from alignment import align_form  # noqa: E402

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Màu RGB
GREEN = (0, 170, 0)       # ok
ORANGE = (230, 130, 0)    # low-confidence
GRAY = (150, 150, 150)    # rỗng


def _find_image(img_dir: str, base: str):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def _label_for(field: dict) -> str:
    """Chuỗi hiển thị cho 1 field."""
    if field["type"] == "table":
        if field["empty"]:
            return "[table: empty]"
        return f"[table: {field['n_blocks']} block]"
    return field["text"] or "(rỗng)"


def visualize_one(image_path: str, fields: dict, font, font_small, out_path: str) -> None:
    img = cv2.imread(image_path)
    warped, _ = align_form(img)
    pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    for name, r in fields.items():
        x1, y1, x2, y2 = r["bbox"]
        if r["empty"]:
            color = GRAY
        elif r["low_confidence"]:
            color = ORANGE
        else:
            color = GREEN

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Text value (cắt ngắn) đặt ngay trên box
        value = _label_for(r)
        if len(value) > 45:
            value = value[:45] + "…"
        label = f"{name}: {value}"
        ty = max(0, y1 - 20)
        # nền mờ cho dễ đọc
        tb = draw.textbbox((x1, ty), label, font=font_small)
        draw.rectangle(tb, fill=(255, 255, 255))
        draw.text((x1, ty), label, fill=color, font=font_small)

    pil.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="test_image")
    ap.add_argument("--json-dir", default="debug_output/outputs/test_results")
    ap.add_argument("--out-dir", default=None, help="mặc định = json-dir")
    args = ap.parse_args()

    out_dir = args.out_dir or args.json_dir
    os.makedirs(out_dir, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, 22)
    font_small = ImageFont.truetype(FONT_PATH, 16)

    json_files = sorted(glob.glob(os.path.join(args.json_dir, "*_fields.json")))
    print(f"Trực quan hoá {len(json_files)} kết quả → {out_dir}\n")

    ok = 0
    for i, jf in enumerate(json_files, 1):
        base = os.path.basename(jf).replace("_fields.json", "")
        img_path = _find_image(args.img_dir, base)
        if img_path is None:
            print(f"[{i}] ✗ {base}: không thấy ảnh gốc")
            continue
        with open(jf, encoding="utf-8") as f:
            fields = json.load(f)
        out_path = os.path.join(out_dir, f"{base}_viz.jpg")
        try:
            visualize_one(img_path, fields, font, font_small, out_path)
            ok += 1
            print(f"[{i}/{len(json_files)}] ✓ {base} → {os.path.basename(out_path)}")
        except Exception as e:
            print(f"[{i}] ✗ {base}: {e}")

    print(f"\nXong: {ok}/{len(json_files)} → {out_dir}")


if __name__ == "__main__":
    main()
