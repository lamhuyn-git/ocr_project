"""
visualize_roi_overlay.py — Vẽ ROI + tên field lên ảnh canonical/warped để kiểm mắt thường.

Dùng kiểm xem config (roi_norm + padding) có trùng đúng vùng value của từng field không.
Mặc định vẽ lên chính reference; truyền --image để vẽ lên ảnh đã warp (Phase 01).

Chạy:
  .venv/bin/python src/config_detection/tools/visualize_roi_overlay.py
  .venv/bin/python src/config_detection/tools/visualize_roi_overlay.py \
      --image real_test/outputs/alignment/test_alignment_001_warped.jpg \
      --quality poor
"""
import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config_detection import load_config, field_roi_pixels  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "templates", "ct01_v1.0.yaml")
DEFAULT_IMAGE = os.path.join(ROOT, "assets", "ct01_reference.jpg")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "config_debug", "roi_overlay.jpg")

# Màu theo type (BGR) để dễ phân biệt nhóm field.
TYPE_COLORS = {
    "text_line": (0, 180, 0),
    "text_block": (0, 140, 255),
    "date": (255, 0, 0),
    "number": (0, 0, 255),
    "enum": (200, 0, 200),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="Ảnh canonical/warped để overlay.")
    ap.add_argument("--quality", default=None, choices=["good", "medium", "poor"],
                    help="Áp quality_overrides (nới padding) khi vẽ.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    config = load_config(args.config)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.image}")

    h, w = img.shape[:2]
    cw = config["canonical_size"]["width"]
    ch = config["canonical_size"]["height"]
    if (w, h) != (cw, ch):
        print(f"[cảnh báo] ảnh {w}x{h} khác canonical {cw}x{ch} — "
              f"ROI vẫn vẽ theo tỉ lệ ảnh hiện tại.")

    for name, field in config["fields"].items():
        color = TYPE_COLORS.get(field["type"], (128, 128, 128))
        x1, y1, x2, y2 = field_roi_pixels(config, name, w, h, args.quality)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Đã vẽ {len(config['fields'])} ROI → {args.out}")


if __name__ == "__main__":
    main()
