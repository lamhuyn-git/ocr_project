"""
visualize.py — Vẽ bounding box + nhãn lên ảnh kết quả OCR
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 13) -> ImageFont.ImageFont:
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_bounding_boxes(img: np.ndarray, ocr_results: list) -> np.ndarray:
    # PIL dùng RGB, OpenCV dùng BGR → chuyển đổi
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = _load_font()

    for item in ocr_results:
        bbox       = item['bbox']
        text       = item['text']
        confidence = item['confidence']

        if confidence >= 0.85:
            color = (0, 200, 0)      # Xanh lá — tin cậy cao
        elif confidence >= 0.65:
            color = (255, 180, 0)    # Vàng — trung bình
        else:
            color = (220, 0, 0)      # Đỏ — thấp

        pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
        draw.polygon(pts, outline=color + (255,))
        for i in range(len(pts)):
            draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=color, width=1)

        label    = f"{text} ({confidence:.0%})"
        origin_x = int(min(pt[0] for pt in bbox))
        origin_y = max(int(min(pt[1] for pt in bbox)) - 16, 2)

        bbox_text = font.getbbox(label)
        text_w    = bbox_text[2] - bbox_text[0]
        text_h    = bbox_text[3] - bbox_text[1]
        draw.rectangle(
            [origin_x, origin_y, origin_x + text_w + 4, origin_y + text_h + 2],
            fill=(0, 0, 0),
        )
        draw.text((origin_x + 2, origin_y + 1), label, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
