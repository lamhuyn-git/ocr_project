"""
Render text thành ảnh crop với 3 augmentation variants:
  - clean: nền trắng, sắc nét (giống scan)
  - noisy: nhiễu + blur (giống phone_low)
  - paper: nền vàng + hạt (giống giấy cũ)
"""
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

IMG_HEIGHT  = 48
FONT_SIZES  = [28, 30, 32]
VARIANTS    = ["clean", "noisy", "paper"]


def _make_base_image(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    bbox = font.getbbox(text)
    w = max(bbox[2] - bbox[0] + 24, 120)
    img = Image.new("RGB", (w, IMG_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12, (IMG_HEIGHT - (bbox[3] - bbox[1])) // 2 - bbox[1]), text,
              fill=(20, 20, 20), font=font)
    return img


def _apply_noisy(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    arr += np.random.normal(0, random.uniform(10, 20), arr.shape).astype(np.int16)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    radius = random.uniform(0.5, 1.2)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_paper(img: Image.Image) -> Image.Image:
    # Đổi nền sang màu giấy vàng nhạt
    arr = np.array(img).astype(np.float32)
    tint = np.array([245, 238, 210], dtype=np.float32)
    mask = (arr > 200).all(axis=2, keepdims=True)  # vùng nền trắng
    arr  = np.where(mask, tint, arr)
    arr += np.random.normal(0, 6, arr.shape)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_variants(text: str, fonts: list, img_dir: str, prefix: str) -> list:
    """
    Render text thành 3 ảnh (clean/noisy/paper) với font ngẫu nhiên.
    Trả về list tên file đã lưu.
    """
    font_path = random.choice(fonts)
    font_size = random.choice(FONT_SIZES)
    font      = ImageFont.truetype(font_path, size=font_size)

    base     = _make_base_image(text, font)
    saved    = []

    renderers = {
        "clean": lambda img: img,
        "noisy": _apply_noisy,
        "paper": _apply_paper,
    }

    for variant, fn in renderers.items():
        out_img  = fn(base.copy())
        fname    = f"{prefix}_{variant}.jpg"
        out_path = os.path.join(img_dir, fname)
        out_img.save(out_path, quality=92)
        saved.append(fname)

    return saved
