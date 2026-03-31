"""
ocr_engine.py — Nhận diện chữ trong ảnh dùng PaddleOCR
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from typing import List, Dict

# Biến toàn cục lưu instance OCR (tránh tạo lại nhiều lần)
_ocr_instance = None


def get_ocr_instance() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        print("Initial PaddleOCR with fine-tuned model...")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # rec_model_dir = os.path.join(project_root, 'output_mobile_rec_lite', 'inference')
        rec_model_dir = os.path.join(project_root, 'output_rec', 'inference')

        if not os.path.exists(rec_model_dir):
            raise FileNotFoundError(
                f"Inference model not found: {rec_model_dir}\n"
                f"Please run export_model.py first!"
            )

        _ocr_instance = PaddleOCR(
            lang='vi',
            device='cpu',
            # text_recognition_model_name='PP-OCRv5_mobile_rec',
            text_recognition_model_name='PP-OCRv5_server_rec',
            text_recognition_model_dir=rec_model_dir,
        )
        print("Initialized PaddleOCR with fine-tuned model successfully!\n")
    return _ocr_instance


def run_ocr(ocr: PaddleOCR, img: np.ndarray) -> List[Dict]:
    print("Running OCR on the image...")
    raw_results = ocr.ocr(img)

    if not raw_results or raw_results[0] is None:
        print("Not found any text in the image!")
        return []
    
    print(f"Raw OCR results: {raw_results[0]}")

    # Paddleocr 3.x: lấy dữ liệu từ OCRResult object
    result     = raw_results[0]
    texts      = result['rec_texts']   # ['Họ và tên:', 'Nguyễn Văn A', ...]
    scores     = result['rec_scores']  # [0.97, 0.98, ...]
    polys      = result['rec_polys']   # [array([[x,y],[x,y],[x,y],[x,y]]), ...]

    parsed = []
    for text, confidence, bbox in zip(texts, scores, polys):
        if not text.strip():
            continue

        # bbox là numpy array shape (4, 2) → [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        bbox_list = bbox.tolist()
        center_y  = sum(pt[1] for pt in bbox_list) / 4
        x_left    = min(pt[0] for pt in bbox_list)

        parsed.append({
            'text':       text.strip(),
            'confidence': round(float(confidence), 4),
            'bbox':       bbox_list,
            'center_y':   center_y,
            'x_left':     x_left,
        })

    parsed.sort(key=lambda x: (round(x['center_y'] / 20) * 20, x['x_left']))

    print(f"Find out {len(parsed)} text lines:\n")
    for i, item in enumerate(parsed):
        # conf_rate = "High" if item['confidence'] > 0.85 else if "⚠️ "
        if item['confidence'] >= 0.85:
            conf_rate = "High confidence"
        elif item['confidence'] >= 0.50 and item['confidence'] < 0.85:
            conf_rate = "Medium confidence"
        else:
            conf_rate = "Low confidence"
        print(f"  {conf_rate} [{i+1:02d}] ({item['confidence']:.1%}) {item['text']}")

    return parsed

def draw_bounding_boxes(img: np.ndarray, ocr_results: list) -> np.ndarray:
    # Chuyển BGR (OpenCV) → RGB (Pillow)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    # Load font hỗ trợ tiếng Việt
    # Dùng font hệ thống macOS có sẵn
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size=13)
            break
    if font is None:
        font = ImageFont.load_default()

    for item in ocr_results:
        bbox       = item['bbox']
        text       = item['text']
        confidence = item['confidence']

        # Màu theo confidence
        if confidence >= 0.85:
            color = (0, 200, 0)      # Xanh lá
        elif confidence >= 0.70:
            color = (255, 180, 0)    # Vàng
        else:
            color = (220, 0, 0)      # Đỏ

        # Vẽ bounding box (polygon)
        pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
        draw.polygon(pts, outline=color + (255,))
        for i in range(len(pts)):
            draw.line([pts[i], pts[(i+1) % len(pts)]], fill=color, width=2)

        # Vẽ label phía trên box
        label    = f"{text} ({confidence:.0%})"
        origin_x = int(min(pt[0] for pt in bbox))
        origin_y = int(min(pt[1] for pt in bbox)) - 16

        # Tránh label bị cắt ở mép trên
        origin_y = max(origin_y, 2)

        # Nền nhỏ sau label cho dễ đọc
        bbox_text = font.getbbox(label)
        text_w    = bbox_text[2] - bbox_text[0]
        text_h    = bbox_text[3] - bbox_text[1]
        draw.rectangle(
            [origin_x, origin_y, origin_x + text_w + 4, origin_y + text_h + 2],
            fill=(0, 0, 0)
        )
        draw.text((origin_x + 2, origin_y + 1), label, font=font, fill=color)

    # Chuyển lại BGR cho OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def filter_by_confidence(ocr_results: List[Dict], min_confidence: float = 0.75) -> List[Dict]:
    """
    Lọc bỏ các dòng OCR có độ tin cậy thấp

    Args:
        ocr_results: Kết quả từ run_ocr()
        min_confidence: Ngưỡng tối thiểu (mặc định 75%)

    Returns:
        Danh sách đã lọc
    """
    filtered = [r for r in ocr_results if r['confidence'] >= min_confidence]
    removed = len(ocr_results) - len(filtered)
    if removed > 0:
        print(f"Filtered out {removed} lines with confidence < {min_confidence:.0%}")
    return filtered

def get_text_lines(ocr_results: List[Dict]) -> List[str]:
    """Lấy danh sách dòng text thuần (không kèm metadata)"""
    return [r['text'] for r in ocr_results]

def engine_pipeline(img: np.ndarray) -> np.ndarray:
    print("\n" + "="*50)
    print("OCR ENGINE PIPELINE")
    print("="*50)

    print("\n[1/3] Initializing OCR model...")
    ocr = get_ocr_instance()

    print("\n[2/3] Running OCR...")
    ocr_results = run_ocr(ocr, img)

    if not ocr_results:
        print("No text found!")
        return img  # Trả về ảnh gốc nếu không có kết quả

    print("\n[3/3] Drawing bounding boxes...")
    filtered      = filter_by_confidence(ocr_results, min_confidence=0.70)
    annotated_img = draw_bounding_boxes(img, filtered)

    print(f"Drew {len(filtered)} bounding boxes on image.")

    # Lưu ảnh có bounding box
    cv2.imwrite('outputs/ocr_result.jpg', annotated_img)
    print("Saved: outputs/ocr_result.jpg")

    print("\n" + "="*50)
    print("DONE OCR ENGINE!")
    print("="*50)
    return annotated_img