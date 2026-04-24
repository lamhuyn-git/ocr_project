"""
engine.py — PaddleOCR wrapper + recognition pipeline
"""
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict

from recognition.visualize import draw_bounding_boxes

_ocr_instance = None


def get_ocr_instance() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        print("Initial PaddleOCR with fine-tuned model...")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rec_model_dir = os.path.join(project_root, 'models', 'recognition', 'inference')

        if not os.path.exists(rec_model_dir):
            raise FileNotFoundError(
                f"Inference model not found: {rec_model_dir}\n"
                f"Expected {rec_model_dir} to exist — check plan step C."
            )

        _ocr_instance = PaddleOCR(
            lang='vi',
            device='cpu',
            text_recognition_model_name='PP-OCRv5_server_rec',
            text_recognition_model_dir=rec_model_dir,
        )
        print("Initialized PaddleOCR with fine-tuned model successfully!\n")
    return _ocr_instance


def run_ocr(ocr: PaddleOCR, img: np.ndarray) -> List[Dict]:
    raw_results = ocr.ocr(img)

    if not raw_results or raw_results[0] is None:
        print("Not found any text in the image!")
        return []

    result = raw_results[0]
    texts  = result['rec_texts']
    scores = result['rec_scores']
    polys  = result['rec_polys']

    parsed = []
    for text, confidence, bbox in zip(texts, scores, polys):
        if not text.strip():
            continue

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

    parsed.sort(key=lambda x: (x['center_y'], x['x_left']))

    print(f"Find out {len(parsed)} text lines:")
    for i, item in enumerate(parsed):
        if item['confidence'] >= 0.85:
            conf_rate = "High confidence"
        elif item['confidence'] >= 0.50:
            conf_rate = "Medium confidence"
        else:
            conf_rate = "Low confidence"
        print(f"  [{i+1:02d}] ({item['confidence']:.1%}) {conf_rate} {item['text']}")

    return parsed


def filter_by_confidence(ocr_results: List[Dict], min_confidence: float = 0.75) -> List[Dict]:
    filtered = [r for r in ocr_results if r['confidence'] >= min_confidence]
    removed = len(ocr_results) - len(filtered)
    if removed > 0:
        print(f"Filtered out {removed} lines with confidence < {min_confidence:.0%}")
    return filtered


def get_text_lines(ocr_results: List[Dict]) -> List[str]:
    return [r['text'] for r in ocr_results]


def engine_pipeline(img: np.ndarray, img_path: str = None) -> np.ndarray:
    print("=" * 50)
    print("OCR ENGINE PIPELINE")
    print("=" * 50)

    print("[1/4] Initializing OCR model...")
    ocr = get_ocr_instance()

    print("[2/4] Running OCR...")
    ocr_results = run_ocr(ocr, img)

    if not ocr_results:
        print("No text found!")
        return img

    print("[3/4] Drawing bounding boxes...")
    filtered      = filter_by_confidence(ocr_results, min_confidence=0.5)
    annotated_img = draw_bounding_boxes(img, filtered)

    if img_path:
        name, ext = os.path.splitext(os.path.basename(img_path))
        out_dir   = 'outputs/test_results'
        os.makedirs(out_dir, exist_ok=True)
        out_path  = f'{out_dir}/{name}_result{ext}'
        cv2.imwrite(out_path, annotated_img)
        print(f"[4/4] Saved in {out_path}")

    return annotated_img
