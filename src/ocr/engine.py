"""
engine.py — Lớp giao tiếp với PaddleOCR (tầng OCR thấp nhất).

  - get_ocr_instance() : khởi tạo model 1 lần (singleton), dùng lại cho mọi lần gọi.
  - run_ocr(ocr, img)  : chạy OCR, trả list block {text, confidence, bbox, center_y, x_left} đã sort.

Detection: PP-OCRv5_mobile_det (official). Recognition: model fine-tune local (models/inference).
Orchestration (align → config → crop → OCR) nằm ở main.py, KHÔNG ở đây.
"""
import os
from typing import Dict, List

import numpy as np
from paddleocr import PaddleOCR

_ocr_instance = None
_ocr_model_version = None   # version hiện tại của singleton


def reset_instance():
    """Xoá singleton để lần gọi get_ocr_instance() tiếp theo load model mới."""
    global _ocr_instance, _ocr_model_version
    _ocr_instance = None
    _ocr_model_version = None


def get_ocr_instance(model_version=None) -> PaddleOCR:
    """
    Trả PaddleOCR singleton.
    model_version: tên thư mục trong models/ (vd 'paddle_v9', 'paddle_v12').
    Nếu None → lấy từ env OCR_MODEL_VERSION (mặc định 'paddle_v9') → cho phép host
    (vd backend) chọn model qua config mà không cần sửa code pipeline.
    Nếu model_version thay đổi so với lần trước, singleton được tạo lại.
    """
    global _ocr_instance, _ocr_model_version

    if model_version is None:
        model_version = os.getenv("OCR_MODEL_VERSION", "paddle_v9")

    if _ocr_instance is not None and _ocr_model_version == model_version:
        return _ocr_instance

    if _ocr_instance is not None:
        # model version thay đổi → reset
        _ocr_instance = None

    print(f"Initial PaddleOCR with fine-tuned model ({model_version})...")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec_model_dir = os.path.join(project_root, 'models', model_version, 'inference')

    if not os.path.exists(rec_model_dir):
        raise FileNotFoundError(
            f"Inference model not found: {rec_model_dir}\n"
            f"Expected models/{model_version}/inference/ to exist."
        )

    _ocr_model_version = model_version
    _ocr_instance = PaddleOCR(
        lang='vi',
        device='cpu',
        text_detection_model_name='PP-OCRv5_mobile_det',     # detection: bản mobile official
        text_recognition_model_name='PP-OCRv5_mobile_rec',   # kiến trúc rec
        text_recognition_model_dir=rec_model_dir,            # weights rec fine-tune local
        # Tắt các bước PaddleOCR tự xử lý vì pipeline của ta đã lo (align/warp)
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        use_doc_unwarping=False,
    )
    print("Initialized PaddleOCR with fine-tuned model successfully!\n")
    return _ocr_instance


def run_ocr(ocr: PaddleOCR, img: np.ndarray) -> List[Dict]:
    raw_results = ocr.ocr(img)

    if not raw_results or raw_results[0] is None:
        print("Not found any text in the image!")
        return []

    result = raw_results[0]
    texts = result['rec_texts']      # list chuỗi text
    scores = result['rec_scores']    # list confidence tương ứng
    polys = result['rec_polys']      # list bbox (4 điểm) tương ứng

    parsed = []
    for text, confidence, bbox in zip(texts, scores, polys):
        if not text.strip():
            continue

        bbox_list = bbox.tolist()
        center_y = sum(pt[1] for pt in bbox_list) / 4
        x_left = min(pt[0] for pt in bbox_list)

        parsed.append({
            'text': text.strip(),
            'confidence': round(float(confidence), 4),
            'bbox': bbox_list,
            'center_y': center_y,
            'x_left': x_left,
        })

    parsed.sort(key=lambda x: (x['center_y'], x['x_left']))

    print(f"Find out {len(parsed)} text lines:")
    return parsed
