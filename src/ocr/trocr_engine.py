"""
trocr_engine.py — Recognizer TrOCR (fine-tune tiếng Việt) để SO SÁNH với PaddleOCR.

TrOCR đọc 1 dòng/ảnh (không có detection). Dùng cho eval so sánh recognizer trên cùng crop field.
Lazy singleton để chỉ load model 1 lần. CPU + greedy (num_beams=1) cho nhanh, tránh treo.
"""
import os
import unicodedata
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "trocr_vi")
MAX_LEN = 64

_model = None
_processor = None
_device = None


def _load(model_dir: str = DEFAULT_MODEL_DIR):
    """Lazy-load model + processor (1 lần)."""
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print("Loading TrOCR fine-tuned model...")
    _processor = TrOCRProcessor.from_pretrained(model_dir)
    _model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device).eval()
    print(f"TrOCR loaded ({_device}).")
    return _model, _processor, _device


def recognize(crop_bgr: np.ndarray, max_len: int = MAX_LEN) -> Tuple[str, float]:
    """
    Nhận dạng 1 ảnh crop (BGR, như cv2.imread) → (text NFC, score).

    score: trung bình xác suất token (sequence-level), 0..1; càng cao càng tự tin.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    import torch

    model, processor, device = _load()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        out = model.generate(
            pixel_values,
            max_length=max_len,
            num_beams=1,                      # greedy: nhanh trên CPU, tránh treo
            output_scores=True,
            return_dict_in_generate=True,
        )
    ids = out.sequences
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    text = unicodedata.normalize("NFC", text)

    # score = exp(mean log-prob) của chuỗi sinh (nếu có), else 0.0
    score = 0.0
    try:
        if getattr(out, "sequences_scores", None) is not None:
            score = float(torch.exp(out.sequences_scores[0]))
        elif out.scores:
            lp = torch.stack(out.scores, dim=1).log_softmax(-1)
            gen = ids[:, 1:]
            n = min(gen.shape[1], lp.shape[1])
            tok_lp = lp[0, :n].gather(-1, gen[0, :n].unsqueeze(-1)).squeeze(-1)
            score = float(torch.exp(tok_lp.mean())) if n else 0.0
    except Exception:
        score = 0.0
    return text, round(score, 4)
