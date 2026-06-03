"""
evaluate_recognition.py — Đo độ chính xác RECOGNITION của model trên ground-truth label_test.txt.

Cách đo (recognition thuần, tách khỏi lỗi align/ROI):
  với mỗi vùng GT (transcription + points trên ảnh gốc):
    perspective-crop vùng đó → chạy recognition → so text dự đoán với GT.

Metrics:
  - Accuracy (exact match): % vùng đọc ĐÚNG HOÀN TOÀN.
  - CER (Character Error Rate): edit-distance / độ dài GT (càng thấp càng tốt).
Tách nhóm: form chữ in (tên file KHÔNG có 'hand') vs form có viết tay ('hand').

Chạy:
  python evaluate_recognition.py
"""
import json
import os
import sys

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

IMG_DIR = "test_image"
GT_FILE = "test_image/label_test.txt"


def perspective_crop(img, points):
    """Cắt vùng tứ giác (4 điểm) thành ảnh chữ nhật bằng perspective transform."""
    pts = np.array(points, dtype=np.float32)
    # kích thước đích = cạnh dài nhất
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts[:4], dst)
    return cv2.warpPerspective(img, M, (w, h))


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def main():
    from paddleocr import TextRecognition
    rec = TextRecognition(model_dir="models/inference", model_name="PP-OCRv5_mobile_rec")

    def recognize(crop):
        out = rec.predict(crop)
        return (out[0]["rec_text"] or "").strip() if out else ""

    only = "hand"   # chỉ đánh giá ảnh có 'hand' trong tên (đặt None để lấy tất cả)

    with open(GT_FILE, encoding="utf-8") as f:
        lines = [ln for ln in f if "\t" in ln]

    results = []  # (exact, cer)
    for ln in lines:
        fname, raw = ln.split("\t", 1)
        if only and only not in fname:
            continue
        img = cv2.imread(os.path.join(IMG_DIR, fname.strip().split("/")[-1]))
        if img is None:
            continue
        for item in json.loads(raw):
            if item.get("difficult"):
                continue
            gt = (item.get("transcription") or "").strip()
            if not gt:
                continue
            crop = perspective_crop(img, item["points"])
            if crop is None:
                continue
            pred = recognize(crop)
            results.append((int(pred == gt), cer(pred, gt)))

    n = len(results)
    acc = sum(e for e, _ in results) / n if n else 0.0
    mcer = sum(c for _, c in results) / n if n else 0.0
    print("\n=== Đánh giá text (recognition) ===")
    print(f"Lọc ảnh    : {only or 'tất cả'}")
    print(f"Số vùng    : {n}")
    print(f"Accuracy   : {acc:.1%}   (khớp tuyệt đối)")
    print(f"CER        : {mcer:.3f}  (tỉ lệ lỗi ký tự)")


if __name__ == "__main__":
    main()
