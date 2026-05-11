"""
OCR evaluation metrics: CER, WER, Exact Match.

Dùng rapidfuzz để tính Levenshtein distance hiệu quả.
"""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def cer(pred: str, gt: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(gt).

    Thang điểm: <1% xuất sắc | 1-5% tốt | 5-10% chấp nhận | >10% kém
    """
    pred = pred.strip()
    gt   = gt.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    """Word Error Rate = edit_distance(words) / len(gt_words)."""
    pred_words = pred.strip().split()
    gt_words   = gt.strip().split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return Levenshtein.distance(pred_words, gt_words) / len(gt_words)


def exact_match(pred: str, gt: str) -> bool:
    """Exact Match — pred và gt giống nhau hoàn toàn (sau strip)."""
    return pred.strip() == gt.strip()


def bbox_iou(b1: list, b2: list) -> float:
    """Tính IoU giữa 2 polygon bbox dạng [[x,y], ...] (4 điểm).

    Dùng bounding box của polygon để xấp xỉ nhanh.
    """
    def to_rect(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    x1a, y1a, x2a, y2a = to_rect(b1)
    x1b, y1b, x2b, y2b = to_rect(b2)

    ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    return inter / (area_a + area_b - inter)


def aggregate_metrics(records: list[dict]) -> dict:
    """Tính mean ± std từ danh sách metric records.

    records: [{"cer": 0.05, "wer": 0.08, "em": True}, ...]
    """
    import statistics

    if not records:
        return {}

    cer_vals = [r["cer"] for r in records]
    wer_vals = [r["wer"] for r in records]
    em_vals  = [r["em"]  for r in records]

    def stats(vals):
        mean = sum(vals) / len(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    return {
        "n":   len(records),
        "cer": stats(cer_vals),
        "wer": stats(wer_vals),
        "em":  round(sum(em_vals) / len(em_vals), 4),  # exact match rate
    }
