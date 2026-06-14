"""
compare_engines.py — So sánh field-level PaddleOCR (v9) vs TrOCR trên CÙNG field crop.

Mỗi form trong test_image/scan/ có GT json (<stem>.json, format {field:{type,text,bbox}}):
  align_form → mỗi field 1-dòng: crop ROI → đọc bằng 2 engine → normalize giống nhau → CER/WER vs GT.

Chỉ so field 1 dòng (text_line/number). Field đa dòng (text_block) + table BỎ QUA (TrOCR đọc 1 dòng).

Chạy:
  .venv/bin/python -m src.evaluation.compare_engines            # từ gốc project
  hoặc: .venv/bin/python src/evaluation/compare_engines.py --limit 3
"""
import argparse
import json
import os
import statistics
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alignment import align_form                                   # noqa: E402
from config_detection import load_config, field_roi_pixels         # noqa: E402
from ocr.crop_ocr import crop_roi, ocr_crop, join_blocks, _group_lines  # noqa: E402
from ocr.normalizer import apply_normalizers                       # noqa: E402
from ocr.trocr_engine import recognize as trocr_recognize          # noqa: E402
from evaluation.metrics import cer, wer, exact_match               # noqa: E402

DEFAULT_FORMS = os.path.join(ROOT, "test_image", "scan")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "templates", "ct01_tt53.0.yaml")
DEFAULT_OUT = os.path.join(ROOT, "debug_output", "outputs", "compare_engines.json")

# So field text (1 dòng + đa dòng). Table (members) cần so kiểu khác → loại khỏi v1.
ELIGIBLE_TYPES = {"text_line", "number", "text_block"}


def _tight_crop(crop, blocks_local, pad=4):
    """Cắt sát vùng có chữ (union bbox các line PaddleOCR detect, toạ độ crop-local).
    Cho TrOCR ăn crop dòng sát — đúng phân bố lúc train (tránh đọc nền/đuôi rác)."""
    if not blocks_local:
        return crop
    xs = [p[0] for b in blocks_local for p in b["bbox"]]
    ys = [p[1] for b in blocks_local for p in b["bbox"]]
    h, w = crop.shape[:2]
    x1 = max(0, int(min(xs)) - pad); y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad); y2 = min(h, int(max(ys)) + pad)
    if x2 <= x1 or y2 <= y1:
        return crop
    return crop[y1:y2, x1:x2]


def read_both(crop, normalize_ops):
    """Trả (paddle_text, trocr_text). Paddle: det+rec trên crop.
    TrOCR: đọc TỪNG DÒNG (theo line Paddle detect) trên crop sát → ghép '\\n' (giống join_blocks)."""
    blocks = ocr_crop(crop, box_offset=(0, 0))           # crop-local bbox để cắt sát
    p_text, _ = join_blocks(blocks)

    if not blocks:
        t_text, _ = trocr_recognize(crop)                # không detect được → đọc cả crop
    else:
        line_texts = []
        for line in _group_lines(blocks):               # mỗi dòng → crop sát → TrOCR
            tight = _tight_crop(crop, line)
            txt, _ = trocr_recognize(tight)
            if txt.strip():
                line_texts.append(txt)
        t_text = "\n".join(line_texts)

    return apply_normalizers(p_text, normalize_ops), apply_normalizers(t_text, normalize_ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms-dir", default=DEFAULT_FORMS)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số form (CPU chậm)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    config = load_config(args.config)
    fields_cfg = config["fields"]

    gts = sorted(f for f in os.listdir(args.forms_dir) if f.endswith(".json"))
    if args.limit:
        gts = gts[: args.limit]
    if not gts:
        print("Không thấy GT json trong", args.forms_dir); return

    samples = []   # per (form, field): gt, paddle, trocr + metrics
    for gi, gtname in enumerate(gts, 1):
        stem = gtname[:-5]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(args.forms_dir, stem + ext)
            if os.path.exists(p):
                img_path = p; break
        if img_path is None:
            print(f"[SKIP] {stem}: không thấy ảnh"); continue

        gt = json.load(open(os.path.join(args.forms_dir, gtname), encoding="utf-8"))
        img = cv2.imread(img_path)
        warped, meta = align_form(img)
        h, w = warped.shape[:2]
        print(f"[{gi}/{len(gts)}] {stem} (align={meta['method']} reproj={meta.get('reproj_error')})")

        for fname, fcfg in fields_cfg.items():
            if fcfg.get("type") not in ELIGIBLE_TYPES:
                continue
            if fname not in gt or "text" not in gt[fname]:
                continue
            ops = fcfg.get("normalize")
            gt_text = apply_normalizers(gt[fname]["text"], ops)   # GT cũng chuẩn hoá cho đồng nhất
            box = field_roi_pixels(config, fname, w, h)
            crop = crop_roi(warped, box)

            p_text, t_text = read_both(crop, ops)

            samples.append({
                "form": stem, "field": fname, "gt": gt_text,
                "paddle": p_text, "trocr": t_text,
                "paddle_cer": cer(p_text, gt_text), "trocr_cer": cer(t_text, gt_text),
                "paddle_wer": wer(p_text, gt_text), "trocr_wer": wer(t_text, gt_text),
                "paddle_em": exact_match(p_text, gt_text), "trocr_em": exact_match(t_text, gt_text),
            })

    if not samples:
        print("Không có field nào để so sánh."); return

    # ---- Tổng hợp ----
    def mean(xs): return round(statistics.mean(xs), 4) if xs else 0.0

    def block(rows, label):
        print(f"\n{label}  (n={len(rows)})")
        print(f"  {'':22s}  {'CER':>8} {'WER':>8} {'EM%':>6}")
        for eng in ("paddle", "trocr"):
            c = mean([r[f"{eng}_cer"] for r in rows])
            wv = mean([r[f"{eng}_wer"] for r in rows])
            em = round(100 * mean([1.0 if r[f"{eng}_em"] else 0.0 for r in rows]), 1)
            print(f"  {eng:22s}  {c:>8.4f} {wv:>8.4f} {em:>6.1f}")

    block(samples, "=== TỔNG (tất cả field 1 dòng) ===")

    print("\n=== Theo field ===")
    print(f"  {'field':30s} {'n':>3} {'Paddle CER':>11} {'TrOCR CER':>11}")
    by_field = {}
    for r in samples:
        by_field.setdefault(r["field"], []).append(r)
    for fname, rows in by_field.items():
        pc = mean([r["paddle_cer"] for r in rows])
        tc = mean([r["trocr_cer"] for r in rows])
        win = "TrOCR" if tc < pc else ("Paddle" if pc < tc else "=")
        print(f"  {fname:30s} {len(rows):>3} {pc:>11.4f} {tc:>11.4f}   {win}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(samples, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nPer-sample → {args.out}")


if __name__ == "__main__":
    main()
