"""
run_and_eval.py — Chạy CẢ 2 model (PaddleOCR v9 + TrOCR) trên test_image/scan,
mỗi model lưu vào 1 folder riêng trong debug_output/outputs_v9/:
  <engine>/<form>_fields.json   — kết quả trích xuất (field → text/conf/bbox)
  <engine>/<form>_viz.jpg       — ảnh warped + box field + text dự đoán (xanh=khớp GT, cam=khác)
  <engine>/eval_results.json    — CER/WER/EM (metrics.py), tổng + theo bucket + per-field

So field TEXT (text_line/number/text_block). Table (members) ngoài phạm vi recognizer-comparison.
TrOCR đọc từng dòng theo line PaddleOCR detect (công bằng + xử lý đa dòng).

Chạy: .venv/bin/python src/evaluation/run_and_eval.py [--limit N]
"""
import argparse
import json
import os
import sys
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alignment import align_form                                       # noqa: E402
from config_detection import load_config, field_roi_pixels             # noqa: E402
from ocr.crop_ocr import crop_roi, ocr_crop, join_blocks, _group_lines # noqa: E402
from ocr.engine import reset_instance                                  # noqa: E402
from ocr.normalizer import apply_normalizers                           # noqa: E402
from ocr.digit_grid import recognize_digit_grid                        # noqa: E402
from ocr.trocr_engine import recognize as trocr_recognize             # noqa: E402
from evaluation.metrics import cer, wer, exact_match, aggregate_metrics  # noqa: E402

FORMS_DIR = os.path.join(ROOT, "test_image", "scan")
CONFIG = os.path.join(ROOT, "configs", "templates", "ct01_tt53.0.yaml")
OUT_ROOT = os.path.join(ROOT, "debug_output", "outputs_v9")
FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
TEXT_TYPES = {"text_line", "number", "text_block"}
EVAL_TYPES = TEXT_TYPES | {"digit_grid"}
ENGINES = ["paddle_v9", "trocr"]


def _trocr_strip_reader(img):
    return trocr_recognize(img)   # (text, score)


def _tight_crop(crop, blocks, pad=4):
    if not blocks:
        return crop
    xs = [p[0] for b in blocks for p in b["bbox"]]
    ys = [p[1] for b in blocks for p in b["bbox"]]
    h, w = crop.shape[:2]
    x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x2, y2 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)
    return crop[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else crop


def recognize_field(engine, crop):
    """Trả (raw_text, confidence). Cả 2 dùng PaddleOCR detect để tách dòng."""
    model_version = engine if engine.startswith("paddle") else "paddle_v9"
    blocks = ocr_crop(crop, box_offset=(0, 0), model_version=model_version)
    if engine.startswith("paddle"):
        text, conf = join_blocks(blocks)
        return text, conf
    # trocr: đọc từng dòng (line PaddleOCR detect) → ghép '\n'
    if not blocks:
        t, s = trocr_recognize(crop)
        return t, s
    parts, scores = [], []
    for line in _group_lines(blocks):
        t, s = trocr_recognize(_tight_crop(crop, line))
        if t.strip():
            parts.append(t); scores.append(s)
    return "\n".join(parts), (round(sum(scores) / len(scores), 4) if scores else 0.0)


def extract_form(engine, warped, config):
    h, w = warped.shape[:2]
    results = {}
    for fname, fcfg in config["fields"].items():
        ftype = fcfg.get("type")
        if ftype not in EVAL_TYPES:
            continue
        box = field_roi_pixels(config, fname, w, h)
        if ftype == "digit_grid":
            if engine == "trocr":
                raw, conf = recognize_digit_grid(warped, box, fcfg.get("cells", 12), reader=_trocr_strip_reader)
            else:
                # Paddle engine: tạo reader closure với đúng model_version
                mv = engine if engine.startswith("paddle") else "paddle_v9"
                paddle_reader = lambda img, _mv=mv: join_blocks(ocr_crop(img, box_offset=(0, 0), model_version=_mv))
                raw, conf = recognize_digit_grid(warped, box, fcfg.get("cells", 12), reader=paddle_reader)
        else:
            crop = crop_roi(warped, box)
            raw, conf = recognize_field(engine, crop)
        text = apply_normalizers(raw, fcfg.get("normalize"))
        results[fname] = {
            "type": fcfg["type"], "text": text, "text_raw": raw,
            "confidence": round(float(conf), 4), "bbox": [int(v) for v in box],
        }
    return results


def draw_viz(warped, results, gt, out_path):
    img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 22)
    for fname, r in results.items():
        x1, y1, x2, y2 = r["bbox"]
        gt_text = gt.get(fname, {}).get("text", "")
        ok = (r["text"] == apply_normalizers(gt_text, None) or r["text"].strip() == gt_text.strip())
        color = (0, 160, 0) if ok else (230, 120, 0)
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        d.text((x1, max(0, y1 - 24)), r["text"][:40] or "(rỗng)", fill=color, font=font)
    img.save(out_path, quality=92)


def bucket_of(stem):
    return "hand" if stem.startswith("hand") else ("print" if stem.startswith("print") else "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--engines", default=",".join(ENGINES),
                    help="danh sách engine, ngăn cách dấu phẩy (vd: paddle_v9)")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    config = load_config(CONFIG)
    gts = sorted(f for f in os.listdir(FORMS_DIR) if f.endswith(".json"))
    if args.limit:
        gts = gts[: args.limit]

    for engine in engines:
        # Reset OCR singleton khi chuyển sang engine khác (tránh dùng model cũ)
        if engine.startswith("paddle"):
            reset_instance()
        outdir = os.path.join(args.out_root, engine)
        os.makedirs(outdir, exist_ok=True)
        records = []                       # per-field metric records (kèm bucket+field)
        print(f"\n########## ENGINE: {engine} ##########")
        for gi, gtname in enumerate(gts, 1):
            stem = gtname[:-5]
            img_path = next((os.path.join(FORMS_DIR, stem + e)
                             for e in (".jpg", ".jpeg", ".png")
                             if os.path.exists(os.path.join(FORMS_DIR, stem + e))), None)
            if not img_path:
                continue
            gt = json.load(open(os.path.join(FORMS_DIR, gtname), encoding="utf-8"))
            warped, meta = align_form(cv2.imread(img_path))
            print(f"[{gi}/{len(gts)}] {stem} (align={meta['method']} reproj={meta.get('reproj_error')})")

            results = extract_form(engine, warped, config)
            json.dump(results, open(os.path.join(outdir, f"{stem}_fields.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            draw_viz(warped, results, gt, os.path.join(outdir, f"{stem}_viz.jpg"))

            for fname, r in results.items():
                if fname not in gt or "text" not in gt[fname]:
                    continue
                g = apply_normalizers(gt[fname]["text"], config["fields"][fname].get("normalize"))
                g_m = apply_normalizers(g, ["strip_commas", "to_lower"])
                pred_m = apply_normalizers(r["text"], ["strip_commas", "to_lower"])
                records.append({
                    "form": stem, "bucket": bucket_of(stem), "field": fname,
                    "gt": g, "pred": r["text"],
                    "cer": cer(pred_m, g_m), "wer": wer(pred_m, g_m),
                    "em": exact_match(pred_m, g_m),
                })

        # ---- Eval ----
        overall = aggregate_metrics(records)
        by_bucket = {b: aggregate_metrics([r for r in records if r["bucket"] == b])
                     for b in sorted({r["bucket"] for r in records})}
        by_field = {f: aggregate_metrics([r for r in records if r["field"] == f])
                    for f in {r["field"] for r in records}}
        eval_out = {"engine": engine, "overall": overall, "by_bucket": by_bucket,
                    "by_field": by_field, "samples": records}
        json.dump(eval_out, open(os.path.join(outdir, "eval_results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        # ---- Report ----
        print(f"--- EVAL [{engine}] ---")
        o = overall
        print(f"  OVERALL n={o['n']}  CER={o['cer']['mean']*100:.2f}%±{o['cer']['std']*100:.2f}  "
              f"WER={o['wer']['mean']*100:.2f}%  EM={o['em']*100:.1f}%")
        for b, m in by_bucket.items():
            if m:
                print(f"    {b:6s} n={m['n']:3d}  CER={m['cer']['mean']*100:.2f}%  "
                      f"WER={m['wer']['mean']*100:.2f}%  EM={m['em']*100:.1f}%")
        print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
