"""
run_v8.py — Chạy workflow trích xuất + viz + eval với model PaddleOCR v8
(rec = models/paddle_v8/inference), lưu vào debug_output/outputs_v9/paddle_v8/.

Tái dùng extract_form/draw_viz/bucket_of của run_and_eval; chỉ override rec model = v8.
Chạy: .venv/bin/python src/evaluation/run_v8.py
"""
import json
import os
import sys

import cv2
from paddleocr import PaddleOCR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ocr.engine as engine                                            # noqa: E402

# ---- Override singleton PaddleOCR sang model v8 (trước khi extract_form chạy) ----
V8_REC_DIR = os.path.join(ROOT, "models", "paddle_v8", "inference")
print(f"Khởi tạo PaddleOCR v8: {V8_REC_DIR}")
engine._ocr_instance = PaddleOCR(
    lang="vi", device="cpu",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_recognition_model_dir=V8_REC_DIR,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    use_doc_unwarping=False,
)

from alignment import align_form                                       # noqa: E402
from config_detection import load_config                              # noqa: E402
from ocr.normalizer import apply_normalizers                          # noqa: E402
from evaluation.metrics import cer, wer, exact_match, aggregate_metrics  # noqa: E402
from evaluation.run_and_eval import FORMS_DIR, CONFIG, extract_form, draw_viz, bucket_of  # noqa: E402

OUTDIR = os.path.join(ROOT, "debug_output", "outputs_v9", "paddle_v8")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    config = load_config(CONFIG)
    gts = sorted(f for f in os.listdir(FORMS_DIR) if f.endswith(".json"))
    records = []
    print("########## ENGINE: paddle_v8 ##########")
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

        results = extract_form("paddle_v9", warped, config)   # nhánh paddle; singleton = v8
        json.dump(results, open(os.path.join(OUTDIR, f"{stem}_fields.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        draw_viz(warped, results, gt, os.path.join(OUTDIR, f"{stem}_viz.jpg"))

        for fname, r in results.items():
            if fname not in gt or "text" not in gt[fname]:
                continue
            g = apply_normalizers(gt[fname]["text"], config["fields"][fname].get("normalize"))
            records.append({
                "form": stem, "bucket": bucket_of(stem), "field": fname,
                "gt": g, "pred": r["text"],
                "cer": cer(r["text"], g), "wer": wer(r["text"], g),
                "em": exact_match(r["text"], g),
            })

    overall = aggregate_metrics(records)
    by_bucket = {b: aggregate_metrics([r for r in records if r["bucket"] == b])
                 for b in sorted({r["bucket"] for r in records})}
    by_field = {f: aggregate_metrics([r for r in records if r["field"] == f])
                for f in {r["field"] for r in records}}
    json.dump({"engine": "paddle_v8", "overall": overall, "by_bucket": by_bucket,
               "by_field": by_field, "samples": records},
              open(os.path.join(OUTDIR, "eval_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    o = overall
    print("--- EVAL [paddle_v8] ---")
    print(f"  OVERALL n={o['n']}  CER={o['cer']['mean']*100:.2f}%  WER={o['wer']['mean']*100:.2f}%  EM={o['em']*100:.1f}%")
    for b, m in by_bucket.items():
        if m:
            print(f"    {b:6s} n={m['n']:3d}  CER={m['cer']['mean']*100:.2f}%  WER={m['wer']['mean']*100:.2f}%  EM={m['em']*100:.1f}%")
    print(f"  -> {OUTDIR}")


if __name__ == "__main__":
    main()
