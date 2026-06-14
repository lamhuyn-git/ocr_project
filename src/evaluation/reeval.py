"""
reeval.py — Tính LẠI metric (và vẽ lại viz) từ kết quả trích xuất ĐÃ LƯU, dùng GT mới.
KHÔNG chạy lại OCR (output model không đổi khi GT đổi) → nhanh.

Đọc debug_output/outputs_v9/<engine>/<form>_fields.json + GT mới trong test_image/scan/
→ ghi lại <engine>/eval_results.json + <engine>/<form>_viz.jpg (recolor theo GT mới).

Chạy: .venv/bin/python src/evaluation/reeval.py
"""
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alignment import align_form                                       # noqa: E402
from config_detection import load_config                              # noqa: E402
from ocr.normalizer import apply_normalizers                          # noqa: E402
from evaluation.metrics import cer, wer, exact_match, aggregate_metrics  # noqa: E402
from evaluation.run_and_eval import (                                  # noqa: E402
    FORMS_DIR, CONFIG, OUT_ROOT, ENGINES, draw_viz, bucket_of,
)


def main():
    config = load_config(CONFIG)
    gts = sorted(f for f in os.listdir(FORMS_DIR) if f.endswith(".json"))

    for engine in ENGINES:
        outdir = os.path.join(OUT_ROOT, engine)
        records = []
        print(f"\n########## RE-EVAL: {engine} ##########")
        for gtname in gts:
            stem = gtname[:-5]
            fpath = os.path.join(outdir, f"{stem}_fields.json")
            if not os.path.exists(fpath):
                print(f"  [SKIP] {stem}: chưa có fields.json"); continue
            results = json.load(open(fpath, encoding="utf-8"))
            gt = json.load(open(os.path.join(FORMS_DIR, gtname), encoding="utf-8"))

            # vẽ lại viz theo GT mới (re-align, KHÔNG OCR)
            img_path = next((os.path.join(FORMS_DIR, stem + e)
                             for e in (".jpg", ".jpeg", ".png")
                             if os.path.exists(os.path.join(FORMS_DIR, stem + e))), None)
            if img_path:
                warped, _ = align_form(cv2.imread(img_path))
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

        overall = aggregate_metrics(records)
        by_bucket = {b: aggregate_metrics([r for r in records if r["bucket"] == b])
                     for b in sorted({r["bucket"] for r in records})}
        by_field = {f: aggregate_metrics([r for r in records if r["field"] == f])
                    for f in {r["field"] for r in records}}
        json.dump({"engine": engine, "overall": overall, "by_bucket": by_bucket,
                   "by_field": by_field, "samples": records},
                  open(os.path.join(outdir, "eval_results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        o = overall
        if o:
            print(f"  OVERALL n={o['n']}  CER={o['cer']['mean']*100:.2f}%±{o['cer']['std']*100:.2f}  "
                  f"WER={o['wer']['mean']*100:.2f}%  EM={o['em']*100:.1f}%")
            for b, m in by_bucket.items():
                if m:
                    print(f"    {b:6s} n={m['n']:3d}  CER={m['cer']['mean']*100:.2f}%  "
                          f"WER={m['wer']['mean']*100:.2f}%  EM={m['em']*100:.1f}%")
        print(f"  -> {os.path.join(outdir, 'eval_results.json')}")


if __name__ == "__main__":
    main()
