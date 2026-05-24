from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2

# Thêm project root vào sys.path để import src modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.visualize import draw_bounding_boxes
from src.ocr.engine import run_ocr_pipeline
from src.preprocess.preprocess import preprocess_pipeline
from src.evaluation.metrics import cer, wer, exact_match, bbox_iou, aggregate_metrics

# Paths
TEST_DIR   = PROJECT_ROOT / "test_image"
LABEL_FILE = TEST_DIR / "label_test.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Pattern nhận diện quality bucket từ tên file
BUCKET_PATTERN = re.compile(r"^(scan_hand|phone_good_hand|phone_low_hand|scan|phone_good|phone_low)")


# Covert from label.txt to JSON
def parse_label_txt(label_file: Path) -> list[dict]:
    entries = []
    with label_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                path_part, json_part = line.split("\t", 1)
                regions_raw = json.loads(json_part)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  [WARN] Parse error: {e} — {line[:60]}")
                continue

            # từ regions_raw chỉ lấy text + points, bỏ các trường khác
            regions = [
                {"text": r["transcription"], "points": r["points"]}
                for r in regions_raw
                if r.get("transcription", "").strip()
            ]

            # Suy ra quality bucket từ tên file
            fname  = Path(path_part).name
            m      = BUCKET_PATTERN.match(fname)
            bucket = m.group(1) if m else "unknown"

            entries.append({
                "form_path": path_part,   # vd. "test/phone_good_004.jpg"
                "quality":   bucket,
                "regions":   regions,
            })

    return entries


# Hàm match predicted OCR blocks với GT regions dựa trên IoU, trả về list các cặp matched cùng metrics
def match_regions(pred_blocks: list[dict], gt_regions: list[dict],iou_threshold: float = 0.3) -> list[dict]:
    matched  = []
    used_gt  = set()
    used_pred = set()

    # Tính ma trận IoU
    for pi, block in enumerate(pred_blocks):
        best_iou = 0.0
        best_gi  = -1
        for gi, gt in enumerate(gt_regions):
            if gi in used_gt:
                continue
            iou = bbox_iou(block["bbox"], gt["points"])
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi

        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_text   = gt_regions[best_gi]["text"]
            pred_text = block["text"]
            matched.append({
                "pred_text": pred_text,
                "gt_text":   gt_text,
                "iou":       round(best_iou, 4),
                "cer":       cer(pred_text, gt_text),
                "wer":       wer(pred_text, gt_text),
                "em":        exact_match(pred_text, gt_text),
            })
            used_gt.add(best_gi)
            used_pred.add(pi)

    return matched


def evaluate_form(entry: dict, iou_threshold: float, image_dir: Path | None = None, result_dir: str = "outputs/test_results") -> dict | None:
    fname = Path(entry["form_path"]).name
    if image_dir:
        img_path = image_dir / fname
    else:
        img_path = PROJECT_ROOT / entry["form_path"]
        if not img_path.exists():
            img_path = TEST_DIR / fname
    if not img_path.exists():
        print(f"Not found the img with path: {fname} in {image_dir or PROJECT_ROOT}")
        return None

    # preprocess_pipeline nhận file path (str), không phải numpy array
    img_pre = preprocess_pipeline(str(img_path))

    # Chạy OCR pipeline
    _, ocr_blocks = run_ocr_pipeline(img_pre)

    os.makedirs(result_dir, exist_ok=True)
    save_img = draw_bounding_boxes(img_pre, ocr_blocks)
    cv2.imwrite(f'{result_dir}/{Path(entry["form_path"]).stem}_ocr_result.jpg', save_img)
    
    gt_regions  = entry["regions"] # Lấy GT regions từ entry
    n_gt        = len(gt_regions)
    matched     = match_regions(ocr_blocks, gt_regions, iou_threshold)
    n_matched   = len(matched)

    # Metrics
    detection_rate = n_matched / n_gt if n_gt else 0.0
    rec_metrics    = aggregate_metrics(matched) if matched else {}

    return {
        "form":            Path(entry["form_path"]).stem,
        "quality":         entry["quality"],
        "n_gt_regions":    n_gt,
        "n_pred_regions":  len(ocr_blocks),
        "n_matched":       n_matched,
        "detection_rate":  round(detection_rate, 4),
        "ocr_metrics":     rec_metrics,
        "matched_pairs":   matched,  # raw pairs cho debug
    }


def _read_model_name() -> str:
    """Đọc model_name từ inference.yml, fallback về tên thư mục."""
    try:
        import yaml
        yml = PROJECT_ROOT / "models" / "inference" / "inference.yml"
        if yml.exists():
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
            return data.get("Global", {}).get("model_name", "unknown")
    except Exception:
        pass
    return "unknown"


def print_report(results: list[dict], model_label: str = "") -> None:
    """In báo cáo tổng hợp ra terminal."""
    from collections import defaultdict

    by_group: dict[str, list] = defaultdict(list)
    for r in results:
        by_group[r["quality"]].append(r)

    model_name = _read_model_name()
    display = f"{model_name} ({model_label})" if model_label else model_name

    print("\n" + "=" * 65)
    print(f"{'OCR EVALUATION REPORT':^65}")
    print("=" * 65)
    print(f"  Model  : {display}")
    print(f"  Test   : {LABEL_FILE.relative_to(PROJECT_ROOT)} — {len(results)} forms")
    print("=" * 65)

    header = f"{'Group':<12} {'N':>3} {'DetRate':>8} {'CER mean':>9} {'CER std':>8} {'WER mean':>9} {'EM Rate':>8}"
    print(header)
    print("-" * 65)

    all_records = []
    for group in ["scan", "phone_good", "phone_low", "scan_hand", "phone_good_hand", "phone_low_hand"]:
        forms = by_group.get(group, [])
        if not forms:
            continue

        det_rates   = [f["detection_rate"] for f in forms]
        rec_records = []
        for f in forms:
            rec_records.extend(f.get("matched_pairs", []))
        all_records.extend(rec_records)

        agg = aggregate_metrics(rec_records) if rec_records else {}
        det_mean = sum(det_rates) / len(det_rates)

        cer_m = f"{agg.get('cer', {}).get('mean', 0)*100:.1f}%" if agg else "—"
        cer_s = f"{agg.get('cer', {}).get('std',  0)*100:.1f}%" if agg else "—"
        wer_m = f"{agg.get('wer', {}).get('mean', 0)*100:.1f}%" if agg else "—"
        em    = f"{agg.get('em', 0)*100:.1f}%"                  if agg else "—"

        print(f"  {group:<12} {len(forms):>3} {det_mean*100:>7.1f}% {cer_m:>9} {cer_s:>8} {wer_m:>9} {em:>8}")

    # Overall
    if all_records:
        total_agg = aggregate_metrics(all_records)
        all_det   = [f["detection_rate"] for f in results]
        print("-" * 65)
        print(f"  {'OVERALL':<12} {len(results):>3} "
              f"{sum(all_det)/len(all_det)*100:>7.1f}% "
              f"{total_agg['cer']['mean']*100:>8.1f}% "
              f"{total_agg['cer']['std']*100:>8.1f}% "
              f"{total_agg['wer']['mean']*100:>8.1f}% "
              f"{total_agg['em']*100:>7.1f}%")
    print("=" * 65)


def main():
    ap = argparse.ArgumentParser(description="Evaluate OCR pipeline on test set")
    ap.add_argument("--group",         default=None,
                    help="Chỉ chạy 1 nhóm: scan | phone_good | phone_low")
    ap.add_argument("--iou-threshold", type=float, default=0.3,
                    help="IoU threshold để match predicted vs GT bbox (default: 0.3)")
    ap.add_argument("--output",        default=None,
                    help="Lưu raw results ra JSON file (vd. outputs/eval_results.json)")
    ap.add_argument("--no-pairs",      action="store_true",
                    help="Bỏ matched_pairs khỏi output JSON (tiết kiệm dung lượng)")
    ap.add_argument("--image-dir",     default=None,
                    help="Folder chứa ảnh form (vd. image_test/). Mặc định resolve từ Label.txt path")
    ap.add_argument("--model-label",   default="",
                    help="Nhãn model hiển thị trong report (vd. v7, v8-domain)")
    ap.add_argument("--label-file",    default=None,
                    help="Path tới Label.txt tuỳ chỉnh (mặc định: test_image/label_test.txt)")
    ap.add_argument("--result-dir",    default=None,
                    help="Folder lưu ảnh kết quả OCR (mặc định: outputs/test_results)")
    args = ap.parse_args()

    label_file = Path(args.label_file) if args.label_file else LABEL_FILE
    if not label_file.exists():
        raise SystemExit(f"GT file not found: {label_file}")

    result_dir = args.result_dir or "outputs/test_results"

    # Parse label.txt thành JSON entries
    entries = parse_label_txt(label_file)
    # Nếu có truyền group vào terminal thì chỉ lấy các entries của group đó để evaluate
    if args.group:
        entries = [e for e in entries if e["quality"] == args.group]
        if not entries:
            raise SystemExit(f"Not have any form in group '{args.group}'")

    print(f"Evaluating {len(entries)} forms with IoU threshold={args.iou_threshold} ...")

    image_dir = Path(args.image_dir) if args.image_dir else None

    results = []
    for i, entry in enumerate(entries, 1):
        form_name = Path(entry["form_path"]).stem  #chỉ lấy phần tên file, bỏ đi thư mục và đuôi mở rộng
        print(f"\n[{i:02d}/{len(entries):02d}] {form_name} ({entry['quality']})")
        result = evaluate_form(entry, args.iou_threshold, image_dir, result_dir)
        if result:
            if args.no_pairs:
                result.pop("matched_pairs", None)
            results.append(result)
            print(f"  GT={result['n_gt_regions']} | Pred={result['n_pred_regions']} "
                  f"| Matched={result['n_matched']} | DetRate={result['detection_rate']:.1%}")
            if result["ocr_metrics"]:
                m = result["ocr_metrics"]
                print(f"  CER={m['cer']['mean']:.1%}  WER={m['wer']['mean']:.1%}  "
                      f"EM={m['em']:.1%}  (n={m['n']})")

    if not results:
        print("Không có kết quả nào.")
        return

    print_report(results, model_label=args.model_label)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nRaw results saved: {out_path}")


if __name__ == "__main__":
    main()
