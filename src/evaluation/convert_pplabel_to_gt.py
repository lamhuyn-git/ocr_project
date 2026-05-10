"""
Convert PPOCRLabel `Label.txt` → individual GT JSON files.

PPOCRLabel xuất file `Label.txt` trong đúng folder ảnh, mỗi dòng có dạng:
    <relative_image_path>\t<json_array_of_regions>

Script này:
  - đọc Label.txt
  - tạo file GT JSON cho từng ảnh ở `image_test/ground_truth/{stem}_gt.json`
  - giữ nguyên bbox (4 điểm) + transcription
  - field `quality` được suy ra từ tên folder con (scan/phone_good/...)

Usage:
    python -m src.evaluation.convert_pplabel_to_gt \
        --label-file image_test/scan/Label.txt \
        --quality scan \
        --out-dir image_test/ground_truth
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def parse_label_file(label_file: Path) -> List[Dict[str, Any]]:
    """Parse PPOCRLabel Label.txt → list of {image, regions}."""
    entries: List[Dict[str, Any]] = []
    with label_file.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            try:
                img_path, regions_json = line.split("\t", 1)
            except ValueError:
                print(f"[WARN] Line {line_no}: missing TAB, skip.")
                continue
            try:
                regions = json.loads(regions_json)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: bad JSON ({e}), skip.")
                continue
            entries.append({"image": img_path, "regions": regions})
    return entries


def to_gt_record(entry: Dict[str, Any], quality: str) -> Dict[str, Any]:
    """Chuyển 1 dòng PPOCRLabel sang format GT chuẩn của plan."""
    regions_out = []
    for r in entry["regions"]:
        # bỏ qua region đánh dấu khó/không rõ nếu muốn
        if r.get("difficult"):
            continue
        regions_out.append({
            "bbox": r.get("points", []),
            "text": r.get("transcription", ""),
            "key_cls": r.get("key_cls", "None"),
        })
    return {
        "image": entry["image"],
        "quality": quality,
        "regions": regions_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-file", required=True, type=Path,
                    help="Đường dẫn tới Label.txt do PPOCRLabel xuất.")
    ap.add_argument("--quality", required=True,
                    choices=["scan", "phone_good", "phone_low", "phone_bad"],
                    help="Nhóm chất lượng ảnh.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Thư mục lưu file GT (vd: image_test/ground_truth).")
    args = ap.parse_args()

    if not args.label_file.exists():
        raise SystemExit(f"Không tìm thấy {args.label_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_label_file(args.label_file)
    print(f"[INFO] Đọc {len(entries)} ảnh từ {args.label_file}")

    written = 0
    for e in entries:
        gt = to_gt_record(e, args.quality)
        stem = Path(e["image"]).stem  # scan_001
        out = args.out_dir / f"{stem}_gt.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        written += 1
        print(f"  ✓ {out}  ({len(gt['regions'])} regions)")
    print(f"[DONE] Ghi {written} GT file vào {args.out_dir}")


if __name__ == "__main__":
    main()
