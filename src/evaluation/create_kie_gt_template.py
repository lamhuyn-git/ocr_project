"""
Tạo KIE GT template cho từng ảnh trong image_test/.

Cách hoạt động:
  1. Chạy pipeline (preprocess → OCR → KIE) trên mỗi ảnh
  2. Xuất kết quả KIE thành file template:
       image_test/ground_truth/{stem}_kie_gt.json
  3. Annotator chỉ cần MỞ file, sửa lại field sai → save thành GT thật.

Bạn nên review TỪNG field, tuyệt đối không tin output pipeline 100%.

Usage:
    # Một folder
    python -m src.evaluation.create_kie_gt_template \
        --input-dir image_test/scan \
        --quality scan \
        --out-dir image_test/ground_truth

    # Một ảnh
    python -m src.evaluation.create_kie_gt_template \
        --input-file image_test/scan/scan_001.jpg \
        --quality scan \
        --out-dir image_test/ground_truth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Cho phép import từ src/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import preprocess_pipeline                # noqa: E402
from ocr.engine import run_ocr_pipeline                   # noqa: E402
from kie.kie import kie                                   # noqa: E402

# 14 fields theo plan + thứ tự đúng của form CT01
KIE_FIELDS = [
    "ho_chu_dem_va_ten",
    "ngay_thang_nam_sinh",
    "gioi_tinh",
    "so_dinh_dan_ca_nhan",
    "so_dien_thoai_lien_he",
    "email",
    "ho_chu_dem_va_ten_chu_ho",
    "moi_quan_he_voi_chu_ho",
    "so_dinh_dan_ca_nhan_cua_chu_ho",
    "noi_dung_de_nghi",
]
META_FIELDS = ["title", "validate_title", "main_title", "kinh_gui"]


def run_pipeline(img_path: Path) -> Dict[str, str]:
    img = preprocess_pipeline(str(img_path))
    img, ocr_blocks = run_ocr_pipeline(img)
    raw = kie(ocr_blocks, img=img)
    # raw[field] = {'text': ..., ...}; ta chỉ giữ text
    flat: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat[k] = v.get("text", "")
        else:
            flat[k] = str(v) if v is not None else ""
    return flat


def make_template(img_rel: str, quality: str, kie_flat: Dict[str, str]) -> Dict:
    fields = {f: kie_flat.get(f, "") for f in KIE_FIELDS}
    meta = {f: kie_flat.get(f, "") for f in META_FIELDS}
    return {
        "image": img_rel,
        "quality": quality,
        "annotator": "",            # ← điền tên người làm
        "annotated_at": "",         # ← yyyy-mm-dd
        "reviewed": False,           # ← True khi đã sửa xong
        "fields": fields,            # ← 14 fields nội dung
        "meta": meta,                # ← 4 trường tham khảo (không tính vào metric)
    }


def iter_images(input_dir: Path) -> List[Path]:
    EXTS = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in EXTS])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", type=Path, help="Folder chứa ảnh")
    g.add_argument("--input-file", type=Path, help="Một ảnh đơn lẻ")
    ap.add_argument("--quality", required=True,
                    choices=["scan", "phone_good", "phone_low", "phone_bad"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true",
                    help="Ghi đè template cũ (mặc định bỏ qua nếu file đã có).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        images = [args.input_file]
        rel_root = args.input_file.parent.parent  # image_test/
    else:
        images = iter_images(args.input_dir)
        rel_root = args.input_dir.parent

    print(f"[INFO] {len(images)} ảnh sẽ chạy pipeline.")
    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        out = args.out_dir / f"{stem}_kie_gt.json"
        if out.exists() and not args.overwrite:
            print(f"  [SKIP] {out} (đã tồn tại, dùng --overwrite để ghi đè)")
            continue

        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            flat = run_pipeline(img_path)
        except Exception as e:
            print(f"  [ERR] pipeline fail: {e}")
            flat = {}

        rel = str(img_path.relative_to(rel_root)) if rel_root in img_path.parents \
            else img_path.name
        tmpl = make_template(rel, args.quality, flat)
        with out.open("w", encoding="utf-8") as f:
            json.dump(tmpl, f, ensure_ascii=False, indent=2)
        print(f"  ✓ wrote {out}")

    print("[DONE] Mở các file *_kie_gt.json trong out-dir, sửa từng field rồi đặt 'reviewed': true.")


if __name__ == "__main__":
    main()
