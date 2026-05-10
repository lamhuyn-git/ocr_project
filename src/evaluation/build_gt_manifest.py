"""
Quét image_test/ground_truth/ → tạo gt_manifest.json tổng hợp.

Mỗi entry trong manifest = 1 ảnh, gồm:
  - image
  - quality (scan / phone_good / phone_low / phone_bad)
  - has_ocr_gt   (True/False)
  - has_kie_gt   (True/False)
  - num_regions  (số bbox OCR)
  - kie_reviewed (True nếu file kie_gt có "reviewed": true)
  - annotator, annotated_at

Usage:
    python -m src.evaluation.build_gt_manifest \
        --image-root image_test \
        --gt-dir image_test/ground_truth \
        --out image_test/gt_manifest.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

QUALITIES = ["scan", "phone_good", "phone_low", "phone_bad"]


def discover_images(image_root: Path) -> List[Dict]:
    EXTS = {".jpg", ".jpeg", ".png"}
    out: List[Dict] = []
    for q in QUALITIES:
        sub = image_root / q
        if not sub.is_dir():
            continue
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in EXTS:
                out.append({"image": f"{q}/{p.name}", "quality": q, "stem": p.stem})
    return out


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] {path}: bad json ({e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    images = discover_images(args.image_root)
    print(f"[INFO] Phát hiện {len(images)} ảnh trong {args.image_root}")

    entries = []
    for it in images:
        ocr_gt = load_json(args.gt_dir / f"{it['stem']}_gt.json")
        kie_gt = load_json(args.gt_dir / f"{it['stem']}_kie_gt.json")
        entry = {
            "image": it["image"],
            "quality": it["quality"],
            "has_ocr_gt": ocr_gt is not None,
            "has_kie_gt": kie_gt is not None,
            "num_regions": len(ocr_gt.get("regions", [])) if ocr_gt else 0,
            "kie_reviewed": bool(kie_gt and kie_gt.get("reviewed", False)),
            "annotator": (kie_gt or {}).get("annotator", ""),
            "annotated_at": (kie_gt or {}).get("annotated_at", ""),
        }
        entries.append(entry)

    # tổng kết
    total = len(entries)
    by_q: Dict[str, Dict[str, int]] = {q: {"total": 0, "ocr": 0, "kie": 0,
                                           "kie_reviewed": 0} for q in QUALITIES}
    for e in entries:
        b = by_q[e["quality"]]
        b["total"] += 1
        b["ocr"] += int(e["has_ocr_gt"])
        b["kie"] += int(e["has_kie_gt"])
        b["kie_reviewed"] += int(e["kie_reviewed"])

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "image_root": str(args.image_root),
        "gt_dir": str(args.gt_dir),
        "summary": {
            "total_images": total,
            "with_ocr_gt": sum(int(e["has_ocr_gt"]) for e in entries),
            "with_kie_gt": sum(int(e["has_kie_gt"]) for e in entries),
            "kie_reviewed": sum(int(e["kie_reviewed"]) for e in entries),
            "by_quality": by_q,
        },
        "items": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {args.out}")
    s = manifest["summary"]
    print(f"  total={s['total_images']}  ocr_gt={s['with_ocr_gt']}  "
          f"kie_gt={s['with_kie_gt']}  kie_reviewed={s['kie_reviewed']}")
    for q, b in s["by_quality"].items():
        print(f"  {q:11s} total={b['total']:2d}  ocr={b['ocr']:2d}  "
              f"kie={b['kie']:2d}  reviewed={b['kie_reviewed']:2d}")


if __name__ == "__main__":
    main()
