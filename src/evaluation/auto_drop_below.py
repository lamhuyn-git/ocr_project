"""
Xoá hẳn (drop) các box trong Label.txt nằm dưới ngưỡng Y — vùng
out-of-scope (bảng & signature ở nửa dưới form CT01).

Khác với auto_mark_difficult.py: script này XOÁ luôn box khỏi Label.txt
thay vì set difficult=true. Kết quả Label.txt gọn hơn, dễ verify hơn.

Usage:
    python -m src.evaluation.auto_drop_below \
        --label-file image_test/scan/Label.txt \
        --image-dir image_test/scan \
        --threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Cần Pillow: pip install pillow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-file", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not args.no_backup:
        bak = args.label_file.with_suffix(args.label_file.suffix + ".bak")
        shutil.copy2(args.label_file, bak)
        print(f"[INFO] Backup → {bak}")

    rows = []
    with args.label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            path, js = line.split("\t", 1)
            rows.append([path, json.loads(js)])

    total_dropped = 0
    new_rows = []
    for path, regs in rows:
        img_path = args.image_dir / Path(path).name
        if not img_path.exists():
            img_path = Path(path)
        with Image.open(img_path) as im:
            W, H = im.size
        cutoff = args.threshold * H

        kept = []
        dropped = 0
        for r in regs:
            pts = r.get("points", [])
            if not pts:
                continue
            min_y = min(p[1] for p in pts)
            if min_y > cutoff:
                dropped += 1
                continue
            kept.append(r)
        print(f"  {Path(path).name}: ảnh {W}x{H}, cutoff y={int(cutoff)}, "
              f"giữ {len(kept)}/{len(regs)} (xoá {dropped}).")
        new_rows.append([path, kept])
        total_dropped += dropped

    with args.label_file.open("w", encoding="utf-8") as f:
        for path, regs in new_rows:
            f.write(f"{path}\t{json.dumps(regs, ensure_ascii=False)}\n")

    print(f"[DONE] Xoá tổng {total_dropped} box. "
          f"Mở lại PPOCRLabel để verify, rồi convert sang GT.")


if __name__ == "__main__":
    main()
