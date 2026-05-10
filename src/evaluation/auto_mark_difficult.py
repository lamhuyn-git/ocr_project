"""
Tự động đánh dấu `difficult: true` cho mọi box trong Label.txt nằm
dưới một ngưỡng Y (vd: phần bảng & signature ở nửa dưới form CT01).

Script làm:
  - Backup Label.txt → Label.txt.bak
  - Với mỗi box, nếu min(y của 4 điểm) > threshold * image_height
    → set difficult = true
  - Ghi lại Label.txt

Sau đó mở lại PPOCRLabel: các box vừa tag sẽ hiện màu xám / có ✓ trong panel.
Có thể tinh chỉnh thủ công thêm nếu cần.

Usage:
    # Mặc định ngưỡng = 0.55 (55% chiều cao ảnh tính từ trên xuống)
    python -m src.evaluation.auto_mark_difficult \
        --label-file image_test/scan/Label.txt \
        --image-dir image_test/scan \
        --threshold 0.55

  Tăng threshold (vd 0.6) nếu thấy script đánh nhầm box mục 10.
  Giảm threshold (vd 0.5) nếu vẫn còn box bảng chưa được tag.
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


def load_label(label_file: Path):
    rows = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            path, js = line.split("\t", 1)
            regions = json.loads(js)
            rows.append([path, regions])
    return rows


def write_label(label_file: Path, rows):
    with label_file.open("w", encoding="utf-8") as f:
        for path, regs in rows:
            f.write(f"{path}\t{json.dumps(regs, ensure_ascii=False)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-file", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="Thư mục chứa ảnh (để đọc kích thước thật).")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="Tỉ lệ Y/H_ảnh; box có min_y vượt ngưỡng → difficult.")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not args.no_backup:
        bak = args.label_file.with_suffix(args.label_file.suffix + ".bak")
        shutil.copy2(args.label_file, bak)
        print(f"[INFO] Backup → {bak}")

    rows = load_label(args.label_file)

    total_marked = 0
    for path, regs in rows:
        # ảnh thật để biết H
        img_path = args.image_dir / Path(path).name
        if not img_path.exists():
            # PPOCRLabel có thể ghi path tương đối project root; thử lại
            img_path = Path(path)
        if not img_path.exists():
            print(f"[WARN] Không tìm thấy ảnh {path}, skip.")
            continue
        with Image.open(img_path) as im:
            W, H = im.size

        cutoff = args.threshold * H
        n_mark = 0
        for r in regs:
            pts = r.get("points", [])
            if not pts:
                continue
            min_y = min(p[1] for p in pts)
            if min_y > cutoff and not r.get("difficult", False):
                r["difficult"] = True
                n_mark += 1
        print(f"  {Path(path).name}: ảnh {W}x{H}, cutoff y={int(cutoff)}, "
              f"đánh dấu {n_mark}/{len(regs)} box → difficult.")
        total_marked += n_mark

    write_label(args.label_file, rows)
    print(f"[DONE] Tổng {total_marked} box đã được tag difficult.")
    print("       Mở lại ảnh trong PPOCRLabel để verify, "
          "rồi Export Label trước khi convert.")


if __name__ == "__main__":
    main()
