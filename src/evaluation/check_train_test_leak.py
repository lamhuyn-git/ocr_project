"""
Kiểm tra overlap giữa image_train/ và image_test/ ở 3 mức:

  1. FILENAME — trùng tên (vd. phone_good_001.jpg)
  2. FILE HASH (md5) — cùng nội dung byte-for-byte (kể cả tên khác)
  3. RESIZED PERCEPTUAL HASH (pHash) — gần giống nhau dù bị resize/recompress

Nếu mức 1 hoặc 2 có overlap → DATA LEAKAGE, phải xử lý.
Nếu chỉ mức 3 trùng → có thể do scan cùng 1 form, cần xem xét.

Usage:
    python -m src.evaluation.check_train_test_leak \
        --train-dir image_train \
        --test-dir image_test
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

EXTS = {".jpg", ".jpeg", ".png"}


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_of(path: Path) -> str | None:
    """Perceptual hash 8x8 mean-binary."""
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as im:
        im = im.convert("L").resize((8, 8))
        px = list(im.getdata())
        avg = sum(px) / len(px)
        bits = "".join("1" if p > avg else "0" for p in px)
    return f"{int(bits, 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--phash-threshold", type=int, default=5,
                    help="Hamming distance ≤ threshold = giống nhau (mức 3).")
    args = ap.parse_args()

    if not args.train_dir.exists():
        raise SystemExit(f"Không tìm thấy {args.train_dir}")
    if not args.test_dir.exists():
        raise SystemExit(f"Không tìm thấy {args.test_dir}")

    train = list_images(args.train_dir)
    test = list_images(args.test_dir)
    print(f"[INFO] train={len(train)} ảnh, test={len(test)} ảnh\n")

    # ---- Mức 1: filename ----
    train_names = {p.name for p in train}
    test_names = {p.name for p in test}
    name_overlap = sorted(train_names & test_names)
    print(f"=== MỨC 1: filename overlap ({len(name_overlap)}) ===")
    for n in name_overlap[:30]:
        print(f"  {n}")
    if len(name_overlap) > 30:
        print(f"  … (+{len(name_overlap) - 30} nữa)")
    print()

    # ---- Mức 2: md5 ----
    print("[INFO] Tính md5 cho train…")
    train_md5: Dict[str, Path] = {md5_of(p): p for p in train}
    print("[INFO] Tính md5 cho test…")
    test_md5_pairs = [(md5_of(p), p) for p in test]
    md5_overlap = [(p, train_md5[h]) for h, p in test_md5_pairs if h in train_md5]
    print(f"=== MỨC 2: md5 overlap ({len(md5_overlap)}) — IDENTICAL FILES ===")
    for tp, trp in md5_overlap[:30]:
        print(f"  test={tp}  ==  train={trp}")
    if len(md5_overlap) > 30:
        print(f"  … (+{len(md5_overlap) - 30} nữa)")
    print()

    # ---- Mức 3: pHash ----
    print(f"[INFO] Tính pHash cho train…")
    train_phash = {phash_of(p): p for p in train if phash_of(p)}
    print(f"[INFO] So pHash với test (threshold={args.phash_threshold})…")
    near_dupes = []
    for tp in test:
        ph = phash_of(tp)
        if not ph:
            continue
        for tph, trp in train_phash.items():
            d = hamming(ph, tph)
            if d <= args.phash_threshold and tp.name != trp.name:
                near_dupes.append((tp, trp, d))
                break
    print(f"=== MỨC 3: pHash near-duplicate ({len(near_dupes)}) ===")
    for tp, trp, d in near_dupes[:30]:
        print(f"  test={tp.name}  ~  train={trp.name}  (Hamming={d})")
    if len(near_dupes) > 30:
        print(f"  … (+{len(near_dupes) - 30} nữa)")
    print()

    print("=== TÓM TẮT ===")
    print(f"  Filename overlap:        {len(name_overlap)}")
    print(f"  Identical (md5) overlap: {len(md5_overlap)}  ← LEAKAGE NGHIÊM TRỌNG nếu > 0")
    print(f"  Near-duplicate (pHash):  {len(near_dupes)}  ← cần review")


if __name__ == "__main__":
    main()
