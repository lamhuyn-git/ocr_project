"""
Split dataset thành train/val/test với 2 ràng buộc bắt buộc:

  1. SPLIT Ở CẤP FORM (không phải crop) — tất cả crops từ cùng 1 form gốc
     phải vào cùng 1 split, tránh leak text content giữa các set.
  2. STRATIFIED THEO QUALITY BUCKET — mỗi nhóm (scan/phone_good/phone_low/
     phone_bad và các biến thể _hand) phân bố đồng đều ở cả 3 split.

Synth data không có form parent → đưa toàn bộ vào train (augmentation).

Workflow:
  Bước 0 (tùy chọn): Gộp handwriting Label.txt vào master image_train/Label.txt
  Bước 1: Đọc rec_gt.txt (real) + synth sources
  Bước 2: Stratified form-level split
  Bước 3: Ghi train/val/test (synth chỉ vào train)

Đầu vào:
  image_train/rec_gt.txt                        — label real crops
  image_train/synth/rec_gt_synth.txt            — synth chữ in
  image_train/hand_synth/rec_gt_synth_hand.txt  — synth chữ viết tay

Đầu ra:
  image_train/train/rec_gt_train.txt  (~70% form real + toàn bộ synth)
  image_train/val/rec_gt_val.txt      (~15% form real)
  image_train/test/rec_gt_test.txt    (~15% form real)
  image_train/form_split_map.txt      (form_key → split, để traceability)

Usage:
  # Gộp hand labels rồi split
  python src/data/split_dataset.py

  # Chỉ split (không gộp lại Label.txt)
  python src/data/split_dataset.py --skip-merge

  # Tuỳ chỉnh tỷ lệ
  python src/data/split_dataset.py --train-ratio 0.7 --val-ratio 0.15
"""
from __future__ import annotations

import argparse
import os
import re
import random
import shutil
from collections import defaultdict
from pathlib import Path

from form_utils import parse_form, stratified_form_split

# ─── Paths ────────────────────────────────────────────────────────────────────

MASTER_LABEL = Path("image_train/Label.txt")
REAL_LABEL   = Path("image_train/rec_gt.txt")
TRAIN_OUT    = Path("image_train/train/rec_gt_train.txt")
VAL_OUT      = Path("image_train/val/rec_gt_val.txt")
TEST_OUT     = Path("image_train/test/rec_gt_test.txt")
MAP_OUT      = Path("image_train/form_split_map.txt")

# Synth sources: tất cả chỉ đưa vào train
SYNTH_SOURCES = [
    (Path("image_train/synth/rec_gt_synth.txt"),           "synth"),
    (Path("image_train/hand_synth/rec_gt_synth_hand.txt"), ""),
]

# Handwriting Label.txt cần gộp vào MASTER_LABEL
# Key: folder chứa Label.txt  |  Value: bucket name để normalize path
# Lưu ý: scan_hand images nằm trong hand_writing/ (không có folder scan_hand riêng)
HAND_LABEL_SOURCES: dict[Path, str] = {
    Path("image_train/hand_writing"):    "hand_writing",
    Path("image_train/phone_good_hand"): "phone_good_hand",
    Path("image_train/phone_low_hand"):  "phone_low_hand",
}

SEED = 42


# ─── Step 0: Merge handwriting Label.txt → master Label.txt ──────────────────

def _normalize_hand_path(raw_path: str, bucket: str) -> str:
    """Chuẩn hoá path từ handwriting Label.txt sang format của master Label.txt.

    Ví dụ:
        "hand_writing/scan_hand_001.jpg"        → "image_train/hand_writing/scan_hand_001.jpg"
        "phone_good_hand/phone_good_hand_1.jpg" → "image_train/phone_good_hand/phone_good_hand_001.jpg"
    """
    filename = Path(raw_path).name
    return f"image_train/{bucket}/{filename}"


def _clean_transcription(json_str: str) -> str:
    """Loại bỏ ký tự \\r trong transcription (PPOCRLabel trên Windows)."""
    return json_str.replace("\\r", "").replace("\r", "")


def merge_hand_labels(dry_run: bool = False) -> int:
    """Gộp tất cả handwriting Label.txt vào MASTER_LABEL.

    - Normalize path về format: image_train/{bucket}/{filename}
    - Dedup: bỏ qua entries đã tồn tại trong master
    - Backup master trước khi ghi (master.Label.txt.bak)
    - Trả về số dòng mới được thêm vào.
    """
    # Đọc các path đã có trong master để dedup
    existing_paths: set[str] = set()
    if MASTER_LABEL.exists():
        for line in MASTER_LABEL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_paths.add(line.split("\t", 1)[0])

    new_lines: list[str] = []

    for folder, bucket in HAND_LABEL_SOURCES.items():
        label_file = folder / "Label.txt"
        if not label_file.exists():
            print(f"  [SKIP] {label_file} không tồn tại")
            continue

        raw_lines = label_file.read_text(encoding="utf-8").splitlines()
        added = 0
        for raw in raw_lines:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split("\t", 1)
            if len(parts) != 2:
                continue

            norm_path = _normalize_hand_path(parts[0], bucket)
            json_str  = _clean_transcription(parts[1])

            if norm_path in existing_paths:
                continue  # đã có, bỏ qua

            new_lines.append(f"{norm_path}\t{json_str}")
            existing_paths.add(norm_path)
            added += 1

        print(f"  {label_file}: {len(raw_lines)} dòng → {added} dòng mới")

    if not new_lines:
        print("  Không có dòng mới cần thêm.")
        return 0

    if not dry_run:
        # Backup
        if MASTER_LABEL.exists():
            bak = MASTER_LABEL.with_suffix(".txt.bak")
            shutil.copy2(MASTER_LABEL, bak)

        with MASTER_LABEL.open("a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f"  → Đã thêm {len(new_lines)} dòng vào {MASTER_LABEL}")
    else:
        print(f"  [dry-run] Sẽ thêm {len(new_lines)} dòng")

    return len(new_lines)


# ─── Step 1: Load lines ───────────────────────────────────────────────────────

def load_lines(path: Path, prefix: str = "") -> list[str]:
    """Đọc label file, tuỳ chọn thêm prefix vào image path."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if prefix:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                out.append(f"{prefix}/{parts[0]}\t{parts[1]}\n")
        else:
            out.append(line + "\n")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Form-level split dataset")
    ap.add_argument("--train-ratio",  type=float, default=0.70)
    ap.add_argument("--val-ratio",    type=float, default=0.15)
    ap.add_argument("--seed",         type=int,   default=SEED)
    ap.add_argument("--skip-merge",   action="store_true",
                    help="Bỏ qua bước gộp handwriting Label.txt")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Chỉ in thống kê, không ghi file")
    args = ap.parse_args()

    test_ratio = round(1.0 - args.train_ratio - args.val_ratio, 6)
    if test_ratio <= 0:
        raise SystemExit("train_ratio + val_ratio phải < 1.0")

    # ── Bước 0: Merge hand labels ──
    if not args.skip_merge:
        print("=== Bước 0: Gộp handwriting Label.txt ===")
        merge_hand_labels(dry_run=args.dry_run)
        print()

    # ── Bước 1: Load real + synth ──
    print("=== Bước 1: Load data ===")
    real_lines = load_lines(REAL_LABEL)
    print(f"Real samples (rec_gt.txt): {len(real_lines)}")

    all_synth: list[str] = []
    for synth_path, prefix in SYNTH_SOURCES:
        lines = load_lines(synth_path, prefix=prefix)
        print(f"Synth [{prefix}]: {len(lines)} samples")
        all_synth.extend(lines)
    print()

    # ── Bước 2: Group theo form ──
    print("=== Bước 2: Form-level grouping ===")
    crops_by_form: dict[str, list[str]] = defaultdict(list)
    bucket_of:     dict[str, str]       = {}
    unmatched:     list[str]            = []

    for line in real_lines:
        img_path = line.split("\t", 1)[0]
        parsed   = parse_form(Path(img_path).name)
        if parsed is None:
            unmatched.append(line)
            continue
        bucket, form_key = parsed
        crops_by_form[form_key].append(line)
        bucket_of[form_key] = bucket

    if unmatched:
        print(f"  [WARN] {len(unmatched)} dòng không khớp form pattern → vào train")

    forms_by_bucket: dict[str, list[str]] = defaultdict(list)
    for fk, b in bucket_of.items():
        forms_by_bucket[b].append(fk)

    total_forms = sum(len(v) for v in forms_by_bucket.values())
    print(f"  Tổng forms: {total_forms}")
    for b in sorted(forms_by_bucket):
        print(f"    {b}: {len(forms_by_bucket[b])} forms")
    print()

    # ── Bước 3: Stratified split ──
    print(f"=== Bước 3: Split {args.train_ratio:.0%}/{args.val_ratio:.0%}/{test_ratio:.0%} ===")
    rng = random.Random(args.seed)
    train_forms, val_forms, test_forms = stratified_form_split(
        forms_by_bucket, args.train_ratio, args.val_ratio, rng
    )

    for bucket in sorted(forms_by_bucket):
        forms  = forms_by_bucket[bucket]
        n_tr   = sum(1 for f in forms if f in train_forms)
        n_va   = sum(1 for f in forms if f in val_forms)
        n_te   = sum(1 for f in forms if f in test_forms)
        print(f"  {bucket:20s} {len(forms):3d} → train {n_tr}  val {n_va}  test {n_te}")

    assert not (train_forms & val_forms)
    assert not (train_forms & test_forms)
    assert not (val_forms   & test_forms)

    # ── Assemble ──
    train_part, val_part, test_part = [], [], []
    for fk, lines in crops_by_form.items():
        if fk in train_forms:   train_part.extend(lines)
        elif fk in val_forms:   val_part.extend(lines)
        else:                   test_part.extend(lines)

    train_part.extend(all_synth)
    train_part.extend(unmatched)
    rng.shuffle(train_part)

    n_real_train = len(train_part) - len(all_synth) - len(unmatched)
    print(f"\nTrain: {len(train_part):5d} samples"
          f"  ({n_real_train} real + {len(all_synth)} synth)")
    print(f"Val  : {len(val_part):5d} samples")
    print(f"Test : {len(test_part):5d} samples")

    if args.dry_run:
        print("\n[dry-run] Không ghi file.")
        return

    # ── Ghi output ──
    for p in (TRAIN_OUT, VAL_OUT, TEST_OUT, MAP_OUT):
        p.parent.mkdir(parents=True, exist_ok=True)

    TRAIN_OUT.write_text("".join(train_part), encoding="utf-8")
    VAL_OUT.write_text  ("".join(sorted(val_part)),  encoding="utf-8")
    TEST_OUT.write_text ("".join(sorted(test_part)), encoding="utf-8")

    with MAP_OUT.open("w", encoding="utf-8") as f:
        f.write("# form_key\tbucket\tsplit\n")
        for fk in sorted(train_forms): f.write(f"{fk}\t{bucket_of[fk]}\ttrain\n")
        for fk in sorted(val_forms):   f.write(f"{fk}\t{bucket_of[fk]}\tval\n")
        for fk in sorted(test_forms):  f.write(f"{fk}\t{bucket_of[fk]}\ttest\n")

    print(f"\n→ {TRAIN_OUT}")
    print(f"→ {VAL_OUT}")
    print(f"→ {TEST_OUT}")
    print(f"→ {MAP_OUT}")


if __name__ == "__main__":
    main()
