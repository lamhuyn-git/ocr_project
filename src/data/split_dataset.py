"""
Split dataset thành train/val/test với 2 ràng buộc bắt buộc:

  1. SPLIT Ở CẤP FORM (không phải crop) — tất cả crops từ cùng 1 form gốc
     phải vào cùng 1 split, tránh leak text content giữa các set.
  2. STRATIFIED THEO QUALITY BUCKET — mỗi nhóm (scan/phone_good/phone_low/
     phone_bad) phân bố đồng đều ở cả 3 split.

Synth data không có form parent → đưa toàn bộ vào train (augmentation).

⚠ Lịch sử: phiên bản trước của file này shuffle crop-level → form-level leak
~97% giữa train và val → val accuracy bị thổi phồng. Đã sửa.

Đầu vào:
  image_train/rec_gt.txt              — label real, path relative image_train/
  image_train/synth/rec_gt_synth.txt  — label synth, path relative synth/

Đầu ra:
  image_train/train/rec_gt_train.txt  (~80% form real + toàn bộ synth)
  image_train/val/rec_gt_val.txt      (~10% form real)
  image_train/test/rec_gt_test.txt    (~10% form real)
  image_train/form_split_map.txt      (form_key → split, để traceability)

Usage:
  python src/data/split_dataset.py
  python src/data/split_dataset.py --train-ratio 0.7 --val-ratio 0.15
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

from form_utils import parse_form, stratified_form_split

REAL_LABEL  = 'image_train/rec_gt.txt'
SYNTH_LABEL = 'image_train/synth/rec_gt_synth.txt'
TRAIN_OUT   = 'image_train/train/rec_gt_train.txt'
VAL_OUT     = 'image_train/val/rec_gt_val.txt'
TEST_OUT    = 'image_train/test/rec_gt_test.txt'
MAP_OUT     = 'image_train/form_split_map.txt'
SEED        = 42


def load_lines(path: str, prefix: str = "") -> list[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if prefix:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    out.append(f"{os.path.join(prefix, parts[0])}\t{parts[1]}\n")
            else:
                out.append(line if line.endswith("\n") else line + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio",   type=float, default=0.1)
    ap.add_argument("--seed",        type=int,   default=SEED)
    args = ap.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise SystemExit("train_ratio + val_ratio phải < 1.0")

    real_lines  = load_lines(REAL_LABEL)
    synth_lines = load_lines(SYNTH_LABEL, prefix="synth")
    print(f"Real samples : {len(real_lines)}")
    print(f"Synth samples: {len(synth_lines)}")

    # Group real crops theo form
    crops_by_form: dict[str, list[str]] = defaultdict(list)
    bucket_of: dict[str, str] = {}
    unmatched: list[str] = []
    for line in real_lines:
        path = line.split("\t", 1)[0]
        parsed = parse_form(Path(path).name)
        if parsed is None:
            unmatched.append(line)
            continue
        bucket, fk = parsed
        crops_by_form[fk].append(line)
        bucket_of[fk] = bucket

    if unmatched:
        print(f"[WARN] {len(unmatched)} dòng không match form pattern, "
              f"đưa hết vào train.")

    # Stratified split form theo bucket
    forms_by_bucket: dict[str, list[str]] = defaultdict(list)
    for fk, b in bucket_of.items():
        forms_by_bucket[b].append(fk)

    rng = random.Random(args.seed)
    train_forms, val_forms, test_forms = stratified_form_split(
        forms_by_bucket, args.train_ratio, args.val_ratio, rng
    )

    print(f"\n=== Split form ({args.train_ratio:.0%}/"
          f"{args.val_ratio:.0%}/{test_ratio:.0%}) ===")
    for bucket in sorted(forms_by_bucket):
        forms = forms_by_bucket[bucket]
        n_train = len([f for f in forms if f in train_forms])
        n_val   = len([f for f in forms if f in val_forms])
        n_test  = len([f for f in forms if f in test_forms])
        print(f"  {bucket:12s} {len(forms):3d} → "
              f"train {n_train}  val {n_val}  test {n_test}")

    assert not (train_forms & val_forms)
    assert not (train_forms & test_forms)
    assert not (val_forms & test_forms)

    # Assemble line lists
    train_part = []
    val_part   = []
    test_part  = []
    for fk, lines in crops_by_form.items():
        if fk in train_forms:   train_part.extend(lines)
        elif fk in val_forms:   val_part  .extend(lines)
        else:                   test_part .extend(lines)
    train_part.extend(synth_lines)
    train_part.extend(unmatched)

    rng.shuffle(train_part)

    for path in (TRAIN_OUT, VAL_OUT, TEST_OUT, MAP_OUT):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(TRAIN_OUT, "w", encoding="utf-8") as f: f.writelines(train_part)
    with open(VAL_OUT,   "w", encoding="utf-8") as f: f.writelines(sorted(val_part))
    with open(TEST_OUT,  "w", encoding="utf-8") as f: f.writelines(sorted(test_part))

    with open(MAP_OUT, "w", encoding="utf-8") as f:
        f.write("# form_key\tbucket\tsplit\n")
        for fk in sorted(train_forms): f.write(f"{fk}\t{bucket_of[fk]}\ttrain\n")
        for fk in sorted(val_forms):   f.write(f"{fk}\t{bucket_of[fk]}\tval\n")
        for fk in sorted(test_forms):  f.write(f"{fk}\t{bucket_of[fk]}\ttest\n")

    print(f"\nTrain: {len(train_part)} samples "
          f"({sum(len(crops_by_form[fk]) for fk in train_forms)} real + "
          f"{len(synth_lines)} synth) → {TRAIN_OUT}")
    print(f"Val  : {len(val_part)} samples → {VAL_OUT}")
    print(f"Test : {len(test_part)} samples → {TEST_OUT}")
    print(f"Map  : {MAP_OUT}")


if __name__ == "__main__":
    main()
