"""
Split dataset thành train/val 80/20, có tuỳ chọn merge synth data vào train only.

  - Val set: chỉ lấy từ real data (honest evaluation baseline)
  - Train set: real 80% + toàn bộ synth (augmentation)

Đầu vào:
  image_train/rec_gt.txt             — label real (path relative image_train/)
  image_train/synth/rec_gt_synth.txt — label synth (path relative image_train/synth/)

Đầu ra:
  image_train/train/rec_gt_train.txt
  image_train/val/rec_gt_val.txt
"""
import os
import random

REAL_LABEL  = 'image_train/rec_gt.txt'
SYNTH_LABEL = 'image_train/synth/rec_gt_synth.txt'
TRAIN_OUT   = 'image_train/train/rec_gt_train.txt'
VAL_OUT     = 'image_train/val/rec_gt_val.txt'
SEED        = 42
RATIO       = 0.8


def _load_with_prefix(path: str, prefix: str = '') -> list:
    """Đọc label file, prepend prefix vào path nếu cần."""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not prefix:
        return lines
    out = []
    for line in lines:
        parts = line.rstrip('\n').split('\t', 1)
        if len(parts) == 2:
            out.append(f"{os.path.join(prefix, parts[0])}\t{parts[1]}\n")
    return out


def main() -> None:
    real  = _load_with_prefix(REAL_LABEL)
    synth = _load_with_prefix(SYNTH_LABEL, prefix='synth')

    print(f"Real samples : {len(real)}")
    print(f"Synth samples: {len(synth)}")

    # Shuffle real rồi split 80/20
    random.seed(SEED)
    random.shuffle(real)
    split = int(len(real) * RATIO)
    real_train = real[:split]
    real_val   = real[split:]

    # Train = real_train + toàn bộ synth; Val = real_val only
    train = real_train + synth
    random.shuffle(train)
    val = real_val

    # Write
    os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(VAL_OUT),   exist_ok=True)
    with open(TRAIN_OUT, 'w', encoding='utf-8') as f:
        f.writelines(train)
    with open(VAL_OUT, 'w', encoding='utf-8') as f:
        f.writelines(val)

    print(f"\nTrain: {len(train)} samples ({len(real_train)} real + {len(synth)} synth) → {TRAIN_OUT}")
    print(f"Val  : {len(val)} samples (real only) → {VAL_OUT}")


if __name__ == '__main__':
    main()
