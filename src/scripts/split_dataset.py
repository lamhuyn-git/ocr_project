import random, os

with open('image_train/rec_gt.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

split = int(len(lines) * 0.8)
train_lines = lines[:split]
val_lines   = lines[split:]

os.makedirs('image_train/train', exist_ok=True)
os.makedirs('image_train/val',   exist_ok=True)

with open('image_train/train/rec_gt_train.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open('image_train/val/rec_gt_val.txt', 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"Train: {len(train_lines)} samples")
print(f"Val:   {len(val_lines)} samples")
