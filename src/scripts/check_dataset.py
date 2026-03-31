import os

def check_labels(label_file, base_dir):
    missing = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Invalid format: {line.strip()}")
                continue
            img_path, text = parts
            full_path = os.path.join(base_dir, img_path)
            if not os.path.exists(full_path):
                missing.append(img_path)

    if missing:
        print(f"Missing {len(missing)} image files!")
        for m in missing[:5]:
            print(f"   {m}")
    else:
        print(f"Dataset OK!")

check_labels('image_train/train/rec_gt_train.txt', 'image_train')
check_labels('image_train/val/rec_gt_val.txt',     'image_train')