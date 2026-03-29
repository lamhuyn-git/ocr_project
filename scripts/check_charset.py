import os

label_files = [
    "image_train/train/rec_gt_train.txt",
    "image_train/val/rec_gt_val.txt",
]

all_chars = set()
total_lines = 0

for lf in label_files:
    if not os.path.exists(lf):
        print(f"Not found: {lf}")
        continue
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                all_chars.update(parts[1])
                total_lines += 1

print(f"Total label lines: {total_lines}")
print(f"Number of unique characters in dataset: {len(all_chars)}")
print(f"\n--- List of characters ---")
print("".join(sorted(all_chars)))

# Check for special Vietnamese characters
viet_special = set("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
                   "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ")

missing = viet_special - all_chars
if missing:
    print(f"\nThe following Vietnamese characters are MISSING from the dataset ({len(missing)} characters):")
    print("".join(sorted(missing)))
else:
    print("\nDataset contains all commonly used Vietnamese characters.")
