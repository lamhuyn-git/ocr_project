# Hướng Dẫn Fine-tune PaddleOCR cho Văn Bản Hành Chính Tiếng Việt

> **Scope:** Chỉ nhận diện chữ **đánh máy / in sẵn** — không bao gồm chữ viết tay
> **Mục tiêu:** CER < 3% trên font chữ in đậm văn bản hành chính tiếng Việt
> **Thời gian ước tính:** ~1 tuần (data + annotation) + ~2 ngày (training + tích hợp)
> **Phần cần fine-tune:** Recognition model (CRNN/SVTRNet)

---

## Tổng quan quy trình

```
Thu thập ảnh → Annotation (PPOCRLabel) → Chuẩn bị dataset → Fine-tune → Đánh giá → Tích hợp
```

---

## BƯỚC 1 — Cài đặt môi trường

```bash
# Cài PPOCRLabel (tool annotation)
pip install PPOCRLabel

# Cài PaddleX (framework fine-tune của PaddleOCR 3.x)
pip install paddlex

# Kiểm tra
PPOCRLabel --help
```

> ⚠️ **Lưu ý:** PaddleOCR 3.x dùng PaddleX để fine-tune, khác với cách của PaddleOCR 2.x

---

## BƯỚC 2 — Thu thập ảnh (200–260 ảnh)

### 2.1 Scope ảnh — chữ đánh máy / in sẵn

Vì scope giới hạn ở chữ in, **tờ khai phải được điền bằng máy tính rồi in ra**,
không dùng tờ khai điền tay. Điều này cho phép bạn **tự tạo data** chủ động
mà không phụ thuộc vào người khác — lợi thế lớn về timeline.

| Loại ảnh | Số lượng | Lưu ý |
|---|---|---|
| Scan máy, 300 DPI, lưu PNG | ~80 ảnh | Ưu tiên loại này — chất lượng chuẩn nhất |
| Chụp điện thoại, ánh sáng tốt | ~80 ảnh | Giữ thẳng, không nghiêng quá 5° |
| Chụp điện thoại, ánh sáng yếu / nghiêng nhẹ | ~60 ảnh | Mô phỏng điều kiện thực tế xấu |
| Photocopy | ~40 ảnh | In → photocopy 1–2 lần |
| **Tổng** | **~260 ảnh** | |

### 2.2 Nội dung tờ khai trong ảnh

- Dùng **dữ liệu giả** (tên giả, CCCD giả) — không cần thông tin thật, đảm bảo privacy
- Mỗi ảnh nên dùng **nội dung khác nhau** (tên khác, ngày sinh khác) để tránh model học vẹt
- Ưu tiên đa dạng **font chữ**: thử in bằng Times New Roman, Arial, font mặc định Word
- Ưu tiên đa dạng **cỡ chữ**: 11pt, 12pt, 13pt
- Các ô số CCCD (ô kẻ sẵn) vẫn cần điền — dùng font Courier New cho số vào ô kẻ

### 2.3 Cấu trúc thư mục ảnh

```
finetune_data/
└── raw_images/
    ├── scan_001.png       ← đặt tên theo loại giúp debug dễ hơn
    ├── scan_002.png
    ├── phone_001.jpg
    ├── phone_002.jpg
    ├── photocopy_001.jpg
    └── ...
```

---

## BƯỚC 3 — Annotation với PPOCRLabel

PPOCRLabel sẽ tự động dùng PaddleOCR để pre-annotate (nhận diện trước),
bạn chỉ việc **sửa lại những chỗ sai** → tiết kiệm 70–80% thời gian.

### 3.1 Khởi động PPOCRLabel

```bash
# Chạy với ngôn ngữ tiếng Việt
PPOCRLabel --lang vi --kie True
```

### 3.2 Quy trình annotation từng bước

**Bước a:** Mở thư mục ảnh
- `File` → `Open Dir` → chọn thư mục `raw_images/`

**Bước b:** Auto-annotation
- Nhấn `Auto recognition` → PPOCRLabel tự nhận diện tất cả ảnh
- Đây là bước tốn thời gian nhất (~2–5 giây/ảnh)

**Bước c:** Kiểm tra và sửa từng ảnh
- Duyệt qua từng ảnh trong danh sách bên trái
- Với mỗi bounding box được highlight:
  - Nếu **text đúng** → bỏ qua
  - Nếu **text sai** → double-click vào box → sửa lại text đúng
  - Nếu **box bị thiếu** → vẽ thêm box mới (nhấn `W` để vẽ rectangle)
  - Nếu **box thừa** → chọn box → nhấn `Delete`

**Bước d:** Lưu annotation
- `File` → `Save` (hoặc Ctrl+S) sau mỗi ảnh

> 💡 **Mẹo tốc độ:** Tập trung sửa những dòng chữ in đậm tiêu đề trước —
> đây là nguồn lỗi chính trong trường hợp của bạn.

### 3.3 Export dataset cho Recognition

Sau khi annotate xong:
- `File` → `Export Recognition Result`
- PPOCRLabel tự **crop từng dòng chữ** và tạo file label

Output sẽ có dạng:

```
finetune_data/
├── crop_img/
│   ├── img_001_0.jpg    ← dòng chữ đã crop
│   ├── img_001_1.jpg
│   └── ...
└── rec_gt.txt           ← file label
```

Nội dung `rec_gt.txt`:
```
crop_img/img_001_0.jpg	CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
crop_img/img_001_1.jpg	Độc lập - Tự do - Hạnh phúc
crop_img/img_001_2.jpg	TỜ KHAI THAY ĐỔI THÔNG TIN CƯ TRÚ
```

> ⚠️ **Quan trọng:** Dấu phân cách giữa đường dẫn và text là `\t` (Tab), không phải Space

---

## BƯỚC 4 — Chuẩn bị dataset

### 4.1 Chia train/val

```python
# split_dataset.py — chạy script này để chia 80/20
import random, shutil, os

with open('finetune_data/rec_gt.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

split = int(len(lines) * 0.8)
train_lines = lines[:split]
val_lines   = lines[split:]

os.makedirs('finetune_data/train', exist_ok=True)
os.makedirs('finetune_data/val',   exist_ok=True)

with open('finetune_data/train/rec_gt_train.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open('finetune_data/val/rec_gt_val.txt', 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"Train: {len(train_lines)} samples")
print(f"Val:   {len(val_lines)} samples")
```

### 4.2 Cấu trúc dataset cuối cùng

```
finetune_data/
├── crop_img/              ← tất cả ảnh crop (train + val dùng chung)
│   ├── img_001_0.jpg
│   └── ...
├── train/
│   └── rec_gt_train.txt
└── val/
    └── rec_gt_val.txt
```

### 4.3 Kiểm tra dataset trước khi train

```python
# check_dataset.py — kiểm tra không có file thiếu
import os

def check_labels(label_file):
    missing = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"⚠️  Sai format: {line.strip()}")
                continue
            img_path, text = parts
            full_path = os.path.join('finetune_data', img_path)
            if not os.path.exists(full_path):
                missing.append(img_path)

    if missing:
        print(f"❌ Thiếu {len(missing)} file ảnh!")
        for m in missing[:5]:
            print(f"   {m}")
    else:
        print(f"✅ Dataset OK!")

check_labels('finetune_data/train/rec_gt_train.txt')
check_labels('finetune_data/val/rec_gt_val.txt')
```

---

## BƯỚC 5 — Fine-tune Recognition Model

PaddleOCR 3.x dùng PaddleX. Có 2 cách:

### Cách A: Fine-tune qua Python API (khuyến nghị)

```python
# finetune_rec.py
from paddlex import create_model

# Load model recognition tiếng Việt
model = create_model("PP-OCRv4_server_rec")

# Fine-tune
model.train(
    dataset_type    = "MSTextRecDataset",
    dataset_root_path = "./finetune_data",
    train_annot_path  = "train/rec_gt_train.txt",
    val_annot_path    = "val/rec_gt_val.txt",

    # Hyperparameters
    epochs_iters    = 50,       # Số epoch — tăng nếu chưa hội tụ
    batch_size      = 32,       # Giảm xuống 8–16 nếu thiếu RAM
    learning_rate   = 0.0001,   # Learning rate nhỏ vì đang fine-tune

    # Đường dẫn lưu model
    output          = "./output_rec",

    # Dùng pretrained weights tiếng Việt
    pretrain_weight_path = None  # Dùng mặc định PP-OCRv4
)
```

```bash
python finetune_rec.py
```

### Cách B: Fine-tune qua CLI

```bash
paddlex --train \
  --pipeline OCR \
  --train_annot_path finetune_data/train/rec_gt_train.txt \
  --val_annot_path finetune_data/val/rec_gt_val.txt \
  --epochs_iters 50 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --output output_rec
```

### Theo dõi quá trình training

```
Epoch [1/50] - train_loss: 2.34, val_acc: 0.61
Epoch [5/50] - train_loss: 1.12, val_acc: 0.78
Epoch [10/50] - train_loss: 0.67, val_acc: 0.87
...
Epoch [50/50] - train_loss: 0.23, val_acc: 0.94  ← mục tiêu
```

> 📊 **Mục tiêu:** val_acc > 0.90 là đạt yêu cầu cho văn bản hành chính

---

## BƯỚC 6 — Đánh giá model sau fine-tune

```python
# evaluate_finetuned.py
from paddlex import create_model

model = create_model("PP-OCRv4_server_rec",
                     model_dir="./output_rec/best_model")

result = model.evaluate(
    dataset_root_path = "./finetune_data",
    val_annot_path    = "val/rec_gt_val.txt"
)

print(f"Accuracy: {result['acc']:.4f}")
print(f"NED (Normalized Edit Distance): {result['norm_edit_dis']:.4f}")
```

**Giải thích metrics:**
- `acc` (Accuracy): % dòng chữ nhận diện **hoàn toàn đúng**
- `norm_edit_dis` (NED): tương đương 1 - CER, càng gần 1 càng tốt

---

## BƯỚC 7 — Tích hợp model mới vào project

Sau khi fine-tune xong, cập nhật `src/ocr_engine.py`:

```python
def get_ocr_instance() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        print("Initial PaddleOCR with fine-tuned model...")
        _ocr_instance = PaddleOCR(
            lang='vi',
            device='cpu',
            # Trỏ tới model recognition đã fine-tune
            rec_model_dir='./output_rec/best_model',
        )
        print("Initialized PaddleOCR successfully!\n")
    return _ocr_instance
```

---

## BƯỚC 8 — So sánh trước/sau

Chạy lại pipeline trên ảnh test và so sánh:

```python
# compare_results.py
import sys
sys.path.insert(0, 'src')

from preprocess import preprocess_pipeline
from ocr_engine import run_ocr, get_ocr_instance

# Test ảnh cũ
img = preprocess_pipeline('images/happy_case.png')
ocr = get_ocr_instance()
results = run_ocr(ocr, img)

# Tính CER thủ công với ground truth
ground_truth = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
predicted    = results[2]['text']  # dòng thứ 3

import editdistance
cer = editdistance.eval(predicted, ground_truth) / len(ground_truth)
print(f"CER: {cer:.2%}")
```

```bash
pip install editdistance
python compare_results.py
```

---

## Tóm tắt timeline (scope chữ in)

| Ngày | Công việc | Ghi chú |
|---|---|---|
| Ngày 1 | Tạo dữ liệu giả + in/scan 80 ảnh đầu tiên | Chủ động 100%, không phụ thuộc ai |
| Ngày 2 | Chụp điện thoại 80 ảnh + photocopy 40 ảnh | Hoàn thành bộ 260 ảnh |
| Ngày 3 | Annotation bằng PPOCRLabel (~20 ảnh/giờ với chữ in) | Nhanh hơn 3x so với viết tay |
| Ngày 4 sáng | Annotation nốt + split dataset + kiểm tra |  |
| Ngày 4 chiều | Fine-tune Recognition model (2–4 giờ CPU) | |
| Ngày 5 | Đánh giá + tích hợp vào project + test thực tế | Còn ~3 tuần buffer cho KIE, Validator, polish |

---

## Lưu ý quan trọng

1. **Scope boundary rõ ràng** — Model này chỉ xử lý form được điền bằng máy tính rồi in ra. Nếu nhận ảnh form viết tay, pipeline nên detect và báo lỗi thay vì trả kết quả sai.
2. **Đừng annotate sai** — 1 label sai còn tệ hơn không có. Nếu không chắc text là gì, bỏ box đó đi.
3. **Ưu tiên chất lượng hơn số lượng** — 200 ảnh annotation đúng tốt hơn 500 ảnh có nhiều lỗi.
4. **Giữ lại test set riêng** — trước khi bắt đầu, tách ra 20–30 ảnh không annotate để test cuối cùng.
5. **Backup model gốc** — PaddleOCR tự động lưu model gốc, nhưng hãy ghi lại đường dẫn model cũ để rollback nếu cần.

---

## Roadmap sau 1 tháng (nếu muốn mở rộng)

Sau khi hoàn thành scope chữ in, v2 có thể mở rộng thêm:
- **Chữ viết tay** — cần thêm 3–4 tuần data collection + annotation
- **Nhiều loại form** — mở rộng FIELD_KEYWORDS trong kie.py
- **Batch processing** — xử lý nhiều ảnh cùng lúc
