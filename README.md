# OCR Tờ Khai Cư Trú (CT01) - Tiếng Việt

Pipeline tự động đọc và trích xuất thông tin từ Tờ Khai Thay Đổi Thông Tin Cư Trú (Mẫu CT01), sử dụng PaddleOCR fine-tuned cho văn bản in tiếng Việt.

```
Ảnh → Tiền xử lý (OpenCV) → OCR (PaddleOCR fine-tuned) → KIE (Rule-based) → Validation → JSON
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Chạy

```bash
python main.py                      # Ảnh mặc định
python main.py path/to/image.jpg    # Chỉ định ảnh
```

## Cấu trúc

```
ocr_project/
├── main.py                 ← Entry point
├── dict_vi.txt             ← Bộ ký tự tiếng Việt (683 ký tự)
├── requirements.txt
├── src/
│   ├── preprocess.py       — Tiền xử lý ảnh (resize, deskew, denoise, CLAHE)
│   ├── ocr_engine.py       — PaddleOCR wrapper
│   ├── kie.py              — Trích xuất thông tin (regex + keyword)
│   ├── validator.py        — Kiểm tra nghiệp vụ (CCCD, SĐT, ngày sinh)
│   └── scripts/
│       ├── split_dataset.py    — Chia train/val (80/20)
│       ├── finetune_rec.py     — Fine-tune recognition model
│       ├── check_dataset.py    — Kiểm tra dataset
│       └── check_charset.py    — Kiểm tra bộ ký tự
├── image_train/            ← Dữ liệu training (crop_img + labels)
├── output_rec/             ← Model fine-tuned (không push lên git)
└── outputs/                ← Kết quả OCR debug
```

## Fine-tune

Xem `HUONG_DAN_FINETUNE.md` để biết hướng dẫn fine-tune model recognition cho văn bản hành chính tiếng Việt.

## Dataset

- Nguồn ảnh gốc: backup trên Kaggle
- Annotation: PPOCRLabel
- Train/Val split: 80/20 (seed=42)
- Tổng: 5,427 mẫu crop text
