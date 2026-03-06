# OCR Hồ Sơ Tiếng Việt

Pipeline tự động đọc và trích xuất thông tin từ ảnh hồ sơ/giấy tờ tiếng Việt.

```
Ảnh → Tiền xử lý (OpenCV) → OCR (PaddleOCR) → KIE (Rule-based) → Validation → JSON
```

## Cài đặt nhanh

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Chạy

```bash
# Đặt ảnh vào thư mục images/
python main.py

# Hoặc chỉ định file cụ thể
python main.py images/cccd.jpg
```

## Cấu trúc

```
ocr_project/
├── images/          ← Ảnh đầu vào
├── outputs/         ← Kết quả JSON và ảnh debug
├── src/
│   ├── preprocess.py   — Tiền xử lý ảnh
│   ├── ocr_engine.py   — PaddleOCR
│   ├── kie.py          — Trích xuất thông tin
│   └── validator.py    — Kiểm tra nghiệp vụ
├── main.py          ← Chạy tại đây
└── requirements.txt
```

Xem file `HUONG_DAN_DU_AN.md` để biết hướng dẫn chi tiết từng bước.
