# OCR Tờ Khai Cư Trú (CT01) - Tiếng Việt

Pipeline tự động đọc và trích xuất thông tin từ Tờ Khai Thay Đổi Thông Tin Cư Trú (Mẫu CT01), sử dụng PaddleOCR fine-tuned cho văn bản in tiếng Việt.

```
Ảnh → Căn chỉnh khung (ORB) → Tiền xử lý (OpenCV) → OCR (PaddleOCR fine-tuned) → KIE (Rule-based) → Validation → JSON
```

## Yêu cầu

- Python **3.9+**
- macOS / Linux (đã test trên macOS Apple Silicon)

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Lưu ý:** `paddlepaddle` / `paddleocr` khá nặng. Nếu chỉ chạy/kiểm thử bước **căn chỉnh ảnh**
> (`src/alignment/`) thì chỉ cần `opencv-python` + `numpy` — không cần PaddlePaddle.

## Chạy

`main.py` duyệt toàn bộ ảnh trong thư mục `test_image/` (đuôi `.jpg/.jpeg/.png`), chạy
tiền xử lý → OCR → KIE, và lưu ảnh kết quả vào `outputs/test_results/`.

```bash
python main.py
```

Kiểm thử riêng bước **căn chỉnh khung** (Phase 01) trên ảnh trong `real_test/`:

```bash
python real_test/run_alignment.py    # warp test_alignment_*.jpg → real_test/outputs/alignment/
```

## Cấu trúc

```
ocr_project/
├── main.py                       ← Entry point (duyệt test_image/, chạy full pipeline)
├── requirements.txt
├── assets/
│   ├── ct01_reference.jpg        — Ảnh CT01 canonical dùng làm reference cho ORB
│   └── fonts/                    — Font phục vụ sinh dữ liệu synthetic
├── models/                       — Model weights & inference (không push lên git)
├── src/
│   ├── alignment/                — Căn chỉnh ảnh về khung canonical 1654×2339 (Phase 01)
│   │   ├── form_aligner.py       · orchestrator align_form() + hằng số CANONICAL_W/H
│   │   ├── orb_register.py       · ORB + Lowe + findHomography (RANSAC)
│   │   └── quality_estimator.py  · blur score (variance Laplacian)
│   ├── preprocess/               — Tiền xử lý ảnh (resize, deskew, denoise, CLAHE)
│   ├── ocr/                      — PaddleOCR wrapper (engine, block_merger, visualize)
│   ├── kie/                      — Trích xuất thông tin (find_label, find_value, cell_row)
│   ├── validator/                — Kiểm tra nghiệp vụ (CCCD, SĐT, ngày sinh)
│   ├── data/                     — Scripts xử lý/sinh/làm sạch dataset
│   └── evaluation/               — Đo OCR + benchmark căn chỉnh (alignment-benchmark/)
├── real_test/                    — Ảnh thật + script kiểm thử (không push lên git)
├── test_image/                   — Ảnh đầu vào cho main.py (không push lên git)
├── outputs/                      — Kết quả OCR/debug (không push lên git)
└── plans/                        — Tài liệu kế hoạch theo phase (không push lên git)
```

## Căn chỉnh khung (Phase 01)

`align_form(img, debug_name=None) -> (warped, meta)` đăng ký ảnh về khung **canonical
1654×2339** (A4 @ 200 DPI) bằng ORB feature registration:

- Thử 4 hướng `{0°, 90°, 180°, 270°}`, chọn hướng có nhiều inlier nhất (short-circuit nếu 0° đủ tốt).
- `meta` trả về: `method` (`"orb"` / `"fallback_resize"`), `rotate`, `n_matches`, `n_inliers`,
  `reproj_error`, `quality`.
- ORB fail → fallback resize (không crash).

Benchmark (cổng quyết định): `src/evaluation/alignment-benchmark/` — 67/67 ảnh thật align thành
công, reproj error ~1.6px.

## Đánh giá OCR

```bash
python src/evaluation/evaluate-ocr.py --model-label "v9" --output outputs/eval_results_v9.json

python src/evaluation/evaluate-ocr.py \
  --label-file real_test/Label.txt \
  --image-dir real_test \
  --output real_test/outputs/eval_results_v9.json \
  --result-dir real_test/outputs/test_results \
  --model-label "v9"
```

Metrics:
- **Accuracy** (exact match): % mẫu dự đoán đúng hoàn toàn
- **CER** (Character Error Rate): tỷ lệ lỗi mức ký tự (càng thấp càng tốt)
- **Confidence** trung bình

## Dataset

- Nguồn ảnh gốc: backup trên Kaggle
- Annotation: PPOCRLabel (`PPOCRLabel --lang vi`)
- Train/Val split: 80/20 (seed=42)
- Tổng: ~5,427 mẫu crop text
