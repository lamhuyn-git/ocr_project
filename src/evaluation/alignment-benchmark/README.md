# Alignment Benchmark — Cổng quyết định ORB (Phase 01)

Bộ đánh giá xác minh **2 giả thuyết lớn** trước khi cam kết xây nhánh config-space detection cho CT01:
1. CT01 có bao nhiêu phiên bản layout? (1 reference đủ chưa?)
2. ORB có đủ mạnh để căn chỉnh (align) mọi loại ảnh về khung chuẩn không? Có method nào tốt hơn?

Dữ liệu: **67 ảnh thật** trong `test_image/` (6 tier: scan, scan_hand, phone_good, phone_good_hand,
phone_low, phone_low_hand) + GT box `test_image/label_test.txt`. Reference canonical:
`real_test/template.jpg` (1654×2339, aspect 1.4141 ≈ A4).

## Cách chạy
```bash
.venv/bin/python src/evaluation/alignment-benchmark/field-drift-spike.py
.venv/bin/python src/evaluation/alignment-benchmark/param-sweep.py
.venv/bin/python src/evaluation/alignment-benchmark/detector-compare.py
```
Output thô lưu sẵn trong `results/*.out.txt`.

## Metric: field-drift (gắn ROI)
Vì mục tiêu cuối là crop ROI cố định sau khi align, metric **không** dùng reprojection error chung chung
mà đo trực tiếp thứ ta cần: **độ ổn định vị trí field trong khung canonical**.

- Với mỗi field nhãn-in cố định (1.Họ, 2.Ngày sinh, 4.CCCD, 5.SĐT, 10.Nội dung), lấy **landmark** =
  `(mép trái x, y trung bình)` của GT box → vị trí nhãn in, **bền với độ dài chữ người dân điền**.
- Chiếu landmark qua phép biến đổi (homography H với ORB / scale với resize) về canonical.
- Đo **độ tản** cùng-field giữa 67 ảnh:
  - `std_x, std_y` — độ chụm điển hình (px). Nhỏ = ổn định.
  - `max_dev` — lệch xa tâm nhất (ca tệ nhất). **Đây là số quyết định ngân sách padding ROI.**
- `mean_std` = trung bình std mọi field; `worst_maxdev` = max_dev tệ nhất mọi field.

## Kết quả

### 1. Số version layout: **1** (xác nhận)
Cả 67 ảnh khớp 1 template, 0 anchor thiếu → **không cần layout classification** trước align.

### 2. ORB vs baseline resize-only (`field-drift-spike`)
| field | ORB std_x | ORB std_y | ORB max_dev | RESIZE max_dev |
|-------|----:|----:|----:|----:|
| f1_hoten | 5.2 | 8.9 | 47.9 | 158.9 |
| f2_ngaysinh | 4.0 | 9.8 | 50.0 | 158.5 |
| f4_cccd | 2.6 | 4.7 | **23.3** | 187.3 |
| f5_sdt | 3.7 | 8.4 | 47.3 | 165.6 |
| f10_noidung | 4.7 | 19.2 | 98.1 | 127.1 |

- **ORB homography fail: 0/67** trên cả 6 tier.
- ORB chụm ~**5-7px** vs resize ~**30-40px** → **ORB thắng baseline ~6×**. Cả ảnh scan thẳng cũng drift
  30px với resize-only → **mọi tier đều cần feature-align**, không tách nhánh deskew_only riêng (KISS, 1 code path).
- `f10_noidung` max_dev cao (98px) một phần do là field nhiều dòng → y-centroid nhạy hơn; xử lý riêng (ROI cao, text_block).

### 3. Độ nhạy tham số (`param-sweep`) — RANSAC×Lowe, 12 tổ hợp
`fail` luôn **0**; `mean_std` **7.0–7.6**; `worst_maxdev` **96.9–107.1**. → Kết quả **không phải artifact**
của việc chọn RANSAC=5.0/Lowe=0.75. Chốt mặc định **RANSAC_REPROJ=5.0, Lowe=0.75**.

### 4. So detector (`detector-compare`) — ORB vs AKAZE vs BRISK vs SIFT
| detector | fail | mean_std | worst_maxdev | ms/ảnh | avg_kp |
|----------|---:|---:|---:|---:|---:|
| **ORB** | 0 | 7.1 | 98.1 | **53** | 3000 |
| AKAZE | 0 | 7.2 | 102.9 | 641 | 13593 |
| BRISK | 0 | 7.1 | 103.5 | 1902 | 25935 |
| SIFT | 0 | 7.0 | 101.6 | 490 | 3000 |

- **Độ chính xác ngang nhau** (chênh ~2-5px = nhiễu đo); kể cả SIFT không chính xác hơn ORB ở bài này.
- **Tốc độ ORB thắng tuyệt đối:** nhanh hơn SIFT ~9×, AKAZE ~12×, BRISK ~36×.
- SURF loại từ đầu: `opencv-python` chuẩn không build (patent).

## Quyết định
- **Dùng ORB** (nfeatures=3000, Lowe 0.75, findHomography RANSAC 5.0): cùng độ chính xác các đối thủ
  nhưng nhanh nhất nhiều lần — đúng "the right tool" cho feature-based registration form chuẩn hóa.
- **Padding budget Phase 02 (thô, từ max_dev):** single-line ~30–50px (f4_cccd có thể siết ~25px);
  `f10_noidung` xử lý riêng (ROI cao). Sẽ tinh chỉnh bằng `measure_padding_from_jitter` khi calibrate.
- **Đường nâng cấp nếu sau gặp ảnh khó** (xoay mạnh/mờ nặng): chuyển **SIFT** (không phải AKAZE/BRISK).

## Giới hạn (thành thật)
- "Mọi loại ảnh" = phân bố 67 ảnh trong 6 tier hiện có; **chưa** test xoay 90°/180°, mờ nặng, cháy sáng,
  mất >30% header. Giữ `fallback_resize` cho ca ngoài phân bố.
- Metric đo qua GT landmark (mép-trái/y-giữa), không đo trực tiếp pixel value sau crop → padding vẫn cần
  overlay kiểm mắt thường ở Phase 02.

## Unresolved
- CT01 có ảnh xoay 90°/180° trong thực tế không? (quyết định có cần bước auto-rotate trước ORB).
- Ngưỡng quality nào kích `fallback_resize` thay vì ORB? (chốt khi build `quality_estimator`).
