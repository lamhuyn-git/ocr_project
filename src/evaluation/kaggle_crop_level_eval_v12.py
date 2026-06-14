"""
kaggle_crop_level_eval_v12.py — Đánh giá model PaddleOCR v12 ở MỨC CROP (không qua pipeline).

Mục tiêu: so chỉ số recognition của v12 khi cắt field bằng bbox GROUND-TRUTH (vị trí đúng)
thay vì đi qua pipeline (align homography + ROI template + detect). Tách riêng chất lượng
NHẬN DẠNG khỏi sai số căn chỉnh/định vị → biết v12 trên crop tốt hơn hay kém hơn pipeline.

Cách map toạ độ: GT bbox nằm trong hệ canvas chuẩn (CANVAS_W x CANVAS_H). Ảnh scan cùng tỉ lệ
khung → chỉ cần scale bbox theo (scan_w/CANVAS_W, scan_h/CANVAS_H), KHÔNG cần align homography.

SELF-CONTAINED: chỉ phụ thuộc paddleocr, opencv, numpy, rapidfuzz. Không import module project
→ chạy thẳng trên Kaggle. Logic crop/join/digit-grid/normalize/metric copy nguyên bản từ repo
để kết quả so sánh được với eval_results.json của pipeline.

Chạy trên Kaggle (bật Internet để Paddle tải det + rec-arch lần đầu):
    python kaggle_crop_level_eval_v12.py \
        --scan-dir   /kaggle/input/<scan-dataset> \
        --rec-dir    /kaggle/input/<v12-dataset>/inference \
        --baseline   /kaggle/input/<v12-dataset>/eval_results.json \
        --out        /kaggle/working/crop_level_eval_v12.json
"""
import argparse
import json
import os
import re
import statistics
import unicodedata

import cv2
import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz.distance import Levenshtein

# Kích thước canvas chuẩn (configs/templates/ct01_tt53.0.yaml -> page.width/height).
CANVAS_W, CANVAS_H = 1654, 2339

# Field cần eval: type + normalize ops (trích từ config). digit_grid xử lý xoá vạch lưới.
# Bám đúng config để metric so được với pipeline. (gioi_tinh_snap chưa hiện thực -> bỏ qua,
# giống apply_normalizers gốc skip op lạ.)
FIELDS = {
    "kinh_gui":                       ("text_line",  ["nfc", "trim", "collapse_ws", "strip_trail_punct"]),
    "ho_chu_dem_va_ten":              ("text_line",  ["nfc", "trim", "collapse_ws", "strip_trail_punct"]),
    "ngay_thang_nam_sinh":            ("text_line",  ["trim", "date_ddmmyyyy", "strip_space"]),
    "gioi_tinh":                      ("text_line",  ["nfc", "trim", "strip_space"]),
    "so_dinh_dan_ca_nhan":            ("digit_grid", ["trim", "strip_space"]),
    "so_dien_thoai_lien_he":          ("number",     ["trim", "strip_space"]),
    "email":                          ("text_line",  ["trim", "strip_space", "email_fix"]),
    "ho_chu_dem_va_ten_chu_ho":       ("text_line",  ["nfc", "trim", "collapse_ws", "strip_trail_punct"]),
    "moi_quan_he_voi_chu_ho":         ("text_line",  ["nfc", "trim", "collapse_ws", "strip_trail_punct"]),
    "so_dinh_dan_ca_nhan_cua_chu_ho": ("digit_grid", ["trim", "strip_space"]),
    "noi_dung_de_nghi":               ("text_block", ["nfc", "trim", "collapse_ws", "strip_label"]),
}

# ─────────────────────────── Normalizers (copy từ src/ocr/normalizer.py) ───────────────────────────
_LABEL_RE = re.compile(r"^.*?\(\s*2\s*\)\s*[:：]?\s*", re.S)
_DOUBLE_AT = re.compile(r"@{2,}")
_DOMAIN_FIXES = {"gonail": "gmail", "gmal": "gmail", "gmall": "gmail", "gmil": "gmail",
                 "gail": "gmail", "hoymail": "hotmail", "hotmal": "hotmail", "yaho": "yahoo"}


def _to_date(s):
    nums = re.findall(r"\d+", s)
    if len(nums) >= 3:
        try:
            d, m, y = int(nums[0]), int(nums[1]), int(nums[2])
            if y < 100:
                y += 2000 if y < 50 else 1900
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                return f"{d:02d}/{m:02d}/{y:04d}"
        except (ValueError, IndexError):
            pass
    return s


def _fix_email(s):
    out = re.sub(r"[.,;:!?\s]+$", "", s.strip().lower())
    out = _DOUBLE_AT.sub("@", out)
    if "@" in out:
        local, domain = out.rsplit("@", 1)
        local = re.sub(r"[.,;:!?\s]+$", "", local)
        parts = domain.split(".", 1)
        if parts[0] in _DOMAIN_FIXES:
            parts[0] = _DOMAIN_FIXES[parts[0]]
        domain = ".".join(parts)
        domain = re.sub(r"\.(com|vn|net|org|edu)(\..*)?$", lambda m: "." + m.group(1), domain)
        out = local + "@" + domain
    return out


NORMALIZERS = {
    "nfc": lambda s: unicodedata.normalize("NFC", s),
    "trim": lambda s: s.strip(),
    "collapse_ws": lambda s: re.sub(r"\s+", " ", s).strip(),
    "strip_space": lambda s: re.sub(r"\s+", "", s),
    "strip_commas": lambda s: re.sub(r"\s*,\s*", " ", s).strip(),
    "to_lower": lambda s: s.lower(),
    "date_ddmmyyyy": _to_date,
    "strip_label": lambda s: (lambda m: s[m.end():].strip() if m else s)(_LABEL_RE.match(s)),
    "strip_trail_punct": lambda s: re.sub(r"[\s.,;:!?。，、…]+$", "", s).strip(),
    "email_fix": _fix_email,
}


def apply_normalizers(text, ops):
    if not text or not ops:
        return text
    out = text
    for op in ops:
        fn = NORMALIZERS.get(op)
        if fn is None:
            continue
        try:
            out = fn(out)
        except Exception:
            pass
    return out


# ─────────────────────────── Metrics (copy từ src/evaluation/metrics.py) ───────────────────────────
def cer(pred, gt):
    pred, gt = pred.strip(), gt.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def wer(pred, gt):
    pw, gw = pred.strip().split(), gt.strip().split()
    if not gw:
        return 0.0 if not pw else 1.0
    return Levenshtein.distance(pw, gw) / len(gw)


def aggregate_metrics(records):
    if not records:
        return {}

    def stats(vals):
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    return {
        "n": len(records),
        "cer": stats([r["cer"] for r in records]),
        "wer": stats([r["wer"] for r in records]),
        "em": round(sum(r["em"] for r in records) / len(records), 4),
    }


# ─────────────────────────── OCR crop/join (copy logic crop_ocr.py + digit_grid.py) ───────────────────────────
def run_ocr(ocr, img):
    raw = ocr.ocr(img)
    if not raw or raw[0] is None:
        return []
    r = raw[0]
    parsed = []
    for text, conf, bbox in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"]):
        if not text.strip():
            continue
        pts = bbox.tolist()
        parsed.append({
            "text": text.strip(),
            "confidence": float(conf),
            "center_y": sum(p[1] for p in pts) / 4,
            "x_left": min(p[0] for p in pts),
        })
    parsed.sort(key=lambda x: (x["center_y"], x["x_left"]))
    return parsed


def optional_preprocess(crop, min_height=48):
    if crop.size == 0:
        return crop
    h = crop.shape[0]
    if 0 < h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _group_lines(blocks, y_tol=14):
    lines = []
    for b in blocks:
        if lines and abs(b["center_y"] - lines[-1][-1]["center_y"]) <= y_tol:
            lines[-1].append(b)
        else:
            lines.append([b])
    for line in lines:
        line.sort(key=lambda b: b["x_left"])
    return lines


def join_blocks(blocks, line_sep="\n", token_sep=" "):
    if not blocks:
        return ""
    lines = _group_lines(blocks)
    return line_sep.join(token_sep.join(b["text"] for b in line) for line in lines)


def remove_grid_lines(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 3)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 15), 1))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk),
                           cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = gray.copy()
    cleaned[lines > 0] = 255
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def recognize_crop(ocr, crop, ftype):
    """Trả raw_text cho 1 crop field. digit_grid: xoá vạch lưới rồi OCR cả dải."""
    if crop is None or crop.size == 0:
        return ""
    if ftype == "digit_grid":
        crop = remove_grid_lines(crop)
    blocks = run_ocr(ocr, optional_preprocess(crop))
    return join_blocks(blocks)


# ─────────────────────────── Eval loop ───────────────────────────
def bucket_of(stem):
    return "hand" if stem.startswith("hand") else ("print" if stem.startswith("print") else "other")


def scale_box(bbox, sx, sy, W, H):
    x1, y1, x2, y2 = bbox
    return (max(0, int(x1 * sx)), max(0, int(y1 * sy)),
            min(W, int(x2 * sx)), min(H, int(y2 * sy)))


def build_ocr(rec_dir, det_dir):
    kw = dict(lang="vi", device="cpu",
              text_detection_model_name="PP-OCRv5_mobile_det",
              text_recognition_model_name="PP-OCRv5_mobile_rec",
              text_recognition_model_dir=rec_dir,
              use_doc_orientation_classify=False,
              use_textline_orientation=False,
              use_doc_unwarping=False)
    if det_dir:
        kw["text_detection_model_dir"] = det_dir
    return PaddleOCR(**kw)


def run_eval(scan_dir, ocr):
    gts = sorted(f for f in os.listdir(scan_dir) if f.endswith(".json"))
    records = []
    for gi, gtname in enumerate(gts, 1):
        stem = gtname[:-5]
        img_path = next((os.path.join(scan_dir, stem + e) for e in (".jpg", ".jpeg", ".png")
                         if os.path.exists(os.path.join(scan_dir, stem + e))), None)
        if not img_path:
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] đọc ảnh lỗi: {img_path}")
            continue
        H, W = img.shape[:2]
        sx, sy = W / CANVAS_W, H / CANVAS_H
        gt = json.load(open(os.path.join(scan_dir, gtname), encoding="utf-8"))
        print(f"[{gi}/{len(gts)}] {stem} ({W}x{H})")

        for fname, (ftype, norm) in FIELDS.items():
            if fname not in gt or "text" not in gt[fname] or "bbox" not in gt[fname]:
                continue
            box = scale_box(gt[fname]["bbox"], sx, sy, W, H)
            crop = img[box[1]:box[3], box[0]:box[2]]
            raw = recognize_crop(ocr, crop, ftype)
            pred = apply_normalizers(raw, norm)
            g = apply_normalizers(gt[fname]["text"], norm)
            # so 2 chiều: bỏ phẩy + lower (giống pipeline run_and_eval)
            pred_m = apply_normalizers(pred, ["strip_commas", "to_lower"])
            g_m = apply_normalizers(g, ["strip_commas", "to_lower"])
            records.append({
                "form": stem, "bucket": bucket_of(stem), "field": fname,
                "gt": g, "pred": pred,
                "cer": cer(pred_m, g_m), "wer": wer(pred_m, g_m),
                "em": pred_m.strip() == g_m.strip(),
            })
    return records


def _line(m):
    return f"CER={m['cer']['mean']*100:.2f}%  WER={m['wer']['mean']*100:.2f}%  EM={m['em']*100:.1f}%"


def _delta(crop_m, base_overall):
    """In so sánh crop-level vs baseline pipeline (mũi tên: ▼ tốt hơn/giảm lỗi, ▲ tệ hơn)."""
    if not base_overall:
        return
    print("\n--- SO SÁNH crop-level  vs  pipeline (baseline) ---")
    for k, better_lower in (("cer", True), ("wer", True), ("em", False)):
        if k == "em":
            c, b = crop_m["em"], base_overall["em"]
        else:
            c, b = crop_m[k]["mean"], base_overall[k]["mean"]
        d = c - b
        # CER/WER: giảm = tốt. EM: tăng = tốt.
        good = (d < 0) if better_lower else (d > 0)
        tag = "TỐT HƠN" if good else ("BẰNG" if abs(d) < 1e-9 else "KÉM HƠN")
        print(f"  {k.upper():3s}  crop={c*100:6.2f}%  pipeline={b*100:6.2f}%  Δ={d*100:+6.2f}pp  -> {tag}")


def main():
    ap = argparse.ArgumentParser(description="Eval PaddleOCR v12 ở mức crop (GT bbox, không qua pipeline)")
    ap.add_argument("--scan-dir", default="/kaggle/input/scan", help="thư mục chứa *.jpg + *.json GT")
    ap.add_argument("--rec-dir", default="/kaggle/input/paddle-v12/inference", help="thư mục inference rec v12")
    ap.add_argument("--det-dir", default=None, help="(tuỳ chọn) thư mục det offline; mặc định Paddle tự tải")
    ap.add_argument("--baseline", default=None, help="(tuỳ chọn) eval_results.json pipeline để so sánh")
    ap.add_argument("--out", default="crop_level_eval_v12.json", help="file kết quả đầu ra")
    args = ap.parse_args()

    if not os.path.isdir(args.rec_dir):
        raise FileNotFoundError(f"Không thấy rec model dir: {args.rec_dir}")
    if not os.path.isdir(args.scan_dir):
        raise FileNotFoundError(f"Không thấy scan dir: {args.scan_dir}")

    print(f"Khởi tạo PaddleOCR v12 (rec={args.rec_dir})...")
    ocr = build_ocr(args.rec_dir, args.det_dir)
    records = run_eval(args.scan_dir, ocr)

    overall = aggregate_metrics(records)
    by_bucket = {b: aggregate_metrics([r for r in records if r["bucket"] == b])
                 for b in sorted({r["bucket"] for r in records})}
    by_field = {f: aggregate_metrics([r for r in records if r["field"] == f])
                for f in sorted({r["field"] for r in records})}

    out = {"engine": "paddle_v12", "mode": "crop_level_gt_bbox",
           "overall": overall, "by_bucket": by_bucket, "by_field": by_field, "samples": records}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n========== KẾT QUẢ CROP-LEVEL (v12) ==========")
    print(f"  OVERALL n={overall['n']}  {_line(overall)}")
    for b, m in by_bucket.items():
        if m:
            print(f"    {b:6s} n={m['n']:3d}  {_line(m)}")
    print("  --- theo field ---")
    for f, m in by_field.items():
        if m:
            print(f"    {f:32s} n={m['n']:3d}  {_line(m)}")

    base = None
    if args.baseline and os.path.exists(args.baseline):
        base = json.load(open(args.baseline, encoding="utf-8")).get("overall")
    _delta(overall, base)
    print(f"\n  -> đã lưu: {args.out}")


if __name__ == "__main__":
    main()
