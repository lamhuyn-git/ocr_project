import os
import sys

# Thêm project root vào path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddlex import create_model
from PIL import Image
import numpy as np


# ============================================================
# CER (Character Error Rate) — dùng edit distance
# ============================================================
def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance giữa 2 chuỗi."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(pred: str, gt: str) -> float:
    """CER = edit_distance / len(ground_truth). Trả về 0.0 nếu cả hai rỗng."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)


# ============================================================
# Load val data
# ============================================================
def load_val_data(val_label_path: str, base_dir: str):
    """Đọc file label, trả về list (image_path, ground_truth)."""
    samples = []
    with open(val_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            img_rel, gt_text = parts
            img_path = os.path.join(base_dir, img_rel)
            samples.append((img_path, gt_text))
    return samples


# ============================================================
# Đánh giá
# ============================================================
def evaluate(model_dir: str, model_name: str, val_samples: list):
    """Chạy OCR trên từng ảnh crop và so sánh với ground truth."""

    print(f"\n{'='*60}")
    print(f"  Đánh giá model: {model_name}")
    print(f"  Model dir: {model_dir}")
    print(f"  Số mẫu val: {len(val_samples)}")
    print(f"{'='*60}\n")

    # Dùng PaddleX create_model để load trực tiếp model recognition
    model = create_model(model_name, model_dir=model_dir)

    total = len(val_samples)
    exact_match = 0
    total_cer = 0.0
    total_confidence = 0.0
    errors = []

    for idx, (img_path, gt_text) in enumerate(val_samples):
        if not os.path.exists(img_path):
            continue

        # Chạy recognition trực tiếp
        pred_text = ""
        confidence = 0.0
        try:
            results = list(model.predict(img_path))
            if results:
                pred_text = results[0].get('rec_text', '')
                confidence = results[0].get('rec_score', 0.0)
        except Exception as e:
            print(f"  Error on {img_path}: {e}")

        pred_text = pred_text.strip()
        gt_text = gt_text.strip()

        # Tính metrics
        is_match = (pred_text == gt_text)
        cer = compute_cer(pred_text, gt_text)

        if is_match:
            exact_match += 1
        total_cer += cer
        total_confidence += confidence

        # Lưu lỗi để phân tích
        if not is_match:
            errors.append({
                'file': os.path.basename(img_path),
                'gt': gt_text,
                'pred': pred_text,
                'cer': cer,
                'confidence': confidence,
            })

        # Progress
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] Accuracy: {exact_match/(idx+1):.1%}  |  CER: {total_cer/(idx+1):.4f}")

    # ============================================================
    # Kết quả tổng hợp
    # ============================================================
    accuracy = exact_match / total if total > 0 else 0
    avg_cer = total_cer / total if total > 0 else 0
    avg_conf = total_confidence / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy (exact match) : {accuracy:.2%}  ({exact_match}/{total})")
    print(f"  CER (Character Error)  : {avg_cer:.4f}  ({avg_cer:.2%})")
    print(f"  Confidence trung bình  : {avg_conf:.4f}  ({avg_conf:.2%})")
    print(f"{'='*60}\n")

    # Top 20 lỗi nặng nhất
    errors.sort(key=lambda x: x['cer'], reverse=True)
    print(f"  Top 20 lỗi nặng nhất:")
    print(f"  {'File':<40} {'CER':>6} {'Conf':>6}")
    print(f"  {'-'*52}")
    for e in errors[:20]:
        print(f"  {e['file']:<40} {e['cer']:>5.2f} {e['confidence']:>5.2f}")
        print(f"    GT:   {e['gt'][:80]}")
        print(f"    Pred: {e['pred'][:80]}")
        print()

    return {
        'model': model_name,
        'accuracy': accuracy,
        'cer': avg_cer,
        'avg_confidence': avg_conf,
        'total': total,
        'exact_match': exact_match,
        'num_errors': len(errors),
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    VAL_LABEL = os.path.join(PROJECT_ROOT, 'image_train', 'val', 'rec_gt_val.txt')
    BASE_DIR = os.path.join(PROJECT_ROOT, 'image_train')

    val_samples = load_val_data(VAL_LABEL, BASE_DIR)
    print(f"Loaded {len(val_samples)} val samples")

    # --- Model 1: PP-OCRv5_mobile_rec (lightweight) ---
    mobile_dir = os.path.join(PROJECT_ROOT, 'output_mobile_rec_lite', 'inference')
    if os.path.exists(mobile_dir):
        result_mobile = evaluate(mobile_dir, 'PP-OCRv5_mobile_rec', val_samples)
    else:
        print(f"SKIP: {mobile_dir} not found")
        result_mobile = None

    # --- Model 2: PP-OCRv5_server_rec (heavy) ---
    server_dir = os.path.join(PROJECT_ROOT, 'output_rec', 'inference')
    if os.path.exists(server_dir):
        result_server = evaluate(server_dir, 'PP-OCRv5_server_rec', val_samples)
    else:
        print(f"SKIP: {server_dir} not found (chưa có, sẽ đánh giá sau)")
        result_server = None

    # --- So sánh ---
    if result_mobile and result_server:
        print(f"\n{'='*60}")
        print(f"  SO SÁNH 2 MODEL")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'Mobile':>12} {'Server':>12}")
        print(f"  {'-'*49}")
        print(f"  {'Accuracy':<25} {result_mobile['accuracy']:>11.2%} {result_server['accuracy']:>11.2%}")
        print(f"  {'CER':<25} {result_mobile['cer']:>11.4f} {result_server['cer']:>11.4f}")
        print(f"  {'Confidence':<25} {result_mobile['avg_confidence']:>11.4f} {result_server['avg_confidence']:>11.4f}")
        print(f"{'='*60}")
