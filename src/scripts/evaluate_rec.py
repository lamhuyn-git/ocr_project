import os
import sys
import unicodedata
from collections import Counter

# Thêm project root vào path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddlex import create_model


def edit_distance(s1, s2) -> int:
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
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)

def compute_wer(pred: str, gt: str) -> float:
    pred_words = pred.split()
    gt_words   = gt.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return edit_distance(pred_words, gt_words) / len(gt_words)

def compute_precision_recall(pred: str, gt: str) -> tuple:
    """
    Precision: Tỷ lệ đúng của predict (so với chính nó)
    Recall: Tỷ lệ đúng của predict (so với GT)
    """

    pred_words = Counter(pred.split())
    gt_words   = Counter(gt.split())

    # Số từ đúng = giao của 2 counter (min của từng từ)
    tp = sum((pred_words & gt_words).values())

    total_pred = sum(pred_words.values())
    total_gt   = sum(gt_words.values())

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall    = tp / total_gt   if total_gt   > 0 else 0.0

    return precision, recall

def get_char_alignments(pred: str, gt: str) -> list:
    m, n = len(gt), len(pred)

    # Build full DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    substitutions = []
    i, j = m, n
    while i > 0 and j > 0:
        if gt[i - 1] == pred[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            substitutions.append((gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            # Deletion (GT char bị xóa)
            i -= 1
        else:
            # Insertion (Pred char thừa)
            j -= 1

    return substitutions

def update_confusion(confusion: dict, pred: str, gt: str):
    for gt_char, pred_char in get_char_alignments(pred, gt):
        key = (gt_char, pred_char)
        confusion[key] = confusion.get(key, 0) + 1

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

def evaluate(model_dir: str, model_name: str, val_samples: list):
    print(f"Evaluate model: {model_name}")
    print(f"Model dir: {model_dir}")
    print(f"The number of valuation samples: {len(val_samples)}")

    model = create_model(model_name, model_dir=model_dir)

    total            = len(val_samples)
    exact_match      = 0
    total_cer        = 0.0
    total_wer        = 0.0
    total_precision  = 0.0
    total_recall     = 0.0
    total_confidence = 0.0
    confusion        = {}
    errors           = []

    for idx, (img_path, gt_text) in enumerate(val_samples):
        if not os.path.exists(img_path):
            continue

        pred_text  = ""
        confidence = 0.0
        try:
            results = list(model.predict(img_path))
            if results:
                pred_text  = results[0].get('rec_text', '')
                confidence = results[0].get('rec_score', 0.0)
        except Exception as e:
            print(f"  Error on {img_path}: {e}")

        # Chuẩn hóa Unicode NFC để tránh lỗi combining diacritics (NFD vs NFC)
        pred_text = unicodedata.normalize('NFC', pred_text.strip())
        gt_text   = unicodedata.normalize('NFC', gt_text.strip())

        # Tính metrics
        is_match          = (pred_text == gt_text)
        cer               = compute_cer(pred_text, gt_text)
        wer               = compute_wer(pred_text, gt_text)
        precision, recall = compute_precision_recall(pred_text, gt_text)

        if is_match:
            exact_match += 1
        total_cer       += cer
        total_wer       += wer
        total_precision += precision
        total_recall    += recall
        total_confidence+= confidence

        # Cập nhật confusion matrix
        update_confusion(confusion, pred_text, gt_text)

        # Lưu lỗi để phân tích
        if not is_match:
            errors.append({
                'file':       os.path.basename(img_path),
                'gt':         gt_text,
                'pred':       pred_text,
                'cer':        cer,
                'wer':        wer,
                'confidence': confidence,
            })

        # Progress
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] Accuracy: {exact_match/(idx+1):.1%}  |  "
                  f"CER: {total_cer/(idx+1):.4f}  |  WER: {total_wer/(idx+1):.4f}")

    # ============================================================
    # Kết quả tổng hợp
    # ============================================================
    accuracy     = exact_match / total if total > 0 else 0
    avg_cer      = total_cer       / total if total > 0 else 0
    avg_wer      = total_wer       / total if total > 0 else 0
    avg_precision= total_precision / total if total > 0 else 0
    avg_recall   = total_recall    / total if total > 0 else 0
    avg_conf     = total_confidence/ total if total > 0 else 0

    print(f"  Accuracy (exact match) : {accuracy:.2%}  ({exact_match}/{total})")
    print(f"  CER (Character Error)  : {avg_cer:.4f}  ({avg_cer:.2%})")
    print(f"  WER (Word Error)       : {avg_wer:.4f}  ({avg_wer:.2%})")
    print(f"  Precision (word-level) : {avg_precision:.4f}  ({avg_precision:.2%})")
    print(f"  Recall    (word-level) : {avg_recall:.4f}  ({avg_recall:.2%})")
    print(f"  Confidence trung bình  : {avg_conf:.4f}  ({avg_conf:.2%})")

    # Top 20 lỗi nặng nhất (theo CER)
    errors.sort(key=lambda x: x['cer'], reverse=True)
    print(f"  Top 20 lỗi nặng nhất:")
    print(f"  {'File':<40} {'CER':>6} {'WER':>6} {'Conf':>6}")
    print(f"  {'-'*58}")
    for e in errors[:20]:
        print(f"  {e['file']:<40} {e['cer']:>5.2f} {e['wer']:>5.2f} {e['confidence']:>5.2f}")
        print(f"    GT:   {e['gt'][:80]}")
        print(f"    Pred: {e['pred'][:80]}")
        print()

    # Top 15 cặp ký tự bị nhầm nhiều nhất (Confusion Matrix)
    sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 15 Confusion Pairs (GT → Pred):")
    print(f"  {'GT':>5} → {'Pred':<5}  {'Count':>6}")
    print(f"  {'-'*25}")
    for (gt_char, pred_char), count in sorted_confusion[:15]:
        print(f"  {repr(gt_char):>5} → {repr(pred_char):<5}  {count:>6}")

    return {
        'model':          model_name,
        'accuracy':       accuracy,
        'cer':            avg_cer,
        'wer':            avg_wer,
        'precision':      avg_precision,
        'recall':         avg_recall,
        'avg_confidence': avg_conf,
        'total':          total,
        'exact_match':    exact_match,
        'num_errors':     len(errors),
        'confusion':      confusion,
    }

if __name__ == '__main__':
    VAL_LABEL = os.path.join(PROJECT_ROOT, 'image_train', 'val', 'rec_gt_val.txt')
    BASE_DIR  = os.path.join(PROJECT_ROOT, 'image_train')

    val_samples = load_val_data(VAL_LABEL, BASE_DIR)
    print(f"Loaded {len(val_samples)} val samples")

    models_to_eval = [
        ('mobile_lite (baseline)', os.path.join(PROJECT_ROOT, 'output_mobile_rec_lite', 'inference'),        'PP-OCRv5_mobile_rec'),
        ('mobile_new_db (new)',    os.path.join(PROJECT_ROOT, 'output_rec_mobile_new_db', 'inference'), 'PP-OCRv5_mobile_rec'),
    ]

    results = {}
    for label, model_dir, model_name in models_to_eval:
        if os.path.exists(model_dir):
            results[label] = evaluate(model_dir, model_name, val_samples)
        else:
            print(f"SKIP: {model_dir} not found")

    # So sánh kết quả nếu chạy cả 2
    if len(results) == 2:
        keys = list(results.keys())
        m, s = results[keys[0]], results[keys[1]]
        print("\n" + "=" * 60)
        print(f"  SO SÁNH: {keys[0]}  vs  {keys[1]}")
        print("=" * 60)
        print(f"  {'Chỉ số':<25} {keys[0]:>20} {keys[1]:>20} {'Winner':>10}")
        print(f"  {'-'*77}")
        for key, label, higher_better in [
            ('accuracy',       'Accuracy',       True),
            ('cer',            'CER',            False),
            ('wer',            'WER',            False),
            ('precision',      'Precision',      True),
            ('recall',         'Recall',         True),
            ('avg_confidence', 'Confidence TB',  True),
        ]:
            mv, sv = m[key], s[key]
            if higher_better:
                winner = keys[1] if sv > mv else (keys[0] if mv > sv else 'Tie')
            else:
                winner = keys[1] if sv < mv else (keys[0] if mv < sv else 'Tie')
            print(f"  {label:<25} {mv:>20.4f} {sv:>20.4f} {winner:>10}")
