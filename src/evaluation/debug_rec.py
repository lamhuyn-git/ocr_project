"""
debug_rec.py — Phân tích chuyên sâu tại sao PaddleOCR accuracy thấp

Chạy: python src/scripts/debug_rec.py
"""

import os
import sys
import unicodedata
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddlex import create_model

# ─── Config ───────────────────────────────────────────────────────────────────
VAL_LABEL = os.path.join(PROJECT_ROOT, 'image_train', 'val', 'rec_gt_val.txt')
BASE_DIR  = os.path.join(PROJECT_ROOT, 'image_train')

# Đổi thành model bạn muốn debug
MODEL_DIR  = os.path.join(PROJECT_ROOT, 'output_rec', 'inference')
MODEL_NAME = 'PP-OCRv5_mobile_rec'

TEST_LIMIT = None  # None = toàn bộ val set, hoặc đặt số ví dụ 500

# ─── Load data ────────────────────────────────────────────────────────────────
def load_val_data(path, base):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                img_rel, gt = parts
                samples.append((os.path.join(base, img_rel), gt))
    return samples


# ─── Helpers ──────────────────────────────────────────────────────────────────
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def cer(pred, gt):
    return edit_distance(pred, gt) / len(gt) if gt else (0.0 if not pred else 1.0)

def has_encoding_issue(text):
    """Phát hiện smart quotes và ký tự lạ thường do lỗi encoding."""
    suspects = ['\u201c', '\u201d', '\u2018', '\u2019',  # smart quotes
                '\ufffd',                                  # replacement char
                '\x00']                                    # null byte
    return any(c in text for c in suspects)

def label_type(text):
    """Phân loại label theo nội dung."""
    t = text.strip()
    if not t:
        return 'empty'
    if t.isdigit():
        return 'numbers_only'
    if all(c.isdigit() or c in '/-.: ' for c in t):
        return 'date_or_id'
    if len(t) <= 10:
        return 'short_text'
    if len(t) <= 30:
        return 'medium_text'
    return 'long_text'


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print(f"  DEBUG: PaddleOCR Recognition Model")
    print(f"  Model : {MODEL_DIR}")
    print(f"{'='*65}\n")

    # Kiểm tra model tồn tại
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] Không tìm thấy model tại: {MODEL_DIR}")
        print("  → Kiểm tra lại đường dẫn MODEL_DIR trong script.")
        # Gợi ý tìm model
        for candidate in ['output_rec', 'output_mobile_rec_lite', 'output_rec_mobile_new_db']:
            d = os.path.join(PROJECT_ROOT, candidate, 'inference')
            status = '✅' if os.path.exists(d) else '❌'
            print(f"  {status} {d}")
        return

    val_samples = load_val_data(VAL_LABEL, BASE_DIR)
    if TEST_LIMIT:
        val_samples = val_samples[:TEST_LIMIT]
    print(f"Val samples: {len(val_samples)}")

    # ── 1. Phân tích label TRƯỚC khi chạy inference ──────────────────────────
    print(f"\n{'─'*65}")
    print("  [1] PHÂN TÍCH LABEL (ground truth)")
    print(f"{'─'*65}")

    lengths = [len(gt) for _, gt in val_samples]
    encoding_issues = [(p, gt) for p, gt in val_samples if has_encoding_issue(gt)]

    print(f"  Độ dài label: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"  Label có encoding issue: {len(encoding_issues)} / {len(val_samples)} "
          f"({len(encoding_issues)/len(val_samples)*100:.1f}%)")

    if encoding_issues[:5]:
        print(f"\n  Ví dụ encoding issue:")
        for p, gt in encoding_issues[:5]:
            print(f"    [{os.path.basename(p)}] {repr(gt[:80])}")

    # Phân bố theo label_type
    type_counts = Counter(label_type(gt) for _, gt in val_samples)
    print(f"\n  Phân loại label:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = '█' * int(c / len(val_samples) * 30)
        print(f"    {t:<15} {c:>5} ({c/len(val_samples)*100:4.1f}%)  {bar}")

    # ── 2. Inference ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [2] CHẠY INFERENCE")
    print(f"{'─'*65}")

    model = create_model(MODEL_NAME, model_dir=MODEL_DIR)

    records = []  # list of dict với đầy đủ thông tin
    for idx, (img_path, gt_raw) in enumerate(val_samples):
        if not os.path.exists(img_path):
            continue

        gt   = unicodedata.normalize('NFC', gt_raw.strip())
        pred = ''
        conf = 0.0
        try:
            results = list(model.predict(img_path))
            if results:
                pred = unicodedata.normalize('NFC', results[0].get('rec_text', '').strip())
                conf = results[0].get('rec_score', 0.0)
        except Exception as e:
            pass

        is_match = (pred == gt)
        char_er  = cer(pred, gt)
        ltype    = label_type(gt)

        records.append({
            'file':       os.path.basename(img_path),
            'gt':         gt,
            'pred':       pred,
            'conf':       conf,
            'is_match':   is_match,
            'cer':        char_er,
            'gt_len':     len(gt),
            'ltype':      ltype,
            'enc_issue':  has_encoding_issue(gt),
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == len(val_samples):
            done = idx + 1
            acc  = sum(r['is_match'] for r in records) / len(records)
            print(f"  [{done}/{len(val_samples)}] running accuracy: {acc:.2%}")

    total   = len(records)
    correct = sum(r['is_match'] for r in records)
    avg_cer = sum(r['cer'] for r in records) / total
    avg_conf= sum(r['conf'] for r in records) / total

    print(f"\n  Overall — Accuracy: {correct/total:.2%}  |  "
          f"CER: {avg_cer:.2%}  |  Conf: {avg_conf:.2%}")

    # ── 3. Accuracy theo label length ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [3] ACCURACY THEO ĐỘ DÀI LABEL")
    print(f"{'─'*65}")

    buckets = [(1,5), (6,10), (11,25), (26,50), (51,128), (129,999)]
    print(f"  {'Range':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Avg CER':>9} {'Avg Conf':>10}")
    print(f"  {'-'*62}")
    for lo, hi in buckets:
        recs = [r for r in records if lo <= r['gt_len'] <= hi]
        if not recs:
            continue
        c = sum(r['is_match'] for r in recs)
        a = c / len(recs)
        avg_c = sum(r['cer'] for r in recs) / len(recs)
        avg_cf= sum(r['conf'] for r in recs) / len(recs)
        bar   = '█' * int(a * 20)
        print(f"  [{lo:3d}–{hi:3d}]{'':<5} {c:>8} {len(recs):>8} {a:>9.1%} {avg_c:>8.1%} {avg_cf:>9.1%}  {bar}")

    # ── 4. Accuracy theo label type ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [4] ACCURACY THEO LOẠI LABEL")
    print(f"{'─'*65}")

    print(f"  {'Type':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Avg CER':>9}")
    print(f"  {'-'*55}")
    for ltype in ['numbers_only', 'date_or_id', 'short_text', 'medium_text', 'long_text', 'empty']:
        recs = [r for r in records if r['ltype'] == ltype]
        if not recs:
            continue
        c = sum(r['is_match'] for r in recs)
        a = c / len(recs)
        avg_c = sum(r['cer'] for r in recs) / len(recs)
        print(f"  {ltype:<15} {c:>8} {len(recs):>8} {a:>9.1%} {avg_c:>8.1%}")

    # ── 5. Confidence calibration ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [5] CONFIDENCE CALIBRATION")
    print(f"{'─'*65}")
    print("  (Model lý tưởng: conf 90% → accuracy ~90%)")
    print()

    conf_buckets = [(0.0,0.5), (0.5,0.7), (0.7,0.8), (0.8,0.9), (0.9,0.95), (0.95,1.01)]
    print(f"  {'Conf range':<15} {'Count':>7} {'Accuracy':>10} {'Avg conf':>10}  Calibration")
    print(f"  {'-'*60}")
    for lo, hi in conf_buckets:
        recs = [r for r in records if lo <= r['conf'] < hi]
        if not recs:
            continue
        c    = sum(r['is_match'] for r in recs)
        acc  = c / len(recs)
        acf  = sum(r['conf'] for r in recs) / len(recs)
        diff = acc - acf
        flag = '⚠️  OVERCONFIDENT' if diff < -0.2 else ('✅ OK' if abs(diff) < 0.1 else '⚠️  gap')
        print(f"  [{lo:.2f}–{hi:.2f}){'':<5} {len(recs):>7} {acc:>9.1%} {acf:>9.1%}  {flag}")

    # ── 6. Encoding issue analysis ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [6] ENCODING ISSUE ANALYSIS")
    print(f"{'─'*65}")

    enc_recs   = [r for r in records if r['enc_issue']]
    clean_recs = [r for r in records if not r['enc_issue']]

    if enc_recs:
        enc_acc   = sum(r['is_match'] for r in enc_recs) / len(enc_recs)
        clean_acc = sum(r['is_match'] for r in clean_recs) / len(clean_recs) if clean_recs else 0
        print(f"  Label CÓ encoding issue : {len(enc_recs):>5} mẫu → accuracy {enc_acc:.1%}")
        print(f"  Label KHÔNG có issue    : {len(clean_recs):>5} mẫu → accuracy {clean_acc:.1%}")
        print(f"\n  → Nếu chênh lệch lớn, encoding errors là nguyên nhân chính.")
    else:
        print("  Không phát hiện encoding issue rõ ràng trong ground truth labels.")
        print("  → Encoding errors có thể đã 'bị ẩn' sau khi normalize NFC.")
        print("  → Xem thêm confusion matrix để tìm pattern lạ.")

    # ── 7. Top confusion pairs ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [7] TOP 20 CONFUSION PAIRS (GT → Pred)")
    print(f"{'─'*65}")

    confusion = defaultdict(int)
    for r in records:
        if not r['is_match']:
            # Đơn giản: so sánh từng ký tự (chỉ với label ngắn ≤50 để tránh alignment noise)
            if r['gt_len'] <= 50:
                for gc, pc in zip(r['gt'], r['pred']):
                    if gc != pc:
                        confusion[(gc, pc)] += 1
                # Ký tự bị xóa (GT dài hơn pred)
                for gc in r['gt'][len(r['pred']):]:
                    confusion[(gc, '<DEL>')] += 1
                # Ký tự thừa (pred dài hơn GT)
                for pc in r['pred'][len(r['gt']):]:
                    confusion[('<INS>', pc)] += 1

    top_conf = sorted(confusion.items(), key=lambda x: -x[1])[:20]
    print(f"  {'GT char':>10}  →  {'Pred char':<10}  {'Count':>6}")
    print(f"  {'-'*35}")
    for (gc, pc), cnt in top_conf:
        note = ''
        if gc in ('"', '"', ''', '''):
            note = '  ← smart quote!'
        print(f"  {repr(gc):>10}  →  {repr(pc):<10}  {cnt:>6}{note}")

    # ── 8. Top 15 lỗi nặng nhất ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [8] TOP 15 LỖI NẶNG NHẤT (cao CER)")
    print(f"{'─'*65}")

    errors = [r for r in records if not r['is_match']]
    errors.sort(key=lambda x: x['cer'], reverse=True)
    for r in errors[:15]:
        print(f"\n  File: {r['file']}  |  CER: {r['cer']:.0%}  |  Conf: {r['conf']:.0%}")
        print(f"  GT  : {r['gt'][:100]}")
        print(f"  Pred: {r['pred'][:100]}")

    print(f"\n{'='*65}")
    print("  TỔNG KẾT")
    print(f"{'='*65}")
    print(f"  Accuracy  : {correct/total:.2%}  ({correct}/{total})")
    print(f"  CER TB    : {avg_cer:.2%}")
    print(f"  Confidence: {avg_conf:.2%}")
    cal_gap = correct/total - avg_conf
    if cal_gap < -0.2:
        print(f"  ⚠️  Overconfident: model tự tin hơn thực tế {abs(cal_gap):.0%}")
    print()

if __name__ == '__main__':
    main()
