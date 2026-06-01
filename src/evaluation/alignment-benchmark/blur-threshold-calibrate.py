"""
Hiệu chỉnh ngưỡng blur cho quality_estimator (variance-of-Laplacian) trên test_image/ thật.
Ngưỡng variance-of-Laplacian PHỤ THUỘC độ phân giải + nội dung (PyImageSearch/Pech-Pacheco 2000)
→ con số 100/150/60 mượn ngoài không dùng thẳng được, phải đo trên CT01.

In: phân bố blur_score (min/median/max + percentile), histogram thô, vài mẫu thấp/cao nhất
→ từ chỗ phân bố tách nhóm mà đặt BLUR_GOOD / BLUR_MEDIUM.
Chạy: .venv/bin/python src/evaluation/alignment-benchmark/blur-threshold-calibrate.py
"""
import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.alignment.quality_estimator import blur_score  # noqa: E402

SRC_DIR = str(ROOT / "test_image")


def main():
    paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    rows = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows.append((os.path.basename(p), blur_score(gray)))

    scores = np.array([s for _, s in rows])
    n = len(scores)
    print(f"images: {n}")
    print(f"blur_score  min={scores.min():.1f}  median={np.median(scores):.1f}  "
          f"mean={scores.mean():.1f}  max={scores.max():.1f}")
    print("percentiles:")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{q:>2} = {np.percentile(scores, q):.1f}")

    # Histogram thô theo log-bin (blur_score trải rộng nhiều bậc).
    print("\nhistogram (count theo khoảng):")
    edges = [0, 50, 100, 150, 200, 300, 500, 800, 1200, 2000, 1e9]
    labels = ["<50", "50-100", "100-150", "150-200", "200-300",
              "300-500", "500-800", "800-1200", "1200-2000", ">2000"]
    hist, _ = np.histogram(scores, bins=edges)
    for lab, c in zip(labels, hist):
        bar = "#" * int(c)
        print(f"  {lab:>10}: {c:>2} {bar}")

    rows.sort(key=lambda r: r[1])
    print("\n10 ảnh BLUR NHẤT (score thấp):")
    for name, s in rows[:10]:
        print(f"  {s:>8.1f}  {name}")
    print("\n10 ảnh NÉT NHẤT (score cao):")
    for name, s in rows[-10:]:
        print(f"  {s:>8.1f}  {name}")


if __name__ == "__main__":
    main()
