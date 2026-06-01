"""Chạy align_form trên mọi ảnh test_alignment_*.jpg/.jpeg trong real_test/.
Lưu ảnh đã warp vào real_test/outputs/alignment/.
Chạy từ gốc project:  .venv/bin/python real_test/run_alignment.py
"""
import os
import sys
import glob
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from alignment import align_form  # noqa: E402

IN_DIR = os.path.join(ROOT, "real_test")
OUT_DIR = os.path.join(ROOT, "real_test", "outputs", "alignment")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(
        glob.glob(os.path.join(IN_DIR, "test_alignment_*.jpg"))
        + glob.glob(os.path.join(IN_DIR, "test_alignment_*.jpeg"))
    )
    print(f"found {len(paths)} images")

    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            print(f"  SKIP (unreadable): {p}")
            continue
        warped, meta = align_form(img)
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_warped.jpg"), warped)
        print(
            f"  {base}: method={meta['method']} rotate={meta['rotate']} "
            f"inliers={meta['n_inliers']} reproj={meta['reproj_error']}"
        )


if __name__ == "__main__":
    main()
