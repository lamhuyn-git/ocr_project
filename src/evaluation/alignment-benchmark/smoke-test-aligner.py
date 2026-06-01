"""
Smoke test module src/alignment trên toàn test_image/ — xác nhận module không hồi quy so benchmark.
Báo: phân bố method, phân bố rot, thống kê reproj_error/inliers, fail count.
Lưu vài overlay blend (warped 50% + reference 50%) ra outputs/alignment_debug/ để mắt thường kiểm.
Chạy: .venv/bin/python src/evaluation/alignment-benchmark/smoke-test-aligner.py
"""
import os
import glob
import sys
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.alignment import align_form, CANONICAL_W, CANONICAL_H  # noqa: E402

SRC_DIR = str(ROOT / "test_image")
REF = str(ROOT / "assets/ct01_reference.jpg")
DEBUG_DIR = str(ROOT / "outputs/alignment_debug")
N_OVERLAY = 6   # số ảnh lưu overlay (rải đều)


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    ref = cv2.imread(REF)
    paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    methods, rots = Counter(), Counter()
    reprojs, inliers = [], []
    crashed = []
    save_every = max(1, len(paths) // N_OVERLAY)

    for i, p in enumerate(paths):
        base = os.path.basename(p)
        img = cv2.imread(p)
        try:
            warped, meta = align_form(img)
        except Exception as e:  # smoke test: align_form không được phép crash
            crashed.append((base, str(e)))
            continue
        assert warped.shape[1] == CANONICAL_W and warped.shape[0] == CANONICAL_H, base
        methods[meta["method"]] += 1
        rots[meta["rot"]] += 1
        if meta["reproj_error"] is not None:
            reprojs.append(meta["reproj_error"])
            inliers.append(meta["n_inliers"])
        if i % save_every == 0:
            blend = cv2.addWeighted(ref, 0.5, warped, 0.5, 0)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"blend_{base}"), blend)

    n = len(paths)
    print(f"images: {n}   crashed: {len(crashed)} {crashed if crashed else ''}")
    print(f"method: {dict(methods)}")
    print(f"rot:    {dict(rots)}")
    if reprojs:
        r = np.array(reprojs)
        inl = np.array(inliers)
        print(f"reproj_error px: mean={r.mean():.2f} median={np.median(r):.2f} max={r.max():.2f}")
        print(f"inliers:         mean={inl.mean():.0f} min={inl.min()} max={inl.max()}")
    success = methods.get("orb", 0)
    print(f"alignment success (method=orb): {success}/{n} = {success/n:.1%}")
    print(f"overlay blends saved to {DEBUG_DIR}/")


if __name__ == "__main__":
    main()
