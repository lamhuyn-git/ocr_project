"""
Sweep độ nhạy tham số cho benchmark ORB (Phase 01).
Trả lời: kết quả field-drift có phụ thuộc việc chọn RANSAC_REPROJ=5.0 và Lowe=0.75 không?
Quét RANSAC_REPROJ ∈ {3,4,5,8} × Lowe ∈ {0.70,0.75,0.80}.
Mỗi tổ hợp đo field drift (như field-drift-spike) trên 67 ảnh → báo:
  - fail count (homography None)
  - mean std (trung bình std_x,std_y mọi field) = độ chụm điển hình
  - worst max_dev (ca tệ nhất mọi field)        = quyết định padding
ORB keypoints/descriptors detect 1 LẦN/ảnh (cache) → đổi tham số chỉ rerun match+homography.
Chạy: .venv/bin/python src/evaluation/alignment-benchmark/param-sweep.py
"""
import os
import re
import json
import glob
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REF = str(ROOT / "real_test/template.jpg")
SRC_DIR = str(ROOT / "test_image")
GT = str(ROOT / "test_image/label_test.txt")
N_FEATURES = 3000
RANSAC_VALUES = [3.0, 4.0, 5.0, 8.0]
LOWE_VALUES = [0.70, 0.75, 0.80]

ANCHORS = {
    "f1_hoten":    re.compile(r"^\s*1\s*[\.\)]?\s*H[oọ]"),
    "f2_ngaysinh": re.compile(r"^\s*2\s*[\.\)]?\s*Ng[aà]y"),
    "f4_cccd":     re.compile(r"^\s*4\s*[\.\)]?\s*S[oố]\s*đ[ịi]nh"),
    "f5_sdt":      re.compile(r"^\s*5\s*[\.\)]?\s*S[oố]\s*đi[eệ]n"),
    "f10_noidung": re.compile(r"^\s*10\s*[\.\)]?\s*N[oộ]i\s*dung"),
}


def left_mid(points):
    pts = np.array(points, dtype=np.float32)
    return np.array([pts[:, 0].min(), pts[:, 1].mean()], dtype=np.float32)


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            name, js = line.split("\t", 1)
            try:
                items = json.loads(js)
            except json.JSONDecodeError:
                continue
            a = {}
            for it in items:
                txt = it.get("transcription", "")
                for k, rx in ANCHORS.items():
                    if k not in a and rx.search(txt):
                        a[k] = left_mid(it["points"])
            gt[os.path.basename(name)] = a
    return gt


def main():
    ref = cv2.cvtColor(cv2.imread(REF), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    gt = load_gt(GT)

    # cache: mỗi ảnh -> (raw knn matches src->ref, anchor landmarks). Detect ORB 1 lần.
    cache = []
    for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg"))):
        base = os.path.basename(p)
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(g, None)
        if des is None:
            continue
        raw = bf.knnMatch(des, des_ref, k=2)
        cache.append((base, kp, raw, gt.get(base, {})))
    print(f"cached {len(cache)} images\n")

    print(f"{'RANSAC':>7}{'Lowe':>6}{'fail':>6}{'mean_std':>10}{'worst_maxdev':>14}")
    print("-" * 43)
    for ransac in RANSAC_VALUES:
        for lowe in LOWE_VALUES:
            proj = {k: [] for k in ANCHORS}
            fail = 0
            for base, kp, raw, anchors in cache:
                good = [m for m, n in raw if m.distance < lowe * n.distance]
                if len(good) < 4:
                    fail += 1
                    continue
                ps = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pr = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(ps, pr, cv2.RANSAC, ransac)
                if H is None:
                    fail += 1
                    continue
                for fld, pt in anchors.items():
                    q = cv2.perspectiveTransform(np.array([[pt]], np.float32), H).reshape(2)
                    proj[fld].append(q)
            stds, maxdevs = [], []
            for fld, arr in proj.items():
                a = np.array(arr)
                if len(a) < 2:
                    continue
                c = a.mean(axis=0)
                stds.append(a.std(axis=0).mean())
                maxdevs.append(np.linalg.norm(a - c, axis=1).max())
            mean_std = np.mean(stds) if stds else float("nan")
            worst = max(maxdevs) if maxdevs else float("nan")
            print(f"{ransac:>7.1f}{lowe:>6.2f}{fail:>6}{mean_std:>10.1f}{worst:>14.1f}")


if __name__ == "__main__":
    main()
