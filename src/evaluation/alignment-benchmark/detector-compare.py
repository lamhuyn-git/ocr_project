"""
So sánh detector cho form-align (Phase 01): ORB vs AKAZE vs BRISK (+ SIFT mốc tham chiếu).
SURF: KHÔNG có (opencv-python chuẩn loại bỏ vì patent) → bỏ.

Cùng 67 ảnh, cùng metric field-drift. Mỗi detector, mỗi ảnh:
detect+compute → knnMatch → Lowe 0.75 → findHomography RANSAC 5.0
→ chiếu landmark anchor (mép-trái, y-giữa GT box) qua H về canonical.
Báo: fail (homography None / <4 match), mean_std (độ chụm điển hình),
     worst_maxdev (ca tệ nhất → quyết padding), ms/ảnh (detect+match), avg_kp.
Binary descriptor (ORB/AKAZE/BRISK) → NORM_HAMMING; SIFT float → NORM_L2.
Chạy: .venv/bin/python src/evaluation/alignment-benchmark/detector-compare.py
"""
import os
import re
import json
import glob
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REF = str(ROOT / "real_test/template.jpg")
SRC_DIR = str(ROOT / "test_image")
GT = str(ROOT / "test_image/label_test.txt")
LOWE = 0.75
RANSAC_REPROJ = 5.0
N_FEATURES = 3000

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


def make_detectors():
    return {
        "ORB":   (cv2.ORB_create(nfeatures=N_FEATURES), cv2.NORM_HAMMING),
        "AKAZE": (cv2.AKAZE_create(), cv2.NORM_HAMMING),
        "BRISK": (cv2.BRISK_create(), cv2.NORM_HAMMING),
        "SIFT":  (cv2.SIFT_create(nfeatures=N_FEATURES), cv2.NORM_L2),
    }


def run(det, norm, ref_gray, images, gt):
    bf = cv2.BFMatcher(norm)
    kp_ref, des_ref = det.detectAndCompute(ref_gray, None)
    proj = {k: [] for k in ANCHORS}
    fail = 0
    t_total = 0.0
    n_kps = []
    for base, g in images:
        anchors = gt.get(base, {})
        t0 = time.perf_counter()
        kp, des = det.detectAndCompute(g, None)
        if des is None or len(kp) < 4:
            t_total += time.perf_counter() - t0
            fail += 1
            continue
        n_kps.append(len(kp))
        raw = bf.knnMatch(des, des_ref, k=2)
        t_total += time.perf_counter() - t0
        good = [m for m, n in (p for p in raw if len(p) == 2)
                if m.distance < LOWE * n.distance]
        if len(good) < 4:
            fail += 1
            continue
        ps = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pr = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(ps, pr, cv2.RANSAC, RANSAC_REPROJ)
        if H is None:
            fail += 1
            continue
        for fld, pt in anchors.items():
            q = cv2.perspectiveTransform(np.array([[pt]], np.float32), H).reshape(2)
            proj[fld].append(q)
    stds, maxdevs = [], []
    for arr in proj.values():
        a = np.array(arr)
        if len(a) < 2:
            continue
        c = a.mean(axis=0)
        stds.append(a.std(axis=0).mean())
        maxdevs.append(np.linalg.norm(a - c, axis=1).max())
    mean_std = np.mean(stds) if stds else float("nan")
    worst = max(maxdevs) if maxdevs else float("nan")
    ms = 1000 * t_total / max(1, len(images))
    avg_kp = int(np.mean(n_kps)) if n_kps else 0
    return fail, mean_std, worst, ms, avg_kp


def main():
    ref = cv2.cvtColor(cv2.imread(REF), cv2.COLOR_BGR2GRAY)
    gt = load_gt(GT)
    images = []
    for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg"))):
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        images.append((os.path.basename(p), g))
    print(f"loaded {len(images)} images, template {ref.shape[1]}x{ref.shape[0]}\n")
    print(f"{'detector':<8}{'fail':>5}{'mean_std':>10}{'worst_maxdev':>14}{'ms/img':>9}{'avg_kp':>8}")
    print("-" * 54)
    for name, (det, norm) in make_detectors().items():
        fail, mean_std, worst, ms, avg_kp = run(det, norm, ref, images, gt)
        print(f"{name:<8}{fail:>5}{mean_std:>10.1f}{worst:>14.1f}{ms:>9.0f}{avg_kp:>8}")


if __name__ == "__main__":
    main()
