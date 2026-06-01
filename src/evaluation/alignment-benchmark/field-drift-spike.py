"""
Benchmark spike Phase 01 — cổng quyết định ORB (metric GẮN ROI, không chỉ reproj).

Ý tưởng: với mỗi field có nhãn in cố định (1.Họ, 2.Ngày sinh, 4.CCCD, 5.SĐT, 10.Nội dung),
lấy landmark = (mép trái, y giữa) của GT box (mép trái = vị trí nhãn in, BỀN với độ dài value).
Chiếu landmark qua phép biến đổi về canonical, đo ĐỘ TẢN (std/max) vị trí cùng-field giữa
67 ảnh. Tản nhỏ => 1 config ROI + padding nhỏ phủ mọi ảnh => alignment đạt ROI-grade.

So 2 method:
  - ORB   : ORB+RANSAC homography (đề xuất)
  - RESIZE: baseline rẻ nhất — chỉ scale ảnh về canonical, KHÔNG feature align.
ORB phải thắng RESIZE rõ rệt mới đáng dùng.

In thẳng số drift (px), KHÔNG ngưỡng pass/fail tự chế — người đọc tự xét theo ngân sách padding.
Chạy: .venv/bin/python src/evaluation/alignment-benchmark/field-drift-spike.py
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
LOWE = 0.75
RANSAC_REPROJ = 5.0
N_FEATURES = 3000

# Nhãn in cố định (có ở MỌI form + template). Regex khớp phần đầu transcription.
ANCHORS = {
    "f1_hoten":   re.compile(r"^\s*1\s*[\.\)]?\s*H[oọ]"),
    "f2_ngaysinh":re.compile(r"^\s*2\s*[\.\)]?\s*Ng[aà]y"),
    "f4_cccd":    re.compile(r"^\s*4\s*[\.\)]?\s*S[oố]\s*đ[ịi]nh"),
    "f5_sdt":     re.compile(r"^\s*5\s*[\.\)]?\s*S[oố]\s*đi[eệ]n"),
    "f10_noidung":re.compile(r"^\s*10\s*[\.\)]?\s*N[oộ]i\s*dung"),
}


def left_mid(points):
    """Landmark bền với độ dài value: x = mép trái, y = trung bình (vị trí nhãn in)."""
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
            anchors = {}
            for it in items:
                txt = it.get("transcription", "")
                for key, rx in ANCHORS.items():
                    if key not in anchors and rx.search(txt):
                        anchors[key] = left_mid(it["points"])
            gt[os.path.basename(name)] = anchors
    return gt


def orb_homography(orb, bf, kp_ref, des_ref, src_gray):
    kp_src, des_src = orb.detectAndCompute(src_gray, None)
    if des_src is None or len(kp_src) < 4:
        return None
    raw = bf.knnMatch(des_src, des_ref, k=2)
    good = [m for m, n in raw if m.distance < LOWE * n.distance]
    if len(good) < 4:
        return None
    ps = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pr = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(ps, pr, cv2.RANSAC, RANSAC_REPROJ)
    return H


def project(H, pt):
    p = np.array([[pt]], dtype=np.float32)
    return cv2.perspectiveTransform(p, H).reshape(2)


def summarize(label, proj_by_field):
    print(f"\n=== {label}: độ tản vị trí field trong canonical (px) ===")
    print(f"{'field':<14}{'n':>4}{'std_x':>8}{'std_y':>8}{'max_dev':>9}")
    print("-" * 43)
    allmax = []
    for fld, arr in proj_by_field.items():
        a = np.array(arr)
        if len(a) < 2:
            print(f"{fld:<14}{len(a):>4}{'n/a':>8}{'n/a':>8}{'n/a':>9}")
            continue
        c = a.mean(axis=0)
        sx, sy = a.std(axis=0)
        maxdev = np.linalg.norm(a - c, axis=1).max()
        allmax.append(maxdev)
        print(f"{fld:<14}{len(a):>4}{sx:>8.1f}{sy:>8.1f}{maxdev:>9.1f}")
    if allmax:
        print(f"{'WORST max_dev across fields:':<30}{max(allmax):>8.1f} px")


def main():
    ref_bgr = cv2.imread(REF)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    h_ref, w_ref = ref_gray.shape
    print(f"template (canonical): {w_ref}x{h_ref}  aspect={h_ref/w_ref:.4f} (A4={1.414:.3f})")

    gt = load_gt(GT)
    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)

    proj_orb = {k: [] for k in ANCHORS}
    proj_resize = {k: [] for k in ANCHORS}
    missing_anchor = {k: 0 for k in ANCHORS}
    fail_orb = []

    for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg"))):
        base = os.path.basename(p)
        anchors = gt.get(base, {})
        src_bgr = cv2.imread(p)
        src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        hs, ws = src_gray.shape

        H = orb_homography(orb, bf, kp_ref, des_ref, src_gray)
        if H is None:
            fail_orb.append(base)
        sx, sy = w_ref / ws, h_ref / hs   # baseline resize-only

        for fld in ANCHORS:
            if fld not in anchors:
                missing_anchor[fld] += 1
                continue
            pt = anchors[fld]
            proj_resize[fld].append([pt[0] * sx, pt[1] * sy])
            if H is not None:
                proj_orb[fld].append(project(H, pt).tolist())

    n = len(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    print(f"\nORB homography fail: {len(fail_orb)}/{n}  {fail_orb if fail_orb else ''}")
    print("anchor không tìm thấy (theo field):", missing_anchor)
    summarize("ORB", proj_orb)
    summarize("RESIZE-only (baseline)", proj_resize)


if __name__ == "__main__":
    main()
