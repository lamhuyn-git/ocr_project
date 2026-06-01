import cv2
import numpy as np

# src benchmark: src/evaluation/alignment-benchmark/
N_FEATURES = 3000     # [default] không quét; 3000 chạy tốt suốt benchmark
LOWE = 0.75           # [benchmark: swept] param-sweep.py
RANSAC_REPROJ = 5.0   # [benchmark: swept] param-sweep.py
MIN_MATCHES = 15      # [default] sàn an toàn; inlier ảnh thật tệ nhất=96 (smoke-test-aligner.py)


def create_orb(n_features: int = N_FEATURES) -> cv2.ORB:
    return cv2.ORB_create(nfeatures=n_features)


def detect(orb: cv2.ORB, gray: np.ndarray):
    return orb.detectAndCompute(gray, None)


def register(orb, bf, ref_kp, ref_des, src_gray, lowe: float = LOWE, ransac: float = RANSAC_REPROJ, min_matches: int = MIN_MATCHES):
    # Tìm keypoints và descriptors của ảnh nguồn
    src_kp, src_des = detect(orb, src_gray)
    if src_des is None or len(src_kp) < 4:
        return None
    
    # Match descriptors giữa ảnh nguồn và ảnh tham chiếu
    # Ứng vs 1 điểm trong ảnh nguồn có thể match với 2 điểm trong ảnh tham chiếu (knnMatch với k=2) tốt nhất
    raw = bf.knnMatch(src_des, ref_des, k=2)

    # Giữ lại các cặp match tốt theo tỉ lệ Lowe
    # Nếu khoảng cách của match tốt nhất (m) nhỏ hơn lowe * khoảng cách của match thứ hai (n), thì coi m là một match tốt.
    good = [m for m, n in (p for p in raw if len(p) == 2)
            if m.distance < lowe * n.distance]
    if len(good) < min_matches:
        return None

    ps = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)    # ps = [[[412, 88]], [[105, 240]], ... ]
    pr = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)    # pr = [[[430, 95]], [[120, 250]], ... ]
    H, mask = cv2.findHomography(ps, pr, cv2.RANSAC, ransac)
    if H is None or mask is None:
        return None

    # đánh giá chất lượng của H bằng số lượng inliers và reprojection error trên inliers
    inl = mask.ravel().astype(bool)
    n_inliers = int(inl.sum())
    if n_inliers < min_matches:
        return None

    # Reprojection error trung bình trên inliers (đo độ khít alignment)
    proj = cv2.perspectiveTransform(ps, H).reshape(-1, 2) # Tọa độ của các keypoints trong ảnh nguồn sau khi áp H
    target = pr.reshape(-1, 2)
    reproj_error = float(np.linalg.norm(proj[inl] - target[inl], axis=1).mean())

    return {
        "H": H,
        "n_matches": len(good),
        "n_inliers": n_inliers,
        "reproj_error": reproj_error,
    }
