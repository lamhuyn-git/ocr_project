import os
import cv2
import numpy as np

from . import orb_register as orb_reg
from .quality_estimator import estimate_quality

CANONICAL_W = 1654  #Khớp chính xác A4 @ 200 DPI
CANONICAL_H = 2339  #Khớp chính xác A4 @ 200 DPI

# Nếu hướng 0° đã đạt mức này thì bỏ qua 90/180/270 (đa số ảnh chụp đúng chiều → nhanh).
GOOD_INLIERS_SHORTCIRCUIT = 80

# cv2.rotate code theo góc xoay NGƯỢC để đưa ảnh về đúng chiều khi thử.
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Cache reference (nạp 1 lần).
_REF = None
REFERENCE_PATH = os.path.join("assets", "ct01_reference.jpg")

# Thư mục lưu ảnh đã warp để debug căn chỉnh.
DEBUG_DIR = os.path.join("outputs", "alignment_debug")


def _save_debug(warped: np.ndarray, debug_name: str) -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(debug_name))[0]
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{base}_warped.jpg"), warped)

def _load_reference():
    global _REF
    if _REF is not None:
        return _REF
    ref_bgr = cv2.imread(REFERENCE_PATH)
    if ref_bgr is None:
        raise FileNotFoundError(f"Reference not found: {REFERENCE_PATH}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    orb = orb_reg.create_orb() 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) # Brute Force Matcher với khoảng cách Hamming (phù hợp với ORB)
    ref_kp, ref_des = orb_reg.detect(orb, ref_gray) #Load keypoints + descriptors của ảnh ref 1 lần để tái sử dụng suốt quá trình align.
    _REF = {"orb": orb, "bf": bf, "kp": ref_kp, "des": ref_des}
    return _REF


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img
    return cv2.rotate(img, _ROTATE_CODES[angle])


def _best_orientation(ref, gray: np.ndarray):
    best_angle, best = None, None

    for angle in (0, 90, 180, 270):
        res = orb_reg.register(ref["orb"], ref["bf"], ref["kp"], ref["des"], _rotate(gray, angle))

        if res is None:
            continue

        # res["n_inliers"] = số điểm khớp đúng; càng nhiều = càng đúng chiều.
        # Lần đầu (chưa có best) hoặc hướng này tốt hơn best cũ → cập nhật.
        is_first = best is None
        is_better = (not is_first) and res["n_inliers"] > best["n_inliers"]
        if is_first or is_better:
            best_angle = angle
            best = res

        # Lối tắt: 0° (chưa xoay) đã quá tốt → khỏi thử 90/180/270 cho nhanh.
        if angle == 0 and res["n_inliers"] >= GOOD_INLIERS_SHORTCIRCUIT:
            break

    return best_angle, best


def align_form(img: np.ndarray, debug_name: str = None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quality = estimate_quality(gray)
    ref = _load_reference()

    # Tìm hướng xoay của img
    angle, res = _best_orientation(ref, gray)
    if res is not None:
        rotated = _rotate(img, angle)
        warped = cv2.warpPerspective(rotated, res["H"], (CANONICAL_W, CANONICAL_H))
        if debug_name:
            _save_debug(warped, debug_name)
        meta = {
            "method": "orb",
            "rotate": angle,
            "n_matches": res["n_matches"],
            "n_inliers": res["n_inliers"],
            "reproj_error": round(res["reproj_error"], 2),
            "quality": quality,
        }
        return warped, meta

    warped = cv2.resize(img, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_AREA)
    if debug_name:
        _save_debug(warped, debug_name)
        
    # Fallback: ORB không đủ match (ngoài phân bố đã test) → resize thẳng, không crash.
    meta = {
        "method": "fallback_resize",
        "rotate": 0,
        "n_matches": 0,
        "n_inliers": 0,
        "reproj_error": None,
        "quality": quality,
    }
    return warped, meta
