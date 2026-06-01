import cv2
import numpy as np

BLUR_GOOD = 800.0     # [benchmark: calibrated]
BLUR_MEDIUM = 300.0   # [benchmark: calibrated]


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_quality(gray: np.ndarray) -> dict:
    score = blur_score(gray)
    if score >= BLUR_GOOD:
        tier = "good"
    elif score >= BLUR_MEDIUM:
        tier = "medium"
    else:
        tier = "poor"
    return {"tier": tier, "blur_score": round(score, 1)}
