"""
preprocess.py — Tiền xử lý ảnh trước khi đưa vào OCR
"""
import os
import cv2
import numpy as np

# def fix_orientation(img: np.ndarray) -> np.ndarray:
#     h, w = img.shape[:2]

#     # Sửa xoay 90° (landscape → portrait)
#     if w > h:
#         print(f"Landscape detected ({w}x{h}px), rotating 90°...")
#         img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#         h, w = img.shape[:2]
#         print(f"Rotated to portrait: {w}x{h}px")

#     # Kiểm tra xoay 180° (ảnh ngược đầu)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     top_pixels    = np.sum(binary[:h//2, :] > 0)
#     bottom_pixels = np.sum(binary[h//2:, :] > 0)

#     if bottom_pixels > top_pixels * 1.5:
#         print(f"180° flip detected (bottom/top ratio = {bottom_pixels/top_pixels:.1f}x), rotating 180°...")
#         img = cv2.rotate(img, cv2.ROTATE_180)
#     else:
#         print(f"Orientation OK — no 180° flip needed (bottom/top ratio = {bottom_pixels/top_pixels:.1f}x)")

#     return img

def resize_image(img: np.ndarray, max_width: int = 1920) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    ratio = max_width / w
    new_h = int(h * ratio)
    img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    print(f"Resize successfully: {img.shape}")
    return img

def deskew(img: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)  # 100 → 50

    if lines is None:
        print("No straight lines detected for deskew")
        return img

    angles = []
    for line in lines[:50]:
        rho, theta = line[0]
        a = np.degrees(theta) - 90
        if abs(a) < 45:
            angles.append(a)
        elif a < -45:          # near-vertical line → convert sang skew tương đương
            angles.append(a + 90)

    if not angles:
        print("No valid angles found for deskew")
        return img

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.5:
        print("Image straight, not need rotate")
        return img

    print(f"Skew angle detected: {median_angle:.2f}°, correcting...")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    # Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Áp dụng CLAHE cho kênh L (độ sáng)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Ghép lại và chuyển về BGR
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    print("Improve clahe successfully!")
    return result  

def preprocess_pipeline(image_path: str) -> np.ndarray:
    print("=" * 45)
    print("PREPROCESSING PIPELINE")
    print("=" * 45)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    print(f"[1/6] Loaded image successfully: {img.shape[1]}x{img.shape[0]}px")

    # print("[1/5] Detecting and fixing orientation...")
    # img = fix_orientation(img)

    print("[2/6] Resizing image...")
    img = resize_image(img)

    print("[3/6] Correcting skew...")
    img = deskew(img)

    print("[4/6] Removing noise...")
    # img = denoise(img)

    print("[5/6] Enhancing contrast...")
    img = apply_clahe(img)

    name, ext = os.path.splitext(os.path.basename(image_path))
    os.makedirs('outputs/preprocessing_results', exist_ok=True)
    debug_path = f"outputs/preprocessing_results/{name}_result{ext}"
    cv2.imwrite(debug_path, img)

    print(f"[6/6] Done preprocessing! Save in  Saved in {debug_path}")
    
    return img
