"""
ocr_engine.py — Nhận diện chữ trong ảnh dùng PaddleOCR
PaddleOCR hỗ trợ tiếng Việt (PP-OCRv4) với độ chính xác cao
"""

import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict


# Biến toàn cục lưu instance OCR (tránh tạo lại nhiều lần)
_ocr_instance = None


def get_ocr_instance() -> PaddleOCR:
    """
    Lấy instance OCR theo pattern Singleton
    (Chỉ khởi tạo 1 lần, dùng lại cho các lần gọi sau)
    """
    global _ocr_instance
    if _ocr_instance is None:
        print("🔄 Đang khởi tạo PaddleOCR...")
        print("   (Lần đầu sẽ tải model ~100MB — vui lòng chờ)")
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,   # Phát hiện chữ bị xoay
            lang='vi',            # Ngôn ngữ tiếng Việt
            use_gpu=False,        # Đổi True nếu có GPU NVIDIA
            show_log=False,       # Tắt log không cần thiết
        )
        print("✅ PaddleOCR sẵn sàng!\n")
    return _ocr_instance


def run_ocr(img: np.ndarray) -> List[Dict]:
    """
    Chạy OCR trên ảnh đã tiền xử lý

    Args:
        img: Ảnh numpy array (BGR format từ OpenCV)

    Returns:
        Danh sách dict, mỗi dict là một dòng text:
        {
            'text': str,          — Nội dung chữ
            'confidence': float,  — Độ tin cậy (0.0 → 1.0)
            'bbox': list,         — Tọa độ 4 góc hộp bao
            'center_y': float,    — Vị trí Y trung tâm
            'x_left': float,      — Cạnh trái nhất
        }
    """
    print("\n" + "="*50)
    print("🔍 BẮT ĐẦU OCR (NHẬN DIỆN CHỮ)")
    print("="*50)

    ocr = get_ocr_instance()

    # Chạy OCR — trả về kết quả thô
    raw_results = ocr.ocr(img, cls=True)

    if not raw_results or raw_results[0] is None:
        print("⚠️  Không tìm thấy chữ nào trong ảnh!")
        return []

    # Chuyển đổi kết quả thô thành định dạng dễ dùng
    parsed = []
    for line in raw_results[0]:
        bbox = line[0]            # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]         # Nội dung chữ
        confidence = line[1][1]   # Điểm tin cậy

        # Tính tọa độ Y trung tâm (để sắp xếp theo dòng)
        center_y = sum(pt[1] for pt in bbox) / 4
        x_left = min(pt[0] for pt in bbox)

        if text.strip():  # Bỏ qua dòng trống
            parsed.append({
                'text': text.strip(),
                'confidence': round(confidence, 4),
                'bbox': bbox,
                'center_y': center_y,
                'x_left': x_left,
            })

    # Sắp xếp từ trên xuống dưới, trái sang phải
    # (nhóm các dòng có center_y gần nhau vào cùng 1 "hàng")
    parsed.sort(key=lambda x: (round(x['center_y'] / 15) * 15, x['x_left']))

    # In kết quả
    print(f"📝 Tìm thấy {len(parsed)} dòng text:\n")
    for i, item in enumerate(parsed):
        conf_icon = "✅" if item['confidence'] > 0.85 else "⚠️ "
        print(f"  {conf_icon} [{i+1:02d}] ({item['confidence']:.1%}) {item['text']}")

    print()
    return parsed


def filter_by_confidence(ocr_results: List[Dict], min_confidence: float = 0.70) -> List[Dict]:
    """
    Lọc bỏ các dòng OCR có độ tin cậy thấp

    Args:
        ocr_results: Kết quả từ run_ocr()
        min_confidence: Ngưỡng tối thiểu (mặc định 70%)

    Returns:
        Danh sách đã lọc
    """
    filtered = [r for r in ocr_results if r['confidence'] >= min_confidence]
    removed = len(ocr_results) - len(filtered)
    if removed > 0:
        print(f"🔽 Đã lọc bỏ {removed} dòng có confidence < {min_confidence:.0%}")
    return filtered


def get_text_lines(ocr_results: List[Dict]) -> List[str]:
    """Lấy danh sách dòng text thuần (không kèm metadata)"""
    return [r['text'] for r in ocr_results]
