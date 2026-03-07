"""
main.py — Pipeline OCR chính
=============================
Chạy lệnh: python main.py                    # Chế độ bình thường
Hoặc:      python main.py --fast            # Chế độ nhanh (bỏ qua denoise & deskew)
Hoặc:      python main.py images/file.jpg   # Chỉ định ảnh cụ thể
"""

import os
import sys

# Thêm thư mục src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocr_engine import engine_pipeline
from preprocess import preprocess_pipeline

if __name__ == '__main__':
    # Dùng ảnh mặc định hoặc nhận từ tham số dòng lệnh
    IMAGE_PATH = 'images/happy_case.png'

    img = preprocess_pipeline(IMAGE_PATH)
    
    engine_pipeline(img)
