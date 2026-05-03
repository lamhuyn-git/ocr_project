import os
import sys

from more_itertools import extract

# Thêm thư mục src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recognition.engine import engine_pipeline
from preprocess import preprocess_pipeline

if __name__ == '__main__':
    # IMAGE_DIR  = 'image_test/phone_good'
    # EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    # image_files = sorted([
    #     f for f in os.listdir(IMAGE_DIR)
    #     if os.path.splitext(f)[1].lower() in EXTENSIONS
    # ])

    # for filename in image_files:
    #     img_path = os.path.join(IMAGE_DIR, filename)
    #     img = preprocess_pipeline(img_path)
    #     ocr_results = engine_pipeline(img, img_path=img_path)
    
    img_path = 'image_test/scan/scan_001.jpg'
    img = preprocess_pipeline(img_path)
    ocr_results = engine_pipeline(img, img_path=img_path)
    print("\nOCR Results:", ocr_results)