import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocr.engine   import run_ocr_pipeline
from ocr.visualize import draw_bounding_boxes
from preprocess   import preprocess_pipeline
from kie.kie      import kie

if __name__ == '__main__':
    img_path = 'image_test/scan/scan_002.jpg'
    base     = os.path.splitext(os.path.basename(img_path))[0]

    print("PREPROCESSING")
    img = preprocess_pipeline(img_path)

    print("RUN OCR")
    img, ocr_blocks = run_ocr_pipeline(img)

    os.makedirs('outputs/test_results', exist_ok=True)
    save_img = draw_bounding_boxes(img.copy(), ocr_blocks)
    cv2.imwrite(f'outputs/test_results/{base}_ocr_result.jpg', save_img)

    print("RUN KIE")
    kie_result = kie(ocr_blocks, img=img)
    print('\n[KIE Result]')
    for field, value in kie_result.items():
        if value:
            print(f"  {field}: {value}")
