import numpy as np
from typing import Dict, List, Optional

from .find_label import find_label
from .find_value import find_value


def kie(ocr_blocks: List[Dict], img: np.ndarray = None) -> Dict:

    result = {
        "title":                          {},
        "validate_title":                 {},
        "main_title":                     {},
        "kinh_gui":                       {},
        "ho_chu_dem_va_ten":              {},
        "ngay_thang_nam_sinh":            {},
        "gioi_tinh":                      {},
        "so_dinh_dan_ca_nhan":            {},
        "so_dien_thoai_lien_he":          {},
        "email":                          {},
        "ho_chu_dem_va_ten_chu_ho":       {},
        "moi_quan_he_voi_chu_ho":         {},
        "so_dinh_dan_ca_nhan_cua_chu_ho": {},
        "noi_dung_de_nghi":               {},
    }

    if not ocr_blocks:
        return result

    img_h = img.shape[0] if img is not None else 1000
    seen_fields = set()

    for block in ocr_blocks:
        label = find_label(block['text'])
        if not label or label in seen_fields:
            continue

        if label in ('title', 'validate_title', 'main_title'):
            value_block = {**block}
        else:
            value_block = find_value(block, ocr_blocks, label, img_h)

        if not value_block:
            continue

        print(f"  [KIE] {label}: '{value_block.get('text', '')}'")

        if label in result:
            result[label] = {
                'text'      : value_block.get('text', ''),
                'confidence': value_block.get('confidence', 0.0),
                'bbox'      : value_block.get('bbox', []),
                'center_y'  : value_block.get('center_y', 0.0),
                'x_left'    : value_block.get('x_left', 0.0),
            }
            seen_fields.add(label)

    return result
