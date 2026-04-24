from typing import Dict, List
from .keyword_matcher import match_keyword
from .spatial_lookup  import find_value
from .validator       import validate_and_clean
from .block_merger    import merge_blocks


def extract_ct01(ocr_blocks: List[Dict]) -> Dict:
    result = {
        "title":                          None,
        "kinh_gui":                       None,
        "ho_chu_dem_va_ten":              None,
        "ngay_thang_nam_sinh":            None,
        "gioi_tinh":                      None,
        "so_dinh_dan_ca_nhan":            None,
        "so_dien_thoai":                  None,
        "email":                          None,
        "ho_chu_dem_va_ten_chu_ho":       None,
        "moi_quan_he_voi_chu_ho":         None,
        "so_dinh_dan_ca_nhan_cua_chu_ho": None,
        "noi_dung_de_nghi":               None,
        "thanh_vien_cung_thay_doi":       [],
    }

    # Gộp block bị break trước khi chạy KIE
    merged_blocks = merge_blocks(ocr_blocks)
    print(f"KIE merge from {len(ocr_blocks)} blocks to {len(merged_blocks)} blocks.")
    
    # seen_fields = set()

    # for block in merged_blocks:
    #     field = match_keyword(block['text'])
    #     if not field or field in seen_fields:
    #         continue

    #     # Lớp 2: tìm value tương ứng
    #     # Đặc biệt: title không có label riêng — chính block text là value
    #     if field == "title":
    #         raw_value = block['text']
    #     else:
    #         raw_value = find_value(block, ocr_blocks)
    #     print(f"  [KIE L1+L2] '{block['text']}' → {field}: '{raw_value}'")

    #     # Lớp 3: validate & clean
    #     cleaned = validate_and_clean(field, raw_value or "")
    #     print(f"  [KIE L3]    {field}: '{raw_value}' → '{cleaned}'")

    #     if field in result:
    #         result[field] = cleaned
    #         seen_fields.add(field)

    return result
