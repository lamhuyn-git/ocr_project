from typing import Dict, Optional
import numpy as np
from config_detection.roi_calculator import field_roi_pixels
from .crop_ocr import crop_roi, join_blocks, ocr_crop
from .table_extractor import extract_table

DEFAULT_CONF_THRESHOLD = 0.5

def _ocr_field_region(
    warped: np.ndarray,
    config: dict,
    name: str,
    quality: Optional[str],
    preprocess: bool,
):
    h, w = warped.shape[:2]
    box = field_roi_pixels(config, name, w, h, quality)
    crop = crop_roi(warped, box)
    blocks = ocr_crop(crop, box_offset=(box[0], box[1]), preprocess=preprocess)
    return blocks, box


def _extract_text(
    warped: np.ndarray,
    config: dict,
    name: str,
    ftype: str,
    threshold: float,
    quality: Optional[str],
    preprocess: bool,
) -> Dict:
    blocks, box = _ocr_field_region(warped, config, name, quality, preprocess)

    text, avg_conf = join_blocks(blocks)
    is_empty = not text

    # Field rỗng coi như low_confidence (không có gì để tin)
    low_conf = True if is_empty else (avg_conf < threshold)

    return {
        "type": ftype,
        "text": text,
        "confidence": avg_conf,
        "low_confidence": low_conf,
        "empty": is_empty,
        "n_blocks": len(blocks),
        "bbox": list(box),
    }


def _extract_one(
    warped: np.ndarray,
    config: dict,
    name: str,
    field: dict,
    conf_threshold: float,
    quality: Optional[str],
    preprocess: bool,
) -> Dict:
    threshold = field.get("confidence_threshold", conf_threshold)

    if field["type"] == "table":
        return extract_table(warped, config, name, threshold, quality, preprocess)

    return _extract_text(warped, config, name, field["type"], threshold, quality, preprocess)


def extract_fields(
    warped: np.ndarray,
    config: dict,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    quality: Optional[str] = None,
    preprocess: bool = True,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for name, field in config["fields"].items():
        results[name] = _extract_one(
            warped, config, name, field, conf_threshold, quality, preprocess
        )
    return results
