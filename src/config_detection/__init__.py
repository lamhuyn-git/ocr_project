from .config_loader import load_config, apply_quality_overrides, ConfigError
from .roi_calculator import roi_norm_to_pixels, field_roi_pixels, pixels_to_roi_norm

# Lưu ý: bước trích xuất (crop+OCR) nằm ở package ocr (ocr.field_extractor.extract_fields).
# config_detection chỉ lo config + toán ROI, không kéo paddleocr vào.

__all__ = [
    "load_config",
    "apply_quality_overrides",
    "ConfigError",
    "roi_norm_to_pixels",
    "field_roi_pixels",
    "pixels_to_roi_norm",
]
