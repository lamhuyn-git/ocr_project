from typing import Optional, Sequence, Tuple

from .config_loader import apply_quality_overrides


def pixels_to_roi_norm(
    points: Sequence[Sequence[int]],
    canvas_w: int,
    canvas_h: int,
) -> dict[str, float]:
    if not points:
        raise ValueError("points cannot be empty")
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError(f"Invalid canvas size: {canvas_w}x{canvas_h}")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return {
        "x": x1 / canvas_w,
        "y": y1 / canvas_h,
        "w": (x2 - x1) / canvas_w,
        "h": (y2 - y1) / canvas_h,
    }


def roi_norm_to_pixels(
    roi: dict,
    padding_x: int,
    padding_y: int,
    canvas_w: int,
    canvas_h: int,
) -> Tuple[int, int, int, int]:
    x1 = roi["x"] * canvas_w - padding_x
    y1 = roi["y"] * canvas_h - padding_y
    x2 = (roi["x"] + roi["w"]) * canvas_w + padding_x
    y2 = (roi["y"] + roi["h"]) * canvas_h + padding_y

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(canvas_w, int(round(x2)))
    y2 = min(canvas_h, int(round(y2)))
    return x1, y1, x2, y2


def field_roi_pixels(
    config: dict,
    field_name: str,
    canvas_w: int,
    canvas_h: int,
    quality: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    cfg = apply_quality_overrides(config, quality)
    if field_name not in cfg["fields"]:
        raise KeyError(f"Field '{field_name}' not found in config.")
    field = cfg["fields"][field_name]
    return roi_norm_to_pixels(
        field["roi_norm"],
        field.get("padding_x", 0),
        field.get("padding_y", 0),
        canvas_w,
        canvas_h,
    )
