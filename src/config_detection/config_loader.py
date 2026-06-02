import copy
import json
import os
from typing import Optional

import yaml
from jsonschema import Draft7Validator

# Thư mục configs/ (gốc project) — schema nằm cùng cây với templates.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "configs")
_SCHEMA_PATH = os.path.join(_CONFIGS_DIR, "schema", "ct01_config.schema.json")


class ConfigError(Exception):
    """Config sai schema hoặc sai ngữ nghĩa."""


def _load_schema() -> dict:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> dict:
    """Đọc YAML config, validate theo JSON Schema, trả dict. Raise ConfigError nếu sai."""
    if not os.path.exists(path):
        raise ConfigError(f"Config không tồn tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML không parse được ({path}): {e}") from e

    if not isinstance(config, dict):
        raise ConfigError(f"Config phải là mapping (dict), nhận: {type(config).__name__}")

    # Validate theo JSON Schema — gom mọi lỗi, in đường dẫn field rõ ràng.
    validator = Draft7Validator(_load_schema())
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            loc = ".".join(str(p) for p in e.path) or "<root>"
            msgs.append(f"  - [{loc}] {e.message}")
        raise ConfigError(
            f"Config sai schema ({path}):\n" + "\n".join(msgs)
        )

    _check_semantics(config, path)
    return config


def _check_semantics(config: dict, path: str) -> None:
    """Các ràng buộc schema không bắt được: ROI không tràn biên [0,1]."""
    for name, field in config["fields"].items():
        roi = field["roi_norm"]
        if roi["x"] + roi["w"] > 1.0 + 1e-9:
            raise ConfigError(
                f"Config {path}: field '{name}' ROI tràn biên phải "
                f"(x={roi['x']} + w={roi['w']} = {roi['x'] + roi['w']:.3f} > 1)."
            )
        if roi["y"] + roi["h"] > 1.0 + 1e-9:
            raise ConfigError(
                f"Config {path}: field '{name}' ROI tràn biên dưới "
                f"(y={roi['y']} + h={roi['h']} = {roi['y'] + roi['h']:.3f} > 1)."
            )


def apply_quality_overrides(config: dict, quality: Optional[str]) -> dict:
    """
    Trả bản copy config với padding_x/padding_y của MỌI field đã nhân padding_scale
    tương ứng mức quality ('good'/'medium'/'poor'). Không có override → trả nguyên.
    """
    overrides = config.get("quality_overrides", {})
    rule = overrides.get(quality) if quality else None
    if not rule or "padding_scale" not in rule:
        return config

    scale = rule["padding_scale"]
    out = copy.deepcopy(config)
    for field in out["fields"].values():
        field["padding_x"] = int(round(field.get("padding_x", 0) * scale))
        field["padding_y"] = int(round(field.get("padding_y", 0) * scale))
    return out
