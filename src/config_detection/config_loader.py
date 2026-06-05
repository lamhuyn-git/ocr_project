"""
config_loader.py — Nạp + kiểm tra file config YAML của biểu mẫu.

  - load_config(path)            : đọc YAML -> validate -> trả dict config.
  - apply_quality_overrides(...) : scale padding theo chất lượng ảnh (trả bản copy).

Mọi lỗi config đều ném ConfigError với thông báo chỉ rõ sai ở đâu.
"""
import copy
import json
import os
from typing import Optional

import yaml
from jsonschema import Draft7Validator

# Tìm đường dẫn tới configs/schema/ct01_config.schema.json dựa theo vị trí file này
# (KHÔNG dùng đường dẫn tương đối với cwd để chạy được từ bất cứ thư mục nào).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../src/config_detection
_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "configs")
_SCHEMA_PATH = os.path.join(_CONFIGS_DIR, "schema", "ct01_config.schema.json")


class ConfigError(Exception):
    """Ném ra khi config sai schema hoặc sai ngữ nghĩa."""


def _load_schema() -> dict:
    """Đọc file JSON Schema dùng để validate config."""
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> dict:
    """
    Đọc 1 file YAML config, kiểm tra hợp lệ, trả về dict.

    Các bước: tồn tại? -> parse YAML -> phải là dict -> đúng schema -> đúng ngữ nghĩa.
    Sai ở bước nào thì ném ConfigError với thông báo tương ứng.
    """
    # Bước 1: file có tồn tại không
    if not os.path.exists(path):
        raise ConfigError(f"Config không tồn tại: {path}")

    # Bước 2: đọc + parse YAML
    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML không parse được ({path}): {e}") from e

    # Bước 3: nội dung phải là 1 mapping (dict), không phải list/số/chuỗi
    if not isinstance(config, dict):
        raise ConfigError(
            f"Config phải là mapping (dict), nhận: {type(config).__name__}"
        )

    # Bước 4: validate theo JSON Schema (kiểu field, roi_norm trong [0,1], ...)
    validator = Draft7Validator(_load_schema())
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
    if errors:
        # Gom TẤT CẢ lỗi lại, mỗi lỗi 1 dòng + chỉ rõ field nào
        messages = []
        for e in errors:
            location = ".".join(str(p) for p in e.path) or "<root>"
            messages.append(f"  - [{location}] {e.message}")
        raise ConfigError(f"Config sai schema ({path}):\n" + "\n".join(messages))

    # Bước 5: kiểm tra ngữ nghĩa mà schema không bắt được
    _check_semantics(config, path)
    return config


def _check_semantics(config: dict, path: str) -> None:
    """
    Kiểm tra ROI không tràn ra ngoài ảnh: x + w <= 1 và y + h <= 1.
    (1e-9 là dung sai nhỏ để bỏ qua sai số làm tròn số thực.)
    """
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
    # TODO(tạm): tắt nhân padding theo quality để calibrate lại ROI 53/2025.
    #   Đang dùng padding gốc trong config, không scale theo quality.
    #   Bật lại bằng cách bỏ `return config` dưới đây + uncomment khối bên dưới.
    return config

    # if quality is None:
    #     return config

    # overrides = config.get("quality_overrides", {})

    # if quality not in overrides:
    #     return config

    # rule = overrides[quality]

    # if "padding_scale" not in rule:
    #     return config

    # scale = rule["padding_scale"]

    # new_config = copy.deepcopy(config)

    # for field in new_config["fields"].values():
    #     old_x = field.get("padding_x", 0)
    #     old_y = field.get("padding_y", 0)
    #     field["padding_x"] = int(round(old_x * scale))
    #     field["padding_y"] = int(round(old_y * scale))

    # return new_config
