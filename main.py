import argparse
import json
import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alignment import align_form               # noqa: E402
from config_detection import load_config        # noqa: E402
from ocr.field_extractor import extract_fields  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "templates", "ct01_v1.0.yaml")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "extract")


def run_pipeline(
    image_path: str,
    config_path: str = DEFAULT_CONFIG,
    conf_threshold: float = 0.5,
    quality: str = None,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    warped, meta = align_form(img)                       # B1: warp về canonical
    config = load_config(config_path)                    # B2: nạp config
    results = extract_fields(                             # B3: crop ROI + OCR mọi field
        warped, config, conf_threshold=conf_threshold, quality=quality
    )
    return results, meta


def _print_results(results: dict) -> None:
    for name, r in results.items():
        flag = "  ⚠ LOW" if r["low_confidence"] else ""
        if r["type"] == "table":
            print(f"[{name}] table — {r['n_blocks']} block, conf={r['confidence']}{flag}")
            for row in r["rows"]:
                print(f"    · {row['text']}  ({row['confidence']})")
        else:
            print(f"[{name}] ({r['confidence']}){flag}  {r['text']!r}")


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _collect_images(path: str):
    """Trả danh sách ảnh: nếu path là thư mục → mọi ảnh trong đó; nếu là file → [file]."""
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
    return [path]


def _process_one(image_path: str, config_path: str, conf: float, quality, out_dir: str) -> bool:
    """Chạy pipeline cho 1 ảnh, lưu JSON. Trả True nếu thành công."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    try:
        results, meta = run_pipeline(image_path, config_path, conf, quality)
    except Exception as e:
        print(f"  ✗ {base}: LỖI — {e}")
        return False

    out_path = os.path.join(out_dir, f"{base}_fields.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_low = sum(1 for r in results.values() if r["low_confidence"])
    print(f"  ✓ {base}: align={meta['method']} reproj={meta['reproj_error']} "
          f"| {len(results)} field, {n_low} low-conf → {os.path.basename(out_path)}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR config-space cho ảnh form CT01 (1 ảnh hoặc cả thư mục).")
    ap.add_argument("path", help="Đường dẫn ảnh, hoặc THƯ MỤC chứa ảnh.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--conf", type=float, default=0.5, help="ngưỡng confidence")
    ap.add_argument("--quality", default=None, choices=["good", "medium", "poor"])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    images = _collect_images(args.path)
    if not images:
        print(f"Không tìm thấy ảnh trong: {args.path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Xử lý {len(images)} ảnh → {args.out_dir}\n")

    ok = 0
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]", end=" ")
        ok += _process_one(img_path, args.config, args.conf, args.quality, args.out_dir)

    print(f"\nXong: {ok}/{len(images)} ảnh thành công → {args.out_dir}")


if __name__ == "__main__":
    main()
