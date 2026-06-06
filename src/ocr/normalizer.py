import re
import unicodedata


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _trim(s: str) -> str:
    return s.strip()


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_space(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _to_date(s: str) -> str:
    nums = re.findall(r"\d+", s)
    if len(nums) >= 3:
        try:
            d, m, y = int(nums[0]), int(nums[1]), int(nums[2])
            if y < 100:                       # năm 2 chữ số → 19xx/20xx
                y += 2000 if y < 50 else 1900
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                return f"{d:02d}/{m:02d}/{y:04d}"
        except (ValueError, IndexError):
            pass
    return s


_LABEL_RE = re.compile(r"^.*?\(\s*2\s*\)\s*[:：]?\s*", re.S)


def _strip_label(s: str) -> str:
    m = _LABEL_RE.match(s)
    return s[m.end():].strip() if m else s


NORMALIZERS = {
    "nfc": _nfc,
    "trim": _trim,
    "collapse_ws": _collapse_ws,
    "strip_space": _strip_space,
    "date_ddmmyyyy": _to_date,
    "strip_label": _strip_label,
}


def apply_normalizers(text: str, ops) -> str:
    if not text or not ops:
        return text
    out = text
    for op in ops:
        fn = NORMALIZERS.get(op)
        if fn is None:
            continue
        try:
            out = fn(out)
        except Exception:
            pass
    return out
