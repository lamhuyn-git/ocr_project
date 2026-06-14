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


def _strip_commas(s: str) -> str:
    """Bỏ dấu phẩy ngăn cách vế câu, dùng để so sánh 2 chiều khi eval."""
    return re.sub(r"\s*,\s*", " ", s).strip()


def _to_lower(s: str) -> str:
    """Lowercase toàn bộ chuỗi, dùng để so sánh 2 chiều khi eval (tránh phạt lỗi hoa/thường)."""
    return s.lower()


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


def _strip_trail_punct(s: str) -> str:
    """Xoá dấu câu thừa cuối chuỗi (.,:;!? và dấu Unicode tương đương)."""
    return re.sub(r"[\s.,;:!?。，、…]+$", "", s).strip()


# ---- email post-process ----
_DOUBLE_AT = re.compile(r"@{2,}")
_DOMAIN_FIXES = {                # OCR hay nhầm trong domain phổ biến
    "gonail": "gmail",
    "gmal":   "gmail",
    "gmall":  "gmail",
    "gmil":   "gmail",
    "gail":   "gmail",
    "hoymail": "hotmail",
    "hotmal":  "hotmail",
    "yaho":    "yahoo",
}

def _fix_email(s: str) -> str:
    """lowercase + bỏ trailing punct + sửa @@ + fix domain OCR hay nhầm."""
    out = s.strip().lower()
    # Bỏ trailing punct (dấu chấm, dấu phẩy, chữ bị cắt sau .com)
    out = re.sub(r"[.,;:!?\s]+$", "", out)
    # Sửa @@ → @
    out = _DOUBLE_AT.sub("@", out)
    # Fix domain nếu có @
    if "@" in out:
        local, domain = out.rsplit("@", 1)
        # Bỏ dấu câu thừa ở local part
        local = re.sub(r"[.,;:!?\s]+$", "", local)
        # Fix domain name (trước dấu chấm đầu tiên)
        parts = domain.split(".", 1)
        if parts[0] in _DOMAIN_FIXES:
            parts[0] = _DOMAIN_FIXES[parts[0]]
        domain = ".".join(parts)
        # Bỏ trailing thừa sau .com / .vn / v.v (vd "gmail.com..")
        domain = re.sub(r"\.(com|vn|net|org|edu)(\..*)?$", lambda m: "." + m.group(1), domain)
        out = local + "@" + domain
    return out


NORMALIZERS = {
    "nfc": _nfc,
    "trim": _trim,
    "collapse_ws": _collapse_ws,
    "strip_space": _strip_space,
    "strip_commas": _strip_commas,
    "to_lower": _to_lower,
    "date_ddmmyyyy": _to_date,
    "strip_label": _strip_label,
    "strip_trail_punct": _strip_trail_punct,
    "email_fix": _fix_email,
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
