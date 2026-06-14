"""
find_label.py — text normalization helper used by validator.py.

Restored as a minimal module: the original KIE find_label module was removed in a
refactor, but validator.py still imports `normalize` from it. Only the self-contained
`normalize()` helper is kept here (no KEYWORD_MAP / fuzzy matching).
"""
import re
import unicodedata


def normalize(text: str) -> str:
    """NFC + lowercase + strip + remove trailing punctuation + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r'[:\(\)\[\]\.]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text
