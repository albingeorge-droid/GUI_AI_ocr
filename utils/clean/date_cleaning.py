# utils/clean/date_cleaning.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

# Formats we expect to see from OCR or CSV
_DATE_FORMATS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%y",
    "%d-%m-%y",
]


def _normalize_date_string(raw: Any) -> Optional[str]:
    """
    Normalize a date-like string into DD/MM/YYYY.
    If parsing fails, returns the original string.
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    # Unify some separators
    s = s.replace(".", "/")

    # Try known formats
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue

    # If there is a time part, try only the first token as date
    if " " in s:
        first = s.split(" ", 1)[0]
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                dt = datetime.strptime(first, fmt)
                return dt.strftime("%d/%m/%Y")
            except ValueError:
                continue

    # Fallback: give back the original string if we cannot parse
    return s


def clean_clu_permission_date_field(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    In-place cleaning of features['clu_permission_date'] to DD/MM/YYYY.
    """
    value = features.get("clu_permission_date")
    if value is None:
        return features

    cleaned = _normalize_date_string(value)
    if cleaned is not None:
        features["clu_permission_date"] = cleaned

    return features
