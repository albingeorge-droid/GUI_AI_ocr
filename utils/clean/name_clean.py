# utils/clean/name_clean.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Patterns like: S/o, W/o, C/o, H/o, D/o (and S/0 variant), case-insensitive
_PARENTAGE_PATTERN = re.compile(
    r"\b(?:S/0|S/o|W/o|C/o|H/o|D/o)\b",
    flags=re.IGNORECASE,
)


def _clean_single_name(raw: Any) -> Optional[str]:
    """
    Remove parentage markers (S/o, W/o, C/o, H/o, D/o, S/0) and everything after them.
    Example:
        'Sh. Amit S/o Sh. Bhagwat Sarup' -> 'Sh. Amit'
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    m = _PARENTAGE_PATTERN.search(s)
    if m:
        # Keep everything before the parentage term
        s = s[: m.start()]

    # Strip trailing punctuation and whitespace
    s = s.strip(" ,;-")

    # Collapse multiple internal spaces
    s = re.sub(r"\s+", " ", s)

    if not s:
        return None

    return s


def clean_applicant_name_field(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean features['applicant_name'] in-place using the parentage rules.
    """
    value = features.get("applicant_name")
    cleaned = _clean_single_name(value)
    if cleaned is not None:
        features["applicant_name"] = cleaned
    return features
