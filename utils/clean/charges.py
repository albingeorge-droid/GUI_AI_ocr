# utils/clean/charges.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Keys we want to clean in the features dict
_CHARGE_KEYS = (
    "conversion_charges",
    "total_external_development_charges",
)

# Remove "Rs", "Rs.", "₹" (case-insensitive)
_CURRENCY_PAT = re.compile(r"(rs\.?|₹)", flags=re.IGNORECASE)
# Remove trailing "/-"
_SLASH_DASH_PAT = re.compile(r"/-\s*$")


def _clean_single_amount(raw: Any) -> Optional[str]:
    """
    Clean one monetary string.
    Example:
        "Rs. 5,59,120/-"  -> "559120"
        "Rs. 1,37,340/-"  -> "137340"
    """
    if raw is None:
        return None

    # ✅ use strip(), Python has no trim()
    s = str(raw).strip()
    if not s:
        return None

    # Remove currency symbol and Rs
    s = _CURRENCY_PAT.sub("", s)

    # Remove "/-" at end
    s = _SLASH_DASH_PAT.sub("", s)

    # Remove commas and extra spaces
    s = s.replace(",", "").strip()

    # Keep only digits and optional decimal dot
    s = re.sub(r"[^0-9.]", "", s)

    if not s:
        return None

    return s


def clean_charge_fields(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean monetary fields in the features dict:
      - remove 'Rs.', 'Rs', '₹', '/-'
      - normalize to a digits-only (or digits + '.') string

    ❌ We no longer add *_num numeric helper fields – only overwrite the
       original strings with cleaned values.
    """
    # (optional) work on a shallow copy so caller's dict isn't mutated unexpectedly
    out = dict(features or {})

    for key in _CHARGE_KEYS:
        raw = out.get(key)
        cleaned = _clean_single_amount(raw)
        if cleaned is not None:
            out[key] = cleaned

    return out
