# utils/clean/csv_mapping.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


PDF_NAME_COLUMN = "PDF_Names"


def _normalize_col(name: str) -> str:
    """Strip spaces from column names."""
    return name.strip()


def _normalize_key(val: Any) -> Optional[str]:
    """
    Normalize the PDF key used for matching.
    - Treat NaN / None / empty as missing.
    - Strip any folder paths.
    - Remove a trailing '.pdf' (case-insensitive) so that
      '114_CLU_ST-2326.pdf' -> '114_CLU_ST-2326'.
    """
    if val is None:
        return None

    s = str(val).strip()
    if not s:
        return None

    # If someone put a path like 'some/folder/114_CLU_ST-2326.pdf'
    s = s.replace("\\", "/").split("/")[-1]

    # Drop .pdf extension (case-insensitive)
    if s.lower().endswith(".pdf"):
        s = s[:-4]

    return s or None



def load_haryana_clean_mapping(csv_or_xlsx_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the cleaned Haryana mapping (CSV or XLSX) and return:

        { "PDF_NAME": {<row-as-dict-with-None-for-NaN>, ...}, ... }
    """
    path = Path(csv_or_xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Clean mapping file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalize column names
    df.columns = [_normalize_col(c) for c in df.columns]

    if PDF_NAME_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a '{PDF_NAME_COLUMN}' column in clean mapping file {path}"
        )

    mapping: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        key = _normalize_key(row[PDF_NAME_COLUMN])
        if not key:
            continue

        row_dict: Dict[str, Any] = {}
        for col, val in row.items():
            # Convert NaN -> None
            if pd.isna(val):
                row_dict[col] = None
            else:
                row_dict[col] = val
        mapping[key] = row_dict

    return mapping


def _pick_first_non_empty(row: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first non-empty string value for any of the candidate column names.
    """
    for c in candidates:
        if c in row and row[c] is not None:
            v = row[c]
            if isinstance(v, str):
                s = v.strip()
                if s:
                    return s
            else:
                # numbers etc.
                return str(v)
    return None


def _pick_float(row: Dict[str, Any], candidates: Iterable[str]) -> Optional[float]:
    """
    Try to parse the first non-empty candidate value as float.
    """
    v_str = _pick_first_non_empty(row, candidates)
    if v_str is None:
        return None
    try:
        return float(str(v_str))
    except Exception:
        return None


def apply_haryana_csv_overrides(
    features: Dict[str, Any],
    csv_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Given the raw 'features' dict from fext and the CSV row for this PDF,
    return a NEW dict with overrides applied:

      - applicant_name
      - location_controlled_area
      - tehsil
      - district
      - purpose
      - clu_permission_date
      - granted_area
      - lat
      - long
    """
    out = dict(features or {})

    # String fields
    overrides = {
        "applicant_name": ["Applicant Name", "Applicant_Name"],
        "location_controlled_area": ["Location/ Controlled Area", "Location/Controlled Area"],
        "tehsil": ["Tehsil"],
        "district": ["District"],
        "purpose": ["Purpose"],
        "clu_permission_date": ["CLU Permission on", "CLU Permission On"],
        "granted_area": ["granted_area_sqm"],
    }

    for feat_key, candidates in overrides.items():
        val = _pick_first_non_empty(csv_row, candidates)
        if val is not None:
            out[feat_key] = val

    # Lat / Long (numeric floats if possible)
    lat = _pick_float(csv_row, ["Lat", "Latitude", "lat", "latitude"])
    if lat is not None:
        out["lat"] = lat

    lon = _pick_float(csv_row, ["Long", "Longitude", "long", "longitude", "Lng"])
    if lon is not None:
        out["long"] = lon

    return out
