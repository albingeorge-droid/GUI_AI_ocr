# utils/fext/schema.py
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# Version tag for downstream consumers / tracing
HARYANA_CLU_SCHEMA_VERSION = "haryana_clu_v1"


class HaryanaCLUPlotFeatures(BaseModel):
    """
    Pydantic model for Haryana CLU (Change of Land Use) plot features.

    All fields are kept as strings exactly as they appear in the document,
    except `terms_and_conditions`, which is a list of bullet/numbered clauses.
    """


    applicant_name: Optional[str] = None
    memo_no: Optional[str] = None
    location_controlled_area: Optional[str] = None
    tehsil: Optional[str] = None
    district: Optional[str] = None

    # NEW FIELD
    subject: Optional[str] = None

    purpose: Optional[str] = None
    granted_area: Optional[str] = None
    clu_permission_date: Optional[str] = None
    conversion_charges: Optional[str] = None
    total_external_development_charges: Optional[str] = None
    khasra_numbers: List[str] = Field(default_factory=list)
    terms_and_conditions: List[str] = Field(default_factory=list)
