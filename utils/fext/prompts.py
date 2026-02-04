# utils/fext/prompts.py
from __future__ import annotations


def get_haryana_clu_system_prompt() -> str:
    """
    System-style instructions for extracting structured fields from a Haryana CLU document.
    """
    return (
        "You are an information extraction engine for official land-use (CLU) permissions "
        "issued in the state of Haryana, India.\n\n"
        "You will be given the raw OCR text of a single CLU permission document. "
        "Your task is to carefully read the document and extract the following fields, "
        "returning a single JSON object ONLY, with exactly these keys:\n\n"
        "  {\n"
        '    \"applicant_name\": string or null,\n'
        '    \"memo_no\": string or null,\n'
        '    \"location_controlled_area\": string or null,\n'
        '    \"tehsil\": string or null,\n'
        '    \"district\": string or null,\n'
        '    \"purpose\": string or null,\n'
        '    \"granted_area\": string or null,\n'
        '    \"clu_permission_date\": string or null,\n'
        '    \"conversion_charges\": string or null,\n'
        '    \"total_external_development_charges\": string or null,\n'
        '    \"terms_and_conditions\": [string, ...]\n'
        "  }\n\n"
        "Guidelines:\n"
        "- If a field is not explicitly mentioned or cannot be inferred with high confidence, set it to null.\n"
        "- Use the text as-is for monetary amounts (include currency, units, and formatting as seen).\n"
        "- Do NOT invent values; do NOT guess missing memo numbers or amounts.\n"
        "- \"granted_area\" should include both numeric value and units (e.g., '2.50 acres', '5000 sq. m.').\n"
        "- \"clu_permission_date\" should be exactly as printed (e.g. '23.09.2014', '23/09/2014').\n"
        "- \"terms_and_conditions\" should be a list of individual clauses or bullet points. "
        "Strip leading numbering like '(i)', '1.', '(a)' but keep the actual clause text.\n"
        "- Return ONLY valid JSON. Do NOT wrap it in markdown. Do NOT include any explanation.\n"
    )


def build_extraction_user_message(ocr_text: str, max_chars: int = 15000) -> str:
    """
    Build the text content for the user message.
    We optionally truncate very long OCR text to keep token usage under control.
    """
    if max_chars is not None and len(ocr_text) > max_chars:
        truncated = ocr_text[:max_chars]
    else:
        truncated = ocr_text

    return (
        get_haryana_clu_system_prompt()
        + "\n\n"
        "Below is the OCR text of the document. Extract the fields ONLY from this text.\n"
        "---- OCR_TEXT_START ----\n"
        f"{truncated}\n"
        "---- OCR_TEXT_END ----\n"
    )
