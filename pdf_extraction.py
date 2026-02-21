import pdfplumber
import csv
import re
from typing import List, Optional

PDF_PATH = "C:\\Users\\albin\\Documents\\GitHub\\GUI_AI_ocr\\Book_1_Peshi_Register_from_01-12-2015_to_31-12-2015.pdf"
OUT_CSV  = "peshi_register.csv"

# Table structure seen in the PDF (first page)
EXPECTED_HEADERS = [
    "No.", "S.No", "Reg.No.", "IstParty", "IIndParty",
    "Type of Deed", "Address", "Value", "Stamp Paid", "Book No."
]

TABLE_SETTINGS = {
    # For bordered tables like this register, "lines" works well.
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 10,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "text_tolerance": 3,
}

def clean_cell(x: Optional[str]) -> str:
    if x is None:
        return ""
    # Normalize whitespace/newlines
    x = re.sub(r"\s+", " ", x).strip()
    return x

def looks_like_header(row: List[str]) -> bool:
    row_norm = [clean_cell(c).lower() for c in row]
    hdr_norm = [h.lower() for h in EXPECTED_HEADERS]
    # If most header tokens match, treat as header row
    matches = sum(1 for a, b in zip(row_norm, hdr_norm) if a == b)
    return matches >= 5

def normalize_row(row: List[str]) -> List[str]:
    # Ensure exactly 10 columns (pad/truncate)
    row = [clean_cell(c) for c in row]
    if len(row) < len(EXPECTED_HEADERS):
        row = row + [""] * (len(EXPECTED_HEADERS) - len(row))
    elif len(row) > len(EXPECTED_HEADERS):
        row = row[:len(EXPECTED_HEADERS)]
    return row

def extract_rows_from_page(page) -> List[List[str]]:
    rows_out: List[List[str]] = []
    table = page.extract_table(TABLE_SETTINGS)

    # If extract_table fails, try extract_tables (sometimes multiple tables per page)
    if not table:
        tables = page.extract_tables(TABLE_SETTINGS) or []
        for t in tables:
            for r in t:
                if not r:
                    continue
                rr = normalize_row(r)
                if looks_like_header(rr):
                    continue
                # Skip empty rows
                if all(not cell for cell in rr):
                    continue
                rows_out.append(rr)
        return rows_out

    for r in table:
        if not r:
            continue
        rr = normalize_row(r)
        if looks_like_header(rr):
            continue
        if all(not cell for cell in rr):
            continue
        rows_out.append(rr)

    return rows_out

def main():
    all_rows: List[List[str]] = []

    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            rows = extract_rows_from_page(page)
            # Optional: attach page number for traceability by adding a column
            # for r in rows:
            #     r.append(str(i))
            all_rows.extend(rows)

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADERS)
        writer.writerows(all_rows)

    print(f"Done. Extracted {len(all_rows)} data rows into: {OUT_CSV}")

if __name__ == "__main__":
    main()
