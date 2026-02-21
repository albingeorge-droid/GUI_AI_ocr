import pdfplumber
import csv
import re
from typing import List, Optional
from pathlib import Path

# ✅ CHANGE THESE TWO LINES ONLY
PDF_DIR = Path(r"C:\Users\albin\Documents\GitHub\GUI_AI_ocr\pdf_data")  # folder containing PDFs
OUT_DIR = Path(r"C:\Users\albin\Documents\GitHub\GUI_AI_ocr\csv_output")  # folder to save CSVs

EXPECTED_HEADERS = [
    "No.", "S.No", "Reg.No.", "IstParty", "IIndParty",
    "Type of Deed", "Address", "Value", "Stamp Paid", "Book No."
]

TABLE_SETTINGS = {
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
    return re.sub(r"\s+", " ", x).strip()

def looks_like_header(row: List[str]) -> bool:
    row_norm = [clean_cell(c).lower() for c in row]
    hdr_norm = [h.lower() for h in EXPECTED_HEADERS]
    matches = sum(1 for a, b in zip(row_norm, hdr_norm) if a == b)
    return matches >= 5

def normalize_row(row: List[str]) -> List[str]:
    row = [clean_cell(c) for c in row]
    if len(row) < len(EXPECTED_HEADERS):
        row += [""] * (len(EXPECTED_HEADERS) - len(row))
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
                if looks_like_header(rr) or all(not cell for cell in rr):
                    continue
                rows_out.append(rr)
        return rows_out

    for r in table:
        if not r:
            continue
        rr = normalize_row(r)
        if looks_like_header(rr) or all(not cell for cell in rr):
            continue
        rows_out.append(rr)

    return rows_out

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # create output folder if not exists

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {PDF_DIR}")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        rows_for_pdf: List[List[str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                rows_for_pdf.extend(extract_rows_from_page(page))

        # ✅ Output CSV name same as PDF name
        out_csv = OUT_DIR / f"{pdf_path.stem}.csv"

        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(EXPECTED_HEADERS)
            writer.writerows(rows_for_pdf)

        print(f"Saved: {out_csv} | Rows: {len(rows_for_pdf)}")

    print("Done. All PDFs processed.")

if __name__ == "__main__":
    main()
