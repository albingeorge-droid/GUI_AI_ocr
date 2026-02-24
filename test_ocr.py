"""
Extract text from a PDF using pdfplumber.

This does NOT do OCR. It will extract embedded/selectable text.
If your PDF is scanned images, pdfplumber will return little/empty text.

Install:
  pip install pdfplumber

Usage:
  - Saves per-page text to raw_outputs/page_###_pdfplumber.txt
  - Saves combined text to pdfplumber_all_text.txt
  - Optionally tries to extract tables to CSVs (if the PDF has real table structure)
"""

from pathlib import Path
import pdfplumber

PDF_PATH = r"C:\Users\albin\Documents\GitHub\GUI_AI_ocr\test_11.pdf"  # <-- change
OUT_DIR = Path(PDF_PATH).parent / "raw_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL_TEXT = Path(PDF_PATH).parent / "pdfplumber_all_text.txt"


def extract_all_text(pdf_path: str):
    combined = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            out_path = OUT_DIR / f"page_{i:03d}_pdfplumber.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"[Page {i}] text chars={len(text)} -> {out_path}")
            combined.append(f"===== PAGE {i} =====\n{text}\n")
    OUT_ALL_TEXT.write_text("\n".join(combined), encoding="utf-8")
    print(f"\nSaved combined -> {OUT_ALL_TEXT}")


def extract_tables_to_csv(pdf_path: str):
    """
    Tries to extract tables when PDF contains vector/text tables (not scanned images).
    Saves one CSV per detected table per page.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue
            for t_idx, table in enumerate(tables, start=1):
                csv_path = OUT_DIR / f"page_{i:03d}_table_{t_idx:02d}.csv"
                # simple CSV write (no pandas required)
                lines = []
                for row in table:
                    row = [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
                    # naive CSV escaping
                    esc = []
                    for c in row:
                        if any(ch in c for ch in [",", '"', "\n"]):
                            c = '"' + c.replace('"', '""') + '"'
                        esc.append(c)
                    lines.append(",".join(esc))
                csv_path.write_text("\n".join(lines), encoding="utf-8")
                print(f"[Page {i}] table {t_idx} -> {csv_path}")


if __name__ == "__main__":
    extract_all_text(PDF_PATH)
    # Uncomment if you want table CSV extraction attempts:
    # extract_tables_to_csv(PDF_PATH)