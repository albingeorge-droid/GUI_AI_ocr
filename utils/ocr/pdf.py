from io import BytesIO
from typing import List
from pdf2image import convert_from_path

def pdf_to_png_bytes(pdf_path: str, dpi: int = 250, max_pages: int | None = None) -> List[bytes]:
    images = convert_from_path(pdf_path, dpi=dpi)
    if max_pages is not None:
        images = images[:max_pages]

    png_pages: List[bytes] = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_pages.append(buf.getvalue())
    return png_pages
