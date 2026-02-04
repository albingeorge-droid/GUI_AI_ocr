from io import BytesIO
from typing import List

from pdf2image import convert_from_path
from PIL import Image

# Pillow safety: allow large images but we will downscale them ourselves
Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb check

# Hard limits for page size we actually send to the LLM
MAX_LONG_EDGE = 3500        # px – longest side after resize
MAX_TOTAL_PIXELS = 40_000_000  # e.g. 5000 x 8000 = 40M


def _downscale_if_needed(img: Image.Image) -> Image.Image:
    """
    Downscale very large images so they are safe/fast for OCR.
    Keeps aspect ratio; uses high-quality resampling.
    """
    w, h = img.size
    total = w * h

    # 1) Limit by total pixels
    if total > MAX_TOTAL_PIXELS:
        scale = (MAX_TOTAL_PIXELS / total) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = img.size  # update

    # 2) Limit by longest edge
    long_edge = max(w, h)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / float(long_edge)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def pdf_to_png_bytes(
    pdf_path: str,
    dpi: int = 180,                 # lower default DPI to keep images sane
    max_pages: int | None = None,
) -> List[bytes]:
    """
    Render a PDF to per-page PNG bytes.

    - Uses a moderate DPI (default 180) to avoid giant images.
    - Downscales each page if it exceeds MAX_TOTAL_PIXELS or MAX_LONG_EDGE
      before encoding to PNG and sending to Bedrock.
    """
    images = convert_from_path(pdf_path, dpi=dpi)

    if max_pages is not None:
        images = images[:max_pages]

    png_pages: List[bytes] = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = _downscale_if_needed(img)

        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_pages.append(buf.getvalue())

    return png_pages
