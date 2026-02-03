# utils/ocr/orientation.py
from __future__ import annotations

from io import BytesIO
from typing import Iterable, Tuple
import numpy as np
from PIL import Image

_ROTATIONS: Iterable[int] = (0, 90, 180, 270)

def _score_text_lines(img: Image.Image) -> float:
    g = img.convert("L")
    g = g.resize((min(1200, g.size[0]), int(min(1200, g.size[0]) * g.size[1] / g.size[0])))
    a = np.array(g, dtype=np.uint8)

    thr = np.mean(a)
    bw = (a < thr).astype(np.uint8)

    proj = bw.sum(axis=1).astype(np.float32)
    return float(np.var(proj))

def auto_rotate_png_bytes(png_bytes: bytes) -> Tuple[bytes, int]:
    """
    Returns: (rotated_png_bytes, best_rot_degrees)
    best_rot_degrees ∈ {0, 90, 180, 270}
    """
    img0 = Image.open(BytesIO(png_bytes)).convert("RGB")

    best_rot = 0
    best_score = -1.0
    for rot in _ROTATIONS:
        img = img0 if rot == 0 else img0.rotate(rot, expand=True)
        s = _score_text_lines(img)
        if s > best_score:
            best_score = s
            best_rot = rot

    out = img0 if best_rot == 0 else img0.rotate(best_rot, expand=True)
    buf = BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), int(best_rot)
