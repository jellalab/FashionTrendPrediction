"""Shared cropping helpers used by all per-garment attribute pipelines.

Both the pattern-complexity (Step 2B) and color-extraction (Step 2A)
modules read garment bounding boxes from ``detections.csv`` and apply the
same two-stage crop:

1. Clip the bbox to image bounds (``clip_bbox_to_image``) and slice it
   out of the source image.
2. Take an inner center crop (``center_crop``) — keeping the middle
   ``fraction`` on each axis — to reduce contamination from skin,
   background, and adjacent garments.

Centralising these helpers keeps the two pipelines in lock-step: any
change to how a garment is isolated propagates to both attribute
extractors.
"""

from __future__ import annotations

import numpy as np


def clip_bbox_to_image(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clip an ``(x, y, w, h)`` bbox to image bounds.

    Returns integer pixel ``(x1, y1, x2, y2)``. A bbox fully outside the image
    collapses to a zero-area rectangle, which the caller is expected to detect
    and skip.
    """
    height, width = image_shape
    x, y, w, h = bbox
    x1 = int(max(0, min(width, round(x))))
    y1 = int(max(0, min(height, round(y))))
    x2 = int(max(0, min(width, round(x + w))))
    y2 = int(max(0, min(height, round(y + h))))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return x1, y1, x2, y2


def center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Return the centered ``fraction``-by-``fraction`` sub-image.

    ``fraction=0.6`` keeps the middle 60% on each axis, discarding 20% on each
    side. The returned array is a view into ``image``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    h, w = image.shape[:2]
    new_w = int(round(w * fraction))
    new_h = int(round(h * fraction))
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    return image[y0 : y0 + new_h, x0 : x0 + new_w]
