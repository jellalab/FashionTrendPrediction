"""Pipeline 1 Step 2B — pattern complexity scoring.

For each garment bounding box in ``detections.csv``, this module crops the
garment from the source image, takes an inner center crop to reduce
contamination from skin / background / adjacent garments, and computes the
variance of the Laplacian as a scalar measure of visual complexity. Each
garment is then bucketed into ``plain`` / ``subtle`` / ``patterned`` via
dataset-relative quantile thresholds.

This module is read-only with respect to upstream outputs (``detections.csv``
is never modified). It does not classify pattern *type* (stripes, florals,
plaid) — only complexity.

Run as a module from the project root::

    uv run python -m src.pattern
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import PatternConfig, load_pattern_config

logger = logging.getLogger(__name__)


CSV_COLUMNS: tuple[str, ...] = (
    "image_id",
    "garment_id",
    "laplacian_variance",
    "pattern_class",
)

CSV_DTYPES: dict[str, str] = {
    "image_id": "string",
    "garment_id": "int64",
    "laplacian_variance": "float64",
    "pattern_class": "string",
}

PATTERN_CLASSES: tuple[str, str, str] = ("plain", "subtle", "patterned")


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


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of ``cv2.Laplacian(gray, CV_64F)`` with the default kernel."""
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F)
    return float(lap.var())


def classify_patterns(
    variances: np.ndarray,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    """Bucket variance values into ``plain`` / ``subtle`` / ``patterned``.

    Thresholds are the empirical ``quantile_low`` and ``quantile_high``
    quantiles of the provided ``variances`` array, so the distribution across
    classes is approximately balanced by construction.
    """
    if not (0.0 < quantile_low < quantile_high < 1.0):
        raise ValueError(
            "quantiles must satisfy 0 < quantile_low < quantile_high < 1"
        )
    if len(variances) == 0:
        return np.array([], dtype=object)

    low = np.quantile(variances, quantile_low)
    high = np.quantile(variances, quantile_high)
    classes = np.where(
        variances < low,
        PATTERN_CLASSES[0],
        np.where(variances < high, PATTERN_CLASSES[1], PATTERN_CLASSES[2]),
    )
    return classes


def compute_variance_for_row(
    row: dict[str, Any],
    images_dir: Path,
    center_crop_fraction: float,
) -> float | None:
    """Open the image, crop to bbox, take a center crop, return variance.

    Returns ``None`` (and logs a warning) on any failure: missing file,
    undecodable image, bbox that clips to zero area, or zero-area center crop.
    Operates on raw pixel data — no resize, normalization, or color correction
    is applied before the variance is computed.
    """
    image_id = row["image_id"]
    garment_id = row["garment_id"]
    image_path = images_dir / image_id

    if not image_path.exists():
        logger.warning(
            "Skipping %s (garment %s): image file not found at %s",
            image_id,
            garment_id,
            image_path,
        )
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning(
            "Skipping %s (garment %s): cv2 failed to decode image",
            image_id,
            garment_id,
        )
        return None

    bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
    x1, y1, x2, y2 = clip_bbox_to_image(bbox, image.shape[:2])

    if x2 <= x1 or y2 <= y1:
        logger.warning(
            "Skipping %s (garment %s): bbox clipped to zero area",
            image_id,
            garment_id,
        )
        return None

    garment = image[y1:y2, x1:x2]
    inner = center_crop(garment, center_crop_fraction)

    if inner.size == 0 or inner.shape[0] == 0 or inner.shape[1] == 0:
        logger.warning(
            "Skipping %s (garment %s): zero-area center crop", image_id, garment_id
        )
        return None

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    return laplacian_variance(gray)


def process_detections(
    detections: pd.DataFrame,
    images_dir: Path,
    center_crop_fraction: float,
) -> tuple[list[dict[str, Any]], int]:
    """Compute Laplacian variance for every row; return ``(rows, skipped)``.

    ``rows`` carries ``image_id``, ``garment_id``, ``laplacian_variance`` —
    ``pattern_class`` is assigned later, once the full distribution is known.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in tqdm(
        detections.to_dict(orient="records"),
        desc="Scoring patterns",
        unit="garment",
    ):
        variance = compute_variance_for_row(record, images_dir, center_crop_fraction)
        if variance is None:
            skipped += 1
            continue
        rows.append(
            {
                "image_id": record["image_id"],
                "garment_id": int(record["garment_id"]),
                "laplacian_variance": variance,
            }
        )
    return rows, skipped


def write_pattern_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    """Write pattern rows to CSV with the canonical schema and dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    df = df.astype(CSV_DTYPES)
    df.to_csv(path, index=False)
    return df


def _print_summary(total_input: int, skipped: int, df: pd.DataFrame) -> None:
    print()
    print("=== Pipeline 1 Step 2B: pattern complexity summary ===")
    print(f"Total detections in input: {total_input}")
    print(f"Processed:                 {len(df)}")
    print(f"Skipped (errors):          {skipped}")
    if len(df) == 0:
        print()
        return

    variances = df["laplacian_variance"].to_numpy()
    print(f"Laplacian variance — mean:   {variances.mean():.4f}")
    print(f"Laplacian variance — median: {float(np.median(variances)):.4f}")
    print(f"Laplacian variance — std:    {variances.std():.4f}")

    print("Pattern class distribution:")
    counts = Counter(df["pattern_class"].tolist())
    total = len(df)
    for cls in PATTERN_CLASSES:
        n = counts.get(cls, 0)
        pct = 100.0 * n / total
        print(f"  {cls:<10} {n:>6}  ({pct:5.1f}%)")
    print()


def run_pattern_detection(config: PatternConfig) -> pd.DataFrame:
    """Execute the full pattern-scoring pipeline. Returns the result DataFrame."""
    detections = pd.read_csv(config.detections_csv)

    rows, skipped = process_detections(
        detections, config.images_dir, config.center_crop_fraction
    )

    if rows:
        variances = np.array([r["laplacian_variance"] for r in rows])
        classes = classify_patterns(
            variances, config.quantile_low, config.quantile_high
        )
        for row, cls in zip(rows, classes, strict=True):
            row["pattern_class"] = str(cls)

    df = write_pattern_csv(rows, config.output_csv)
    _print_summary(len(detections), skipped, df)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_pattern_config()
    run_pattern_detection(config)


if __name__ == "__main__":
    main()
