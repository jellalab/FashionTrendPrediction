"""Pipeline 1 Step 2A — dominant color extraction.

For each garment bounding box in ``detections.csv``, this module crops the
garment from the source image, takes an inner center crop (same fraction as
the pattern module), converts to CIELAB, runs K-means over the pixels, and
records:

* ``dominant_r`` / ``dominant_g`` / ``dominant_b`` — the largest cluster
  centroid converted back to 8-bit RGB.
* ``dominant_color_name`` — the nearest entry in a curated fashion palette,
  measured by Euclidean distance in LAB (not RGB).
* ``palette_rgb`` — the top-3 cluster centroids (by pixel count) as a
  JSON-serialized list of RGB triples.

The clustering happens in LAB because perceptual distances there are more
faithful to human color judgement than in RGB. No skin-tone or background
filtering is applied: we trust the center crop and the majority-cluster
assumption.

This module is read-only with respect to upstream outputs (``detections.csv``
is never modified).

Run as a module from the project root::

    uv run python -m src.color
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.crop_utils import center_crop, clip_bbox_to_image
from src.utils import ColorConfig, PaletteEntry, load_color_config

logger = logging.getLogger(__name__)


CSV_COLUMNS: tuple[str, ...] = (
    "image_id",
    "garment_id",
    "dominant_r",
    "dominant_g",
    "dominant_b",
    "dominant_color_name",
    "palette_rgb",
)

CSV_DTYPES: dict[str, str] = {
    "image_id": "string",
    "garment_id": "int64",
    "dominant_r": "int64",
    "dominant_g": "int64",
    "dominant_b": "int64",
    "dominant_color_name": "string",
    "palette_rgb": "string",
}


RGB = tuple[int, int, int]


# --- color-space conversions ----------------------------------------------


def rgb_to_lab(rgb: RGB) -> np.ndarray:
    """Convert one 8-bit RGB triple to LAB (float64, OpenCV scaling)."""
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    return lab[0, 0].astype(np.float64)


def lab_to_rgb(lab: np.ndarray) -> RGB:
    """Convert one LAB triple (float) back to an 8-bit RGB triple.

    K-means centroids are real-valued; OpenCV's LAB2RGB expects uint8, so we
    round-and-clip before conversion. The output channels are guaranteed to
    fall in ``[0, 255]`` by the clip on the way in.
    """
    lab_u8 = np.clip(np.round(lab), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def bgr_to_lab_pixels(bgr: np.ndarray) -> np.ndarray:
    """Convert an H×W×3 BGR image (as read by cv2.imread) to an N×3 LAB array.

    OpenCV decodes images as BGR; the task spec requires the RGB→LAB code
    path, so we convert BGR→RGB explicitly before the LAB conversion rather
    than using ``COLOR_BGR2LAB`` directly.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3).astype(np.float64)


# --- clustering -----------------------------------------------------------


def cluster_lab_pixels(
    lab_pixels: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run K-means on LAB pixels and return ``(centers, counts)``.

    Both arrays are sorted by descending pixel count so the largest cluster
    is always at index 0. ``centers`` is shape ``(k, 3)`` in LAB; ``counts``
    is shape ``(k,)``.

    If the number of unique pixels is smaller than ``k`` (e.g. a solid-color
    crop after rounding), K-means would warn and produce degenerate clusters;
    we transparently fall back to clustering at the lower K and pad the
    result with zero-count entries so the output schema stays fixed.
    """
    unique = np.unique(lab_pixels, axis=0)
    effective_k = min(k, len(unique))
    if effective_k < 1:
        raise ValueError("cannot cluster zero pixels")

    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(lab_pixels)

    counts = np.bincount(labels, minlength=effective_k)
    order = np.argsort(-counts, kind="stable")
    centers = kmeans.cluster_centers_[order]
    counts = counts[order]

    if effective_k < k:
        pad_centers = np.zeros((k - effective_k, 3), dtype=centers.dtype)
        pad_counts = np.zeros(k - effective_k, dtype=counts.dtype)
        centers = np.vstack([centers, pad_centers])
        counts = np.concatenate([counts, pad_counts])
    return centers, counts


# --- palette naming -------------------------------------------------------


def palette_lab_matrix(palette: tuple[PaletteEntry, ...]) -> np.ndarray:
    """Pre-compute LAB coordinates of every palette entry (shape ``(N, 3)``)."""
    return np.vstack([rgb_to_lab(entry.rgb) for entry in palette])


def nearest_palette_name(
    lab_point: np.ndarray,
    palette: tuple[PaletteEntry, ...],
    palette_lab: np.ndarray,
) -> str:
    """Return the palette name nearest to ``lab_point`` in Euclidean LAB.

    Ties are resolved deterministically: ``np.argmin`` returns the lowest
    index, so palette ordering in ``config/color.yaml`` is load-bearing.
    """
    diffs = palette_lab - lab_point
    distances = np.sqrt((diffs * diffs).sum(axis=1))
    idx = int(np.argmin(distances))
    return palette[idx].name


# --- per-row pipeline -----------------------------------------------------


def extract_color_for_row(
    row: dict[str, Any],
    images_dir: Path,
    center_crop_fraction: float,
    k: int,
    random_state: int,
    palette: tuple[PaletteEntry, ...],
    palette_lab: np.ndarray,
) -> dict[str, Any] | None:
    """Run the full extraction for a single detection row.

    Returns ``None`` (and logs a warning) on any failure: missing file,
    undecodable image, bbox that clips to zero area, or zero-area center
    crop. No clustering is attempted unless the crop has at least one pixel.
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
            "Skipping %s (garment %s): zero-area center crop",
            image_id,
            garment_id,
        )
        return None

    lab_pixels = bgr_to_lab_pixels(inner)
    centers, _counts = cluster_lab_pixels(lab_pixels, k, random_state)

    dominant_lab = centers[0]
    dominant_rgb = lab_to_rgb(dominant_lab)
    top_palette: list[list[int]] = [list(lab_to_rgb(centers[i])) for i in range(k)]

    color_name = nearest_palette_name(dominant_lab, palette, palette_lab)

    return {
        "image_id": image_id,
        "garment_id": int(garment_id),
        "dominant_r": dominant_rgb[0],
        "dominant_g": dominant_rgb[1],
        "dominant_b": dominant_rgb[2],
        "dominant_color_name": color_name,
        "palette_rgb": json.dumps(top_palette),
    }


def process_detections(
    detections: pd.DataFrame,
    images_dir: Path,
    center_crop_fraction: float,
    k: int,
    random_state: int,
    palette: tuple[PaletteEntry, ...],
) -> tuple[list[dict[str, Any]], int]:
    """Run extraction on every detection row; return ``(rows, skipped)``."""
    palette_lab = palette_lab_matrix(palette)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in tqdm(
        detections.to_dict(orient="records"),
        desc="Extracting colors",
        unit="garment",
    ):
        result = extract_color_for_row(
            record,
            images_dir,
            center_crop_fraction,
            k,
            random_state,
            palette,
            palette_lab,
        )
        if result is None:
            skipped += 1
            continue
        rows.append(result)
    return rows, skipped


def write_color_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    """Write color rows to CSV with the canonical schema and dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    df = df.astype(CSV_DTYPES)
    df.to_csv(path, index=False)
    return df


def _print_summary(total_input: int, skipped: int, df: pd.DataFrame) -> None:
    print()
    print("=== Pipeline 1 Step 2A: dominant color summary ===")
    print(f"Total detections in input: {total_input}")
    print(f"Processed:                 {len(df)}")
    print(f"Skipped (errors):          {skipped}")
    if len(df) == 0:
        print()
        return

    counts = Counter(df["dominant_color_name"].tolist())
    total = len(df)
    print("Dominant color distribution:")
    for name, n in counts.most_common():
        pct = 100.0 * n / total
        print(f"  {name:<12} {n:>6}  ({pct:5.1f}%)")
    print()


def run_color_extraction(config: ColorConfig) -> pd.DataFrame:
    """Execute the full color-extraction pipeline. Returns the result DataFrame."""
    detections = pd.read_csv(config.detections_csv)

    rows, skipped = process_detections(
        detections,
        config.images_dir,
        config.center_crop_fraction,
        config.kmeans_k,
        config.random_state,
        config.palette,
    )

    df = write_color_csv(rows, config.output_csv)
    _print_summary(len(detections), skipped, df)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_color_config()
    run_color_extraction(config)


if __name__ == "__main__":
    main()
