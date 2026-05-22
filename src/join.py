"""Pipeline 1 — join step combining detections + color + pattern + CLIP outputs.

Merges the four per-garment CSVs produced by Pipeline 1 Steps 1, 2A, 2B, and
2C into a single ``yolo_fashion_attributes.csv`` with one row per garment.
The join is left-anchored on ``detections.csv`` — the source of truth for
the garment universe — so any garment dropped by a downstream step (missing
image, decode failure, zero-area crop) still appears in the output with
empty attribute columns. Every column from every input CSV is preserved.

This module is read-only with respect to all four input CSVs.

Run as a module from the project root::

    uv run python -m src.join
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils import JoinConfig, load_join_config

logger = logging.getLogger(__name__)


__all__ = [
    "CSV_COLUMNS",
    "JOIN_KEYS",
    "join_attribute_frames",
    "load_attribute_frames",
    "run_join",
    "write_joined_csv",
]


JOIN_KEYS: tuple[str, str] = ("image_id", "garment_id")

# Canonical output column order: join keys, then detection metadata, then
# Step 2A (color), Step 2B (pattern), Step 2C (CLIP refinement). Kept stable
# so downstream analyses can rely on column position.
CSV_COLUMNS: tuple[str, ...] = (
    "image_id",
    "garment_id",
    "category",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "dominant_r",
    "dominant_g",
    "dominant_b",
    "dominant_color_name",
    "palette_rgb",
    "laplacian_variance",
    "pattern_class",
    "category_yolo",
    "category_refined",
    "refined_confidence",
    "all_scores",
)


def _require_join_keys(df: pd.DataFrame, source: str) -> None:
    missing = [k for k in JOIN_KEYS if k not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required join column(s): {missing}"
        )


def _require_unique_keys(df: pd.DataFrame, source: str) -> None:
    dupes = df.duplicated(subset=list(JOIN_KEYS), keep=False)
    if dupes.any():
        sample = df.loc[dupes, list(JOIN_KEYS)].head(5).to_dict(orient="records")
        raise ValueError(
            f"{source} has duplicate {JOIN_KEYS} rows (first few: {sample})"
        )


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``image_id`` to string and ``garment_id`` to int64 in-place-safe.

    Pandas reads numeric-looking image filenames as int otherwise, which would
    break the merge against a string-typed counterpart.
    """
    out = df.copy()
    out["image_id"] = out["image_id"].astype("string")
    out["garment_id"] = out["garment_id"].astype("int64")
    return out


def load_attribute_frames(
    detections_csv: Path,
    color_csv: Path,
    pattern_csv: Path,
    clip_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all four input CSVs and normalize their join-key dtypes."""
    detections = _normalize_keys(pd.read_csv(detections_csv))
    color = _normalize_keys(pd.read_csv(color_csv))
    pattern = _normalize_keys(pd.read_csv(pattern_csv))
    clip = _normalize_keys(pd.read_csv(clip_csv))
    return detections, color, pattern, clip


def join_attribute_frames(
    detections: pd.DataFrame,
    color: pd.DataFrame,
    pattern: pd.DataFrame,
    clip: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge color/pattern/clip onto detections on ``(image_id, garment_id)``.

    Validates that the join keys exist and are unique in every frame. Missing
    downstream rows produce NaN-filled attribute cells; the per-frame skip
    count is reported via :func:`_log_coverage`. The output column order is
    fixed by :data:`CSV_COLUMNS`.
    """
    for df, name in (
        (detections, "detections"),
        (color, "color_attributes"),
        (pattern, "pattern_attributes"),
        (clip, "clip_refinement"),
    ):
        _require_join_keys(df, name)
        _require_unique_keys(df, name)

    keys = list(JOIN_KEYS)
    merged = detections.merge(color, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(pattern, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(clip, on=keys, how="left", validate="one_to_one")

    for col in CSV_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[list(CSV_COLUMNS)]


def _log_coverage(
    detections: pd.DataFrame,
    color: pd.DataFrame,
    pattern: pd.DataFrame,
    clip: pd.DataFrame,
) -> dict[str, int]:
    """Log how many detection rows each downstream frame is missing."""
    det_keys = set(map(tuple, detections[list(JOIN_KEYS)].to_numpy()))
    coverage: dict[str, int] = {}
    for name, df in (
        ("color", color),
        ("pattern", pattern),
        ("clip", clip),
    ):
        downstream_keys = set(map(tuple, df[list(JOIN_KEYS)].to_numpy()))
        missing = len(det_keys - downstream_keys)
        coverage[name] = missing
        if missing:
            logger.warning(
                "%s is missing %d of %d detection rows; those cells will be empty",
                name,
                missing,
                len(det_keys),
            )
    return coverage


def write_joined_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write the joined frame to ``path`` with the canonical column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _print_summary(
    detections: pd.DataFrame,
    joined: pd.DataFrame,
    coverage: dict[str, int],
) -> None:
    print()
    print("=== Pipeline 1: join summary ===")
    print(f"Detections (input):     {len(detections)}")
    print(f"Joined rows (output):   {len(joined)}")
    print("Missing per source CSV:")
    for name, missing in coverage.items():
        print(f"  {name:<10} {missing}")
    print()


def run_join(config: JoinConfig) -> pd.DataFrame:
    """Execute the full join pipeline. Returns the merged DataFrame."""
    detections, color, pattern, clip = load_attribute_frames(
        config.detections_csv,
        config.color_csv,
        config.pattern_csv,
        config.clip_csv,
    )

    coverage = _log_coverage(detections, color, pattern, clip)
    joined = join_attribute_frames(detections, color, pattern, clip)
    write_joined_csv(joined, config.output_csv)
    _print_summary(detections, joined, coverage)
    return joined


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_join_config()
    run_join(config)


if __name__ == "__main__":
    main()
