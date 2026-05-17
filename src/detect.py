"""Pipeline 1 — DeepFashion2 YOLOv8 garment detection and fashion filter.

Splits an input directory of images into ``accepted`` (≥1 garment detected at
or above the confidence threshold) vs. ``rejected_non_fashion`` and writes a
per-detection CSV.

Run as a module from the project root::

    uv run python -m src.detect
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.utils import DetectionConfig, ModelConfig, load_detection_config

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)

CSV_COLUMNS: tuple[str, ...] = (
    "image_id",
    "garment_id",
    "category",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
)

CSV_DTYPES: dict[str, str] = {
    "image_id": "string",
    "garment_id": "int64",
    "category": "string",
    "confidence": "float64",
    "bbox_x": "float64",
    "bbox_y": "float64",
    "bbox_w": "float64",
    "bbox_h": "float64",
}


def download_weights(model_config: ModelConfig) -> Path:
    """Download (or fetch from cache) the YOLOv8 weights file.

    Uses ``huggingface_hub`` with a local cache directory so the file is only
    pulled once per machine.
    """
    from huggingface_hub import hf_hub_download

    model_config.cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=model_config.repo_id,
        filename=model_config.filename,
        cache_dir=str(model_config.cache_dir),
    )
    return Path(path)


def load_model(weights_path: Path) -> Any:
    """Instantiate the YOLO model from a local weights file."""
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert an array-like (torch tensor, numpy array, list) to ndarray."""
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr)


def extract_detection_rows(
    result: Any,
    image_id: str,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Turn a single YOLO ``Result`` into one row per qualifying detection.

    Detections below ``confidence_threshold`` are skipped. ``garment_id`` is a
    0-based index over the *kept* detections only. The category label comes
    from ``result.names`` (the model metadata), never a hardcoded list.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    conf = _to_numpy(boxes.conf)
    cls = _to_numpy(boxes.cls)
    names: dict[int, str] = result.names

    rows: list[dict[str, Any]] = []
    garment_id = 0
    for i in range(len(conf)):
        c = float(conf[i])
        if c < confidence_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        rows.append(
            {
                "image_id": image_id,
                "garment_id": garment_id,
                "category": str(names[int(cls[i])]),
                "confidence": c,
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
            }
        )
        garment_id += 1
    return rows


def list_input_images(input_dir: Path) -> list[Path]:
    """Return a sorted list of image files in ``input_dir`` (non-recursive)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _verify_image(path: Path) -> bool:
    """Return True if PIL can decode the image, False otherwise."""
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Skipping corrupt image %s: %s", path.name, exc)
        return False
    return True


def _reset_output_dir(directory: Path) -> None:
    """Clear and recreate an output directory so re-runs are idempotent."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def write_detections_csv(rows: Iterable[dict[str, Any]], path: Path) -> pd.DataFrame:
    """Write detection rows to CSV with the canonical schema and dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=list(CSV_COLUMNS))
    df = df.astype(CSV_DTYPES)
    df.to_csv(path, index=False)
    return df


def _print_summary(
    total_input: int,
    accepted: int,
    rejected: int,
    corrupt: int,
    df: pd.DataFrame,
) -> None:
    total_garments = len(df)
    mean_per_accepted = (total_garments / accepted) if accepted else 0.0
    print()
    print("=== Pipeline 1: garment detection summary ===")
    print(f"Total input images:     {total_input}")
    print(f"Accepted (>=1 garment): {accepted}")
    print(f"Rejected (0 garments):  {rejected}")
    print(f"Corrupt / unreadable:   {corrupt}")
    print(f"Total garments:         {total_garments}")
    print(f"Mean garments / accept: {mean_per_accepted:.2f}")
    if total_garments:
        print("Category distribution:")
        counts = Counter(df["category"].tolist())
        for category, count in counts.most_common():
            print(f"  {category:<24} {count}")
    print()


def run_detection(config: DetectionConfig) -> pd.DataFrame:
    """Execute the full detection pipeline. Returns the detections DataFrame."""
    images = list_input_images(config.input_dir)
    if not images:
        logger.warning("No images found in %s", config.input_dir)

    _reset_output_dir(config.accepted_dir)
    _reset_output_dir(config.rejected_dir)

    weights = download_weights(config.model)
    model = load_model(weights)

    all_rows: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    corrupt = 0

    for image_path in tqdm(images, desc="Detecting", unit="img"):
        if not _verify_image(image_path):
            corrupt += 1
            continue

        results = model.predict(
            source=str(image_path),
            conf=config.confidence_threshold,
            verbose=False,
        )
        result = results[0]
        rows = extract_detection_rows(
            result, image_path.name, config.confidence_threshold
        )

        if rows:
            shutil.copy2(image_path, config.accepted_dir / image_path.name)
            accepted += 1
            all_rows.extend(rows)
        else:
            shutil.copy2(image_path, config.rejected_dir / image_path.name)
            rejected += 1

    df = write_detections_csv(all_rows, config.detections_csv)
    _print_summary(len(images), accepted, rejected, corrupt, df)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_detection_config()
    run_detection(config)


if __name__ == "__main__":
    main()
