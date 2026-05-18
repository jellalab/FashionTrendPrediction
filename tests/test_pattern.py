"""Tests for Pipeline 1 Step 2B: pattern complexity scoring.

Synthetic images and a small ``detections.csv`` are constructed in a tmp dir
so the suite runs with no network access and no access to ``data/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import pattern
from src.utils import PatternConfig


# --- synthetic image helpers ----------------------------------------------


def _write_solid(
    path: Path,
    size: tuple[int, int] = (200, 200),
    color: tuple[int, int, int] = (128, 128, 128),
) -> None:
    width, height = size
    image = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_checkerboard(
    path: Path,
    size: tuple[int, int] = (200, 200),
    cell: int = 4,
) -> None:
    width, height = size
    ys, xs = np.indices((height, width))
    mask = ((xs // cell) + (ys // cell)) % 2 == 0
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[mask] = 255
    cv2.imwrite(str(path), image)


# --- laplacian_variance ----------------------------------------------------


def test_solid_image_variance_near_zero(tmp_path: Path) -> None:
    img_path = tmp_path / "solid.png"
    _write_solid(img_path, color=(100, 100, 100))
    gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
    assert pattern.laplacian_variance(gray) < 1e-6


def test_checkerboard_variance_much_higher_than_solid(tmp_path: Path) -> None:
    solid_path = tmp_path / "solid.png"
    check_path = tmp_path / "check.png"
    _write_solid(solid_path)
    _write_checkerboard(check_path)

    solid_gray = cv2.cvtColor(cv2.imread(str(solid_path)), cv2.COLOR_BGR2GRAY)
    check_gray = cv2.cvtColor(cv2.imread(str(check_path)), cv2.COLOR_BGR2GRAY)

    solid_var = pattern.laplacian_variance(solid_gray)
    check_var = pattern.laplacian_variance(check_gray)

    assert check_var > 1000.0
    assert check_var > solid_var * 1000


# --- classify_patterns -----------------------------------------------------


def test_quantile_thresholding_distribution_is_balanced() -> None:
    variances = np.arange(1, 101, dtype=np.float64)

    classes = pattern.classify_patterns(variances, 0.33, 0.66)

    counts = {cls: int((classes == cls).sum()) for cls in pattern.PATTERN_CLASSES}
    assert sum(counts.values()) == 100
    for cls, count in counts.items():
        assert 30 <= count <= 36, f"class {cls} count {count} too far from ~33"
    # values below the 33rd percentile must be plain
    assert classes[0] == "plain"
    # values above the 66th percentile must be patterned
    assert classes[-1] == "patterned"


def test_classify_patterns_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValueError):
        pattern.classify_patterns(np.arange(10.0), 0.7, 0.3)
    with pytest.raises(ValueError):
        pattern.classify_patterns(np.arange(10.0), 0.0, 0.5)


# --- bbox clipping ---------------------------------------------------------


def test_bbox_partially_outside_is_clipped() -> None:
    x1, y1, x2, y2 = pattern.clip_bbox_to_image((80, 80, 50, 50), (100, 100))
    assert (x1, y1, x2, y2) == (80, 80, 100, 100)


def test_bbox_negative_origin_is_clipped() -> None:
    x1, y1, x2, y2 = pattern.clip_bbox_to_image((-20, -10, 50, 40), (100, 100))
    assert (x1, y1, x2, y2) == (0, 0, 30, 30)


def test_bbox_entirely_outside_collapses_to_zero_area() -> None:
    x1, y1, x2, y2 = pattern.clip_bbox_to_image((150, 150, 50, 50), (100, 100))
    assert x2 - x1 == 0 and y2 - y1 == 0


# --- end-to-end ------------------------------------------------------------


def _make_config(tmp_path: Path) -> PatternConfig:
    return PatternConfig(
        detections_csv=tmp_path / "detections.csv",
        images_dir=tmp_path / "images",
        output_csv=tmp_path / "pattern_attributes.csv",
        center_crop_fraction=0.6,
        quantile_low=0.33,
        quantile_high=0.66,
    )


def _write_detections(path: Path, records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def test_missing_image_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "missing.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 30,
                "bbox_h": 30,
            }
        ],
    )

    with caplog.at_level(logging.WARNING, logger="src.pattern"):
        df = pattern.run_pattern_detection(config)

    assert len(df) == 0
    assert config.output_csv.exists()
    assert any("missing.jpg" in r.message for r in caplog.records)


def test_bbox_past_image_bounds_does_not_crash(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.jpg", size=(100, 100))
    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "a.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 50,
                "bbox_y": 50,
                "bbox_w": 200,
                "bbox_h": 200,
            }
        ],
    )

    df = pattern.run_pattern_detection(config)

    assert len(df) == 1
    assert df["image_id"].iloc[0] == "a.jpg"
    assert df["pattern_class"].iloc[0] in pattern.PATTERN_CLASSES


def test_end_to_end_assigns_all_three_classes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()

    _write_solid(config.images_dir / "plain.jpg", color=(180, 180, 180))
    _write_checkerboard(config.images_dir / "busy.jpg", cell=2)

    mid = np.zeros((200, 200, 3), dtype=np.uint8)
    mid[::20, :] = 255
    cv2.imwrite(str(config.images_dir / "mid.jpg"), mid)

    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "plain.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 180,
                "bbox_h": 180,
            },
            {
                "image_id": "mid.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 180,
                "bbox_h": 180,
            },
            {
                "image_id": "busy.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 180,
                "bbox_h": 180,
            },
        ],
    )

    df = pattern.run_pattern_detection(config)

    assert len(df) == 3
    assert set(df["pattern_class"]) == set(pattern.PATTERN_CLASSES)
    plain_var = df.loc[df["image_id"] == "plain.jpg", "laplacian_variance"].iloc[0]
    busy_var = df.loc[df["image_id"] == "busy.jpg", "laplacian_variance"].iloc[0]
    assert plain_var < busy_var
    assert (
        df.loc[df["image_id"] == "plain.jpg", "pattern_class"].iloc[0] == "plain"
    )
    assert (
        df.loc[df["image_id"] == "busy.jpg", "pattern_class"].iloc[0] == "patterned"
    )


def test_row_count_matches_input_minus_skipped(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.jpg")
    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "a.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 50,
                "bbox_h": 50,
            },
            {
                "image_id": "missing.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 50,
                "bbox_h": 50,
            },
        ],
    )

    df = pattern.run_pattern_detection(config)

    on_disk = pd.read_csv(config.output_csv)
    assert len(df) == 1
    assert len(on_disk) == 1


def test_run_is_idempotent(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "plain.jpg", color=(180, 180, 180))
    _write_checkerboard(config.images_dir / "busy.jpg")
    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "plain.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 180,
                "bbox_h": 180,
            },
            {
                "image_id": "busy.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 180,
                "bbox_h": 180,
            },
        ],
    )

    pattern.run_pattern_detection(config)
    first = config.output_csv.read_bytes()

    pattern.run_pattern_detection(config)
    second = config.output_csv.read_bytes()

    assert first == second


def test_csv_schema_columns_and_dtypes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.jpg")
    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "a.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 50,
                "bbox_h": 50,
            }
        ],
    )

    df = pattern.run_pattern_detection(config)

    assert list(df.columns) == list(pattern.CSV_COLUMNS)
    assert df["garment_id"].dtype == np.int64
    assert df["laplacian_variance"].dtype == np.float64
    assert str(df["image_id"].dtype) == "string"
    assert str(df["pattern_class"].dtype) == "string"


def test_detections_csv_is_not_modified(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.jpg")
    _write_detections(
        config.detections_csv,
        [
            {
                "image_id": "a.jpg",
                "garment_id": 0,
                "category": "shirt",
                "confidence": 0.9,
                "bbox_x": 10,
                "bbox_y": 10,
                "bbox_w": 50,
                "bbox_h": 50,
            }
        ],
    )
    before = config.detections_csv.read_bytes()

    pattern.run_pattern_detection(config)

    assert config.detections_csv.read_bytes() == before
