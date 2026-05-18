"""Tests for Pipeline 1 Step 2A: dominant color extraction.

Synthetic images and a small ``detections.csv`` are constructed in a tmp dir
so the suite runs with no network access and no access to ``data/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import color
from src.utils import ColorConfig, PaletteEntry, load_color_config


# --- helpers --------------------------------------------------------------


def _write_solid_rgb(
    path: Path,
    rgb: tuple[int, int, int],
    size: tuple[int, int] = (200, 200),
) -> None:
    """Write a PNG of one solid color, given as an RGB triple.

    cv2 stores images as BGR on disk, so the triple is flipped before writing
    to ensure the file decodes back to exactly ``rgb`` when read by cv2.
    """
    width, height = size
    bgr = (rgb[2], rgb[1], rgb[0])
    image = np.full((height, width, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_asymmetric_checkerboard(
    path: Path,
    size: tuple[int, int] = (200, 200),
) -> None:
    """Write a black/white pattern with ~60% black, ~40% white.

    The asymmetry guarantees the dominant cluster is black so the test can
    pin the expected ``dominant_color_name`` deterministically.
    """
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    ys, xs = np.indices((height, width))
    black_mask = ((xs + ys) % 5) < 3
    image[black_mask] = 0
    cv2.imwrite(str(path), image)


def _detection_row(
    image_id: str,
    bbox: tuple[float, float, float, float] = (10, 10, 180, 180),
) -> dict:
    return {
        "image_id": image_id,
        "garment_id": 0,
        "category": "shirt",
        "confidence": 0.9,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
    }


def _make_config(
    tmp_path: Path,
    palette: tuple[PaletteEntry, ...] | None = None,
) -> ColorConfig:
    return ColorConfig(
        detections_csv=tmp_path / "detections.csv",
        images_dir=tmp_path / "images",
        output_csv=tmp_path / "color_attributes.csv",
        center_crop_fraction=0.6,
        kmeans_k=3,
        random_state=42,
        palette=palette if palette is not None else _default_palette(),
    )


def _default_palette() -> tuple[PaletteEntry, ...]:
    """Load the production palette from disk so tests exercise the real list."""
    return load_color_config().palette


def _write_detections(path: Path, records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


# --- color-space conversions ----------------------------------------------


def test_rgb_to_lab_roundtrips_within_one_step() -> None:
    """Round-tripping RGB→LAB→RGB through uint8 should stay within a small
    quantization error. This is a sanity check on our conversion helpers."""
    for rgb in [(0, 0, 0), (255, 255, 255), (200, 30, 30), (28, 38, 65)]:
        lab = color.rgb_to_lab(rgb)
        back = color.lab_to_rgb(lab)
        for channel_in, channel_out in zip(rgb, back, strict=True):
            assert abs(channel_in - channel_out) <= 2


# --- clustering & naming on synthetic crops -------------------------------


def test_solid_red_image_names_to_red(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "red.png", (200, 30, 30))
    _write_detections(config.detections_csv, [_detection_row("red.png")])

    df = color.run_color_extraction(config)

    assert len(df) == 1
    assert df["dominant_color_name"].iloc[0] == "red"
    # Round-trip through LAB shouldn't move the recovered RGB far.
    assert abs(int(df["dominant_r"].iloc[0]) - 200) <= 2
    assert abs(int(df["dominant_g"].iloc[0]) - 30) <= 2
    assert abs(int(df["dominant_b"].iloc[0]) - 30) <= 2


def test_solid_navy_image_names_to_navy(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "navy.png", (28, 38, 65))
    _write_detections(config.detections_csv, [_detection_row("navy.png")])

    df = color.run_color_extraction(config)

    assert len(df) == 1
    assert df["dominant_color_name"].iloc[0] == "navy"


def test_checkerboard_clusters_into_black_and_white(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_asymmetric_checkerboard(config.images_dir / "check.png")
    _write_detections(config.detections_csv, [_detection_row("check.png")])

    df = color.run_color_extraction(config)

    assert len(df) == 1
    # ~60% black means the dominant cluster — and thus the name — is black.
    assert df["dominant_color_name"].iloc[0] == "black"

    palette_rgb = json.loads(df["palette_rgb"].iloc[0])
    assert len(palette_rgb) == config.kmeans_k

    # Top two centroids should be near pure black and pure white.
    near_black = [c for c in palette_rgb[:2] if max(c) <= 5]
    near_white = [c for c in palette_rgb[:2] if min(c) >= 250]
    assert len(near_black) == 1, f"expected one near-black centroid in {palette_rgb}"
    assert len(near_white) == 1, f"expected one near-white centroid in {palette_rgb}"


# --- palette nearest-neighbor logic ---------------------------------------


def test_palette_nearest_neighbor_breaks_ties_deterministically() -> None:
    """Equidistant LAB points must resolve to the lower-index palette entry."""
    palette = (
        PaletteEntry(name="alpha", rgb=(50, 50, 50)),
        PaletteEntry(name="bravo", rgb=(200, 200, 200)),
    )
    palette_lab = color.palette_lab_matrix(palette)

    midpoint = (palette_lab[0] + palette_lab[1]) / 2.0

    # Distances must actually be equal.
    d_a = float(np.linalg.norm(palette_lab[0] - midpoint))
    d_b = float(np.linalg.norm(palette_lab[1] - midpoint))
    assert d_a == d_b

    assert color.nearest_palette_name(midpoint, palette, palette_lab) == "alpha"
    # Re-running gives the same answer.
    assert color.nearest_palette_name(midpoint, palette, palette_lab) == "alpha"


def test_palette_nearest_neighbor_uses_lab_not_rgb() -> None:
    """Two RGB-equidistant entries should still resolve to the LAB-nearest."""
    palette = (
        PaletteEntry(name="red", rgb=(200, 30, 30)),
        PaletteEntry(name="navy", rgb=(28, 38, 65)),
    )
    palette_lab = color.palette_lab_matrix(palette)
    red_lab = color.rgb_to_lab((200, 30, 30))
    assert color.nearest_palette_name(red_lab, palette, palette_lab) == "red"


# --- determinism / idempotency --------------------------------------------


def test_run_is_idempotent(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "red.png", (200, 30, 30))
    _write_asymmetric_checkerboard(config.images_dir / "check.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("red.png"), _detection_row("check.png")],
    )

    color.run_color_extraction(config)
    first = config.output_csv.read_bytes()

    color.run_color_extraction(config)
    second = config.output_csv.read_bytes()

    assert first == second


# --- output schema / acceptance criteria ----------------------------------


def test_csv_schema_columns_and_dtypes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "red.png", (200, 30, 30))
    _write_detections(config.detections_csv, [_detection_row("red.png")])

    df = color.run_color_extraction(config)

    assert list(df.columns) == list(color.CSV_COLUMNS)
    assert df["garment_id"].dtype == np.int64
    assert df["dominant_r"].dtype == np.int64
    assert df["dominant_g"].dtype == np.int64
    assert df["dominant_b"].dtype == np.int64
    assert str(df["image_id"].dtype) == "string"
    assert str(df["dominant_color_name"].dtype) == "string"
    assert str(df["palette_rgb"].dtype) == "string"


def test_all_rgb_channels_in_valid_range(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "red.png", (200, 30, 30))
    _write_solid_rgb(config.images_dir / "white.png", (255, 255, 255))
    _write_solid_rgb(config.images_dir / "black.png", (0, 0, 0))
    _write_detections(
        config.detections_csv,
        [
            _detection_row("red.png"),
            _detection_row("white.png"),
            _detection_row("black.png"),
        ],
    )

    df = color.run_color_extraction(config)

    for col in ("dominant_r", "dominant_g", "dominant_b"):
        assert (df[col] >= 0).all() and (df[col] <= 255).all()

    palette_names = {e.name for e in config.palette}
    assert set(df["dominant_color_name"]).issubset(palette_names)

    for raw in df["palette_rgb"]:
        triples = json.loads(raw)
        assert len(triples) == config.kmeans_k
        for triple in triples:
            assert len(triple) == 3
            for c in triple:
                assert 0 <= c <= 255


# --- edge cases: must log and skip, not crash -----------------------------


def test_missing_image_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_detections(config.detections_csv, [_detection_row("missing.png")])

    with caplog.at_level(logging.WARNING, logger="src.color"):
        df = color.run_color_extraction(config)

    assert len(df) == 0
    assert config.output_csv.exists()
    assert any("missing.png" in r.message for r in caplog.records)


def test_bbox_past_image_bounds_does_not_crash(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "a.png", (200, 30, 30), size=(100, 100))
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", bbox=(50, 50, 200, 200))],
    )

    df = color.run_color_extraction(config)

    assert len(df) == 1
    assert df["dominant_color_name"].iloc[0] in {e.name for e in config.palette}


def test_bbox_entirely_outside_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "a.png", (200, 30, 30), size=(100, 100))
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", bbox=(500, 500, 50, 50))],
    )

    with caplog.at_level(logging.WARNING, logger="src.color"):
        df = color.run_color_extraction(config)

    assert len(df) == 0
    assert any("zero area" in r.message for r in caplog.records)


def test_corrupt_image_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    bad = config.images_dir / "bad.png"
    bad.write_bytes(b"not a real image")
    _write_detections(config.detections_csv, [_detection_row("bad.png")])

    with caplog.at_level(logging.WARNING, logger="src.color"):
        df = color.run_color_extraction(config)

    assert len(df) == 0
    assert any("bad.png" in r.message for r in caplog.records)


def test_all_black_and_all_white_do_not_crash(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "black.png", (0, 0, 0))
    _write_solid_rgb(config.images_dir / "white.png", (255, 255, 255))
    _write_detections(
        config.detections_csv,
        [_detection_row("black.png"), _detection_row("white.png")],
    )

    df = color.run_color_extraction(config)

    assert len(df) == 2
    by_id = df.set_index("image_id")["dominant_color_name"]
    assert by_id["black.png"] == "black"
    assert by_id["white.png"] == "white"


# --- upstream protection --------------------------------------------------


def test_detections_csv_is_not_modified(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "a.png", (200, 30, 30))
    _write_detections(config.detections_csv, [_detection_row("a.png")])
    before = config.detections_csv.read_bytes()

    color.run_color_extraction(config)

    assert config.detections_csv.read_bytes() == before


def test_row_count_matches_input_minus_skipped(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid_rgb(config.images_dir / "a.png", (200, 30, 30))
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png"), _detection_row("missing.png")],
    )

    df = color.run_color_extraction(config)
    on_disk = pd.read_csv(config.output_csv)

    assert len(df) == 1
    assert len(on_disk) == 1
