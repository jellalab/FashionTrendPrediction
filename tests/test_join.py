"""Tests for Pipeline 1 join: merging detections + color + pattern + CLIP.

All fixtures construct small in-memory DataFrames or write tiny CSVs to a tmp
dir; no real ``data/`` files are touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import join
from src.utils import JoinConfig


# --- builders -------------------------------------------------------------


def _detections_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["image_id"] = df["image_id"].astype("string")
    df["garment_id"] = df["garment_id"].astype("int64")
    return df


def _color_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["image_id"] = df["image_id"].astype("string")
    df["garment_id"] = df["garment_id"].astype("int64")
    return df


def _pattern_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["image_id"] = df["image_id"].astype("string")
    df["garment_id"] = df["garment_id"].astype("int64")
    return df


def _clip_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["image_id"] = df["image_id"].astype("string")
    df["garment_id"] = df["garment_id"].astype("int64")
    return df


def _detection_row(image_id: str, garment_id: int, category: str = "shirt") -> dict:
    return {
        "image_id": image_id,
        "garment_id": garment_id,
        "category": category,
        "confidence": 0.9,
        "bbox_x": 10.0,
        "bbox_y": 10.0,
        "bbox_w": 100.0,
        "bbox_h": 100.0,
    }


def _color_row(image_id: str, garment_id: int, name: str = "red") -> dict:
    return {
        "image_id": image_id,
        "garment_id": garment_id,
        "dominant_r": 200,
        "dominant_g": 30,
        "dominant_b": 30,
        "dominant_color_name": name,
        "palette_rgb": json.dumps([[200, 30, 30]]),
    }


def _pattern_row(
    image_id: str, garment_id: int, cls: str = "plain", variance: float = 12.5
) -> dict:
    return {
        "image_id": image_id,
        "garment_id": garment_id,
        "laplacian_variance": variance,
        "pattern_class": cls,
    }


def _clip_row(
    image_id: str,
    garment_id: int,
    category_yolo: str = "shirt",
    refined: str = "button-down shirt",
    confidence: float = 0.83,
) -> dict:
    return {
        "image_id": image_id,
        "garment_id": garment_id,
        "category_yolo": category_yolo,
        "category_refined": refined,
        "refined_confidence": confidence,
        "all_scores": json.dumps({refined: confidence}),
    }


# --- happy path -----------------------------------------------------------


def test_join_preserves_all_columns_from_every_source() -> None:
    detections = _detections_frame([_detection_row("a.jpg", 0)])
    color = _color_frame([_color_row("a.jpg", 0)])
    pattern = _pattern_frame([_pattern_row("a.jpg", 0)])
    clip = _clip_frame([_clip_row("a.jpg", 0)])

    out = join.join_attribute_frames(detections, color, pattern, clip)

    assert list(out.columns) == list(join.CSV_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["image_id"] == "a.jpg"
    assert row["garment_id"] == 0
    assert row["category"] == "shirt"
    assert row["dominant_color_name"] == "red"
    assert row["pattern_class"] == "plain"
    assert row["category_refined"] == "button-down shirt"


def test_join_matches_on_image_id_and_garment_id() -> None:
    detections = _detections_frame(
        [
            _detection_row("a.jpg", 0),
            _detection_row("a.jpg", 1, category="trousers"),
            _detection_row("b.jpg", 0, category="skirt"),
        ]
    )
    color = _color_frame(
        [
            _color_row("a.jpg", 0, name="red"),
            _color_row("a.jpg", 1, name="navy"),
            _color_row("b.jpg", 0, name="green"),
        ]
    )
    pattern = _pattern_frame(
        [
            _pattern_row("a.jpg", 0, cls="plain"),
            _pattern_row("a.jpg", 1, cls="patterned"),
            _pattern_row("b.jpg", 0, cls="subtle"),
        ]
    )
    clip = _clip_frame(
        [
            _clip_row("a.jpg", 0, refined="button-down shirt"),
            _clip_row("a.jpg", 1, category_yolo="trousers", refined="jeans"),
            _clip_row("b.jpg", 0, category_yolo="skirt", refined="mini skirt"),
        ]
    )

    out = join.join_attribute_frames(detections, color, pattern, clip)

    assert len(out) == 3
    a1 = out[(out["image_id"] == "a.jpg") & (out["garment_id"] == 1)].iloc[0]
    assert a1["category"] == "trousers"
    assert a1["dominant_color_name"] == "navy"
    assert a1["pattern_class"] == "patterned"
    assert a1["category_refined"] == "jeans"


def test_join_row_count_equals_detections_row_count() -> None:
    detections = _detections_frame(
        [_detection_row(f"img{i}.jpg", 0) for i in range(5)]
    )
    color = _color_frame(
        [_color_row(f"img{i}.jpg", 0) for i in range(5)]
    )
    pattern = _pattern_frame(
        [_pattern_row(f"img{i}.jpg", 0) for i in range(5)]
    )
    clip = _clip_frame([_clip_row(f"img{i}.jpg", 0) for i in range(5)])

    out = join.join_attribute_frames(detections, color, pattern, clip)

    assert len(out) == len(detections)


# --- left-join semantics --------------------------------------------------


def test_missing_downstream_row_yields_nan_attributes() -> None:
    detections = _detections_frame(
        [_detection_row("a.jpg", 0), _detection_row("b.jpg", 0)]
    )
    color = _color_frame([_color_row("a.jpg", 0)])
    pattern = _pattern_frame([_pattern_row("a.jpg", 0), _pattern_row("b.jpg", 0)])
    clip = _clip_frame([_clip_row("a.jpg", 0), _clip_row("b.jpg", 0)])

    out = join.join_attribute_frames(detections, color, pattern, clip)

    b_row = out[out["image_id"] == "b.jpg"].iloc[0]
    assert pd.isna(b_row["dominant_color_name"])
    assert pd.isna(b_row["dominant_r"])
    assert b_row["pattern_class"] == "plain"
    assert b_row["category_refined"] == "button-down shirt"


def test_orphan_downstream_rows_are_dropped() -> None:
    detections = _detections_frame([_detection_row("a.jpg", 0)])
    color = _color_frame(
        [_color_row("a.jpg", 0), _color_row("ghost.jpg", 0, name="yellow")]
    )
    pattern = _pattern_frame([_pattern_row("a.jpg", 0)])
    clip = _clip_frame([_clip_row("a.jpg", 0)])

    out = join.join_attribute_frames(detections, color, pattern, clip)

    assert len(out) == 1
    assert "ghost.jpg" not in set(out["image_id"])


# --- validation -----------------------------------------------------------


def test_missing_join_key_column_is_rejected() -> None:
    detections = _detections_frame([_detection_row("a.jpg", 0)])
    bad_color = pd.DataFrame(
        [{"image_id": "a.jpg", "dominant_r": 10, "dominant_g": 20, "dominant_b": 30}]
    )
    pattern = _pattern_frame([_pattern_row("a.jpg", 0)])
    clip = _clip_frame([_clip_row("a.jpg", 0)])

    with pytest.raises(ValueError, match="garment_id"):
        join.join_attribute_frames(detections, bad_color, pattern, clip)


def test_duplicate_join_keys_are_rejected() -> None:
    detections = _detections_frame(
        [_detection_row("a.jpg", 0), _detection_row("a.jpg", 0)]
    )
    color = _color_frame([_color_row("a.jpg", 0)])
    pattern = _pattern_frame([_pattern_row("a.jpg", 0)])
    clip = _clip_frame([_clip_row("a.jpg", 0)])

    with pytest.raises(ValueError, match="duplicate"):
        join.join_attribute_frames(detections, color, pattern, clip)


def test_numeric_image_id_does_not_break_string_join() -> None:
    """A purely-numeric filename like ``12345.jpg`` rendered without the
    extension would otherwise be parsed as int64 in one CSV and string in
    another, silently producing zero matches. ``_normalize_keys`` guards
    against this."""
    det_df = _detections_frame([_detection_row("12345", 0)])
    det_df["image_id"] = pd.Series(["12345"], dtype="object")
    color = _color_frame([_color_row("12345", 0)])

    det_df = join._normalize_keys(det_df)
    color = join._normalize_keys(color)
    pattern = join._normalize_keys(_pattern_frame([_pattern_row("12345", 0)]))
    clip = join._normalize_keys(_clip_frame([_clip_row("12345", 0)]))

    out = join.join_attribute_frames(det_df, color, pattern, clip)
    assert len(out) == 1
    assert out.iloc[0]["dominant_color_name"] == "red"


# --- end-to-end via CSV ---------------------------------------------------


def _write_csvs(tmp_path: Path) -> JoinConfig:
    det = _detections_frame(
        [_detection_row("a.jpg", 0), _detection_row("b.jpg", 0, category="skirt")]
    )
    col = _color_frame(
        [_color_row("a.jpg", 0, name="red"), _color_row("b.jpg", 0, name="green")]
    )
    pat = _pattern_frame(
        [_pattern_row("a.jpg", 0, cls="plain"), _pattern_row("b.jpg", 0, cls="subtle")]
    )
    clp = _clip_frame(
        [
            _clip_row("a.jpg", 0, refined="button-down shirt"),
            _clip_row("b.jpg", 0, category_yolo="skirt", refined="mini skirt"),
        ]
    )

    config = JoinConfig(
        detections_csv=tmp_path / "detections.csv",
        color_csv=tmp_path / "color_attributes.csv",
        pattern_csv=tmp_path / "pattern_attributes.csv",
        clip_csv=tmp_path / "clip_refinement.csv",
        output_csv=tmp_path / "out" / "yolo_fashion_attributes.csv",
    )
    det.to_csv(config.detections_csv, index=False)
    col.to_csv(config.color_csv, index=False)
    pat.to_csv(config.pattern_csv, index=False)
    clp.to_csv(config.clip_csv, index=False)
    return config


def test_run_join_writes_csv_with_canonical_schema(tmp_path: Path) -> None:
    config = _write_csvs(tmp_path)

    df = join.run_join(config)

    assert config.output_csv.exists()
    on_disk = pd.read_csv(config.output_csv)
    assert list(on_disk.columns) == list(join.CSV_COLUMNS)
    assert len(on_disk) == 2
    assert len(df) == 2


def test_run_join_is_idempotent(tmp_path: Path) -> None:
    config = _write_csvs(tmp_path)

    join.run_join(config)
    first = config.output_csv.read_bytes()
    join.run_join(config)
    second = config.output_csv.read_bytes()

    assert first == second


def test_inputs_are_not_modified(tmp_path: Path) -> None:
    config = _write_csvs(tmp_path)
    before = {
        path: path.read_bytes()
        for path in (
            config.detections_csv,
            config.color_csv,
            config.pattern_csv,
            config.clip_csv,
        )
    }

    join.run_join(config)

    for path, blob in before.items():
        assert path.read_bytes() == blob, f"{path.name} was modified"


def test_missing_downstream_coverage_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    det = _detections_frame(
        [_detection_row("a.jpg", 0), _detection_row("b.jpg", 0)]
    )
    col = _color_frame([_color_row("a.jpg", 0)])
    pat = _pattern_frame([_pattern_row("a.jpg", 0), _pattern_row("b.jpg", 0)])
    clp = _clip_frame([_clip_row("a.jpg", 0), _clip_row("b.jpg", 0)])

    config = JoinConfig(
        detections_csv=tmp_path / "detections.csv",
        color_csv=tmp_path / "color_attributes.csv",
        pattern_csv=tmp_path / "pattern_attributes.csv",
        clip_csv=tmp_path / "clip_refinement.csv",
        output_csv=tmp_path / "out.csv",
    )
    det.to_csv(config.detections_csv, index=False)
    col.to_csv(config.color_csv, index=False)
    pat.to_csv(config.pattern_csv, index=False)
    clp.to_csv(config.clip_csv, index=False)

    with caplog.at_level(logging.WARNING, logger="src.join"):
        join.run_join(config)

    assert any("color" in r.message and "missing" in r.message for r in caplog.records)


def test_dtypes_preserved_through_round_trip(tmp_path: Path) -> None:
    config = _write_csvs(tmp_path)
    df = join.run_join(config)

    assert df["garment_id"].dtype == np.int64
    assert str(df["image_id"].dtype) == "string"
    assert df["confidence"].dtype == np.float64
    assert df["laplacian_variance"].dtype == np.float64
