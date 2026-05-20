"""Tests for Pipeline 1 Step 2C: CLIP zero-shot garment refinement.

The real CLIP model is never loaded in this suite. We inject a fake
``InferFn`` so the tests run with no network access and no weights on disk,
matching the no-data-dir / no-network policy in AGENTS.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import clip_refine
from src.utils import ClipRefineConfig, load_clip_refine_config


# --- helpers --------------------------------------------------------------


_STARTER_TAXONOMY: dict[str, tuple[str, ...]] = {
    "short sleeve top": (
        "t-shirt",
        "polo shirt",
        "blouse",
        "tank top",
        "crop top",
        "graphic tee",
    ),
    "long sleeve top": (
        "sweater",
        "blazer",
        "hoodie",
        "button-down shirt",
        "turtleneck",
        "cardigan",
        "sweatshirt",
    ),
    "short sleeve outwear": ("cropped jacket", "light overshirt"),
    "long sleeve outwear": (
        "trench coat",
        "puffer jacket",
        "leather jacket",
        "wool coat",
        "denim jacket",
        "bomber jacket",
    ),
    "vest": ("tailored vest", "puffer vest", "knit vest"),
    "sling": ("halter top", "one-shoulder top", "cami"),
    "shorts": (
        "denim shorts",
        "tailored shorts",
        "athletic shorts",
        "bermuda shorts",
    ),
    "trousers": (
        "jeans",
        "tailored trousers",
        "wide-leg pants",
        "leggings",
        "cargo pants",
        "sweatpants",
    ),
    "skirt": (
        "mini skirt",
        "midi skirt",
        "maxi skirt",
        "pleated skirt",
        "denim skirt",
    ),
    "short sleeve dress": ("t-shirt dress", "mini dress", "wrap dress"),
    "long sleeve dress": ("sweater dress", "shirt dress", "maxi dress"),
    "vest dress": ("slip dress", "pinafore", "jumper dress"),
    "sling dress": ("halter dress", "one-shoulder dress", "strappy dress"),
}


def _write_solid(
    path: Path,
    color: tuple[int, int, int] = (180, 180, 180),
    size: tuple[int, int] = (200, 200),
) -> None:
    width, height = size
    image = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _detection_row(
    image_id: str,
    category: str,
    bbox: tuple[float, float, float, float] = (10, 10, 180, 180),
    garment_id: int = 0,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "garment_id": garment_id,
        "category": category,
        "confidence": 0.9,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
    }


def _write_detections(path: Path, records: list[dict[str, Any]]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def _make_config(
    tmp_path: Path,
    taxonomy: dict[str, tuple[str, ...]] | None = None,
    threshold: float = 0.4,
    batch_size: int = 4,
) -> ClipRefineConfig:
    return ClipRefineConfig(
        detections_csv=tmp_path / "detections.csv",
        images_dir=tmp_path / "images",
        output_csv=tmp_path / "clip_refinement.csv",
        center_crop_fraction=0.6,
        model_id="dummy/clip",
        model_cache_dir=tmp_path / "models",
        prompt_template="a photo of a {label}",
        threshold=threshold,
        batch_size=batch_size,
        taxonomy=taxonomy if taxonomy is not None else _STARTER_TAXONOMY,
    )


def _uniform_infer_fn(
    images: list[Image.Image], labels: list[str]
) -> np.ndarray:
    """Equal probability across all labels — top-label prob = 1/L."""
    n = len(images)
    L = len(labels)
    return np.full((n, L), 1.0 / L, dtype=np.float64)


def _top_label_infer_fn(top_prob: float) -> clip_refine.InferFn:
    """Return an infer_fn that pins the first label at ``top_prob`` and
    distributes the remainder uniformly across the rest."""

    def fn(images: list[Image.Image], labels: list[str]) -> np.ndarray:
        n = len(images)
        L = len(labels)
        if L < 2:
            raise AssertionError("test infer_fn requires at least 2 labels")
        rest = (1.0 - top_prob) / (L - 1)
        probs = np.full((n, L), rest, dtype=np.float64)
        probs[:, 0] = top_prob
        return probs

    return fn


# --- normalize_yolo_category ----------------------------------------------


@pytest.mark.parametrize(
    "yolo_name,expected",
    [
        ("short_sleeved_shirt", "short sleeve top"),
        ("long_sleeved_shirt", "long sleeve top"),
        ("short_sleeved_outwear", "short sleeve outwear"),
        ("long_sleeved_outwear", "long sleeve outwear"),
        ("vest", "vest"),
        ("sling", "sling"),
        ("shorts", "shorts"),
        ("trousers", "trousers"),
        ("skirt", "skirt"),
        ("short_sleeved_dress", "short sleeve dress"),
        ("long_sleeved_dress", "long sleeve dress"),
        ("vest_dress", "vest dress"),
        ("sling_dress", "sling dress"),
    ],
)
def test_normalize_covers_all_deepfashion2_classes(
    yolo_name: str, expected: str
) -> None:
    assert clip_refine.normalize_yolo_category(yolo_name) == expected
    assert expected in _STARTER_TAXONOMY


def test_normalize_is_idempotent_on_human_form() -> None:
    for key in _STARTER_TAXONOMY:
        assert clip_refine.normalize_yolo_category(key) == key


# --- select_refined_label -------------------------------------------------


def test_threshold_below_returns_uncertain() -> None:
    scores = {"t-shirt": 0.39, "blouse": 0.31, "polo shirt": 0.30}
    label, conf = clip_refine.select_refined_label(scores, threshold=0.4)
    assert label == clip_refine.UNCERTAIN_LABEL
    assert conf == pytest.approx(0.39)


def test_threshold_at_or_above_returns_top_label() -> None:
    scores = {"t-shirt": 0.41, "blouse": 0.30, "polo shirt": 0.29}
    label, conf = clip_refine.select_refined_label(scores, threshold=0.4)
    assert label == "t-shirt"
    assert conf == pytest.approx(0.41)


def test_threshold_exact_boundary_returns_top_label() -> None:
    scores = {"t-shirt": 0.4, "blouse": 0.6}
    label, _conf = clip_refine.select_refined_label(scores, threshold=0.6)
    assert label == "blouse"


def test_empty_scores_returns_uncertain() -> None:
    label, conf = clip_refine.select_refined_label({}, threshold=0.4)
    assert label == clip_refine.UNCERTAIN_LABEL
    assert conf == 0.0


# --- crop_for_clip --------------------------------------------------------


def test_crop_for_clip_returns_pil_rgb(tmp_path: Path) -> None:
    img_path = tmp_path / "a.png"
    _write_solid(img_path, color=(50, 100, 200))
    pil = clip_refine.crop_for_clip(
        img_path, (10, 10, 100, 100), 0.6, "a.png", 0
    )
    assert pil is not None
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"


def test_crop_for_clip_missing_image_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.clip_refine"):
        pil = clip_refine.crop_for_clip(
            tmp_path / "missing.png", (10, 10, 50, 50), 0.6, "missing.png", 0
        )
    assert pil is None
    assert any("missing.png" in r.message for r in caplog.records)


def test_crop_for_clip_bbox_outside_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    img_path = tmp_path / "a.png"
    _write_solid(img_path, size=(100, 100))
    with caplog.at_level(logging.WARNING, logger="src.clip_refine"):
        pil = clip_refine.crop_for_clip(
            img_path, (500, 500, 50, 50), 0.6, "a.png", 0
        )
    assert pil is None
    assert any("zero area" in r.message for r in caplog.records)


# --- taxonomy lookup: every parent returns a member of its own list -------


@pytest.mark.parametrize("parent", list(_STARTER_TAXONOMY.keys()))
def test_taxonomy_lookup_returns_sub_label_or_uncertain(
    tmp_path: Path, parent: str
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv, [_detection_row("a.png", category=parent)]
    )

    df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    assert len(df) == 1
    refined = df["category_refined"].iloc[0]
    allowed = set(_STARTER_TAXONOMY[parent]) | {clip_refine.UNCERTAIN_LABEL}
    assert refined in allowed, f"{refined!r} not in {allowed!r}"


def test_taxonomy_lookup_accepts_yolo_underscored_names(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="short_sleeved_shirt")],
    )

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.9)
    )

    assert len(df) == 1
    assert df["category_yolo"].iloc[0] == "short_sleeved_shirt"
    assert df["category_refined"].iloc[0] == _STARTER_TAXONOMY["short sleeve top"][0]


# --- threshold logic end-to-end -------------------------------------------


def test_below_threshold_writes_uncertain(tmp_path: Path) -> None:
    config = _make_config(tmp_path, threshold=0.4)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="short_sleeved_shirt")],
    )

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.39)
    )

    assert df["category_refined"].iloc[0] == clip_refine.UNCERTAIN_LABEL
    assert df["refined_confidence"].iloc[0] == pytest.approx(0.39)


def test_above_threshold_writes_top_label(tmp_path: Path) -> None:
    config = _make_config(tmp_path, threshold=0.4)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="short_sleeved_shirt")],
    )

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.41)
    )

    sub_labels = _STARTER_TAXONOMY["short sleeve top"]
    assert df["category_refined"].iloc[0] == sub_labels[0]
    assert df["refined_confidence"].iloc[0] == pytest.approx(0.41)


# --- unknown parent / sparse taxonomy passthrough -------------------------


def test_unknown_parent_passes_through_unrefined(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="space_suit")],
    )

    df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    assert len(df) == 1
    assert df["category_yolo"].iloc[0] == "space_suit"
    assert df["category_refined"].iloc[0] == "space_suit"
    assert df["refined_confidence"].iloc[0] == 1.0
    parsed = json.loads(df["all_scores"].iloc[0])
    assert parsed == {"space_suit": 1.0}


def test_single_sublabel_parent_passes_through_unrefined(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path, taxonomy={"shorts": ("denim shorts",)}
    )
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv, [_detection_row("a.png", category="shorts")]
    )

    df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    assert df["category_refined"].iloc[0] == "shorts"
    assert df["refined_confidence"].iloc[0] == 1.0


# --- JSON validity & probability sum --------------------------------------


def test_all_scores_is_valid_json_summing_to_one(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_solid(config.images_dir / "b.png")
    _write_detections(
        config.detections_csv,
        [
            _detection_row("a.png", category="short_sleeved_shirt"),
            _detection_row("b.png", category="trousers"),
        ],
    )

    df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    for raw in df["all_scores"]:
        scores = json.loads(raw)
        assert isinstance(scores, dict)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


def test_refined_label_is_member_of_sub_label_list(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    for parent in _STARTER_TAXONOMY:
        _write_solid(config.images_dir / f"{parent.replace(' ', '_')}.png")
    records: list[dict[str, Any]] = []
    for i, parent in enumerate(_STARTER_TAXONOMY):
        records.append(
            _detection_row(
                f"{parent.replace(' ', '_')}.png",
                category=parent,
                garment_id=i,
            )
        )
    _write_detections(config.detections_csv, records)

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.99)
    )

    for _, row in df.iterrows():
        parent_key = clip_refine.normalize_yolo_category(row["category_yolo"])
        assert row["category_refined"] in _STARTER_TAXONOMY[parent_key]


# --- output schema --------------------------------------------------------


def test_csv_schema_columns_and_dtypes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="short_sleeved_shirt")],
    )

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.8)
    )

    assert list(df.columns) == list(clip_refine.CSV_COLUMNS)
    assert df["garment_id"].dtype == np.int64
    assert df["refined_confidence"].dtype == np.float64
    assert str(df["image_id"].dtype) == "string"
    assert str(df["category_yolo"].dtype) == "string"
    assert str(df["category_refined"].dtype) == "string"
    assert str(df["all_scores"].dtype) == "string"


def test_confidence_in_unit_interval(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="short_sleeved_shirt")],
    )

    df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    for v in df["refined_confidence"]:
        assert 0.0 <= v <= 1.0


# --- determinism & idempotency --------------------------------------------


def test_run_is_idempotent(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_solid(config.images_dir / "b.png")
    _write_detections(
        config.detections_csv,
        [
            _detection_row(
                "a.png", category="short_sleeved_shirt", garment_id=0
            ),
            _detection_row("b.png", category="trousers", garment_id=0),
        ],
    )

    clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.8)
    )
    first = config.output_csv.read_bytes()

    clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.8)
    )
    second = config.output_csv.read_bytes()

    assert first == second


# --- edge cases: logged, not crashed --------------------------------------


def test_missing_image_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_detections(
        config.detections_csv,
        [_detection_row("missing.png", category="short_sleeved_shirt")],
    )

    with caplog.at_level(logging.WARNING, logger="src.clip_refine"):
        df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    assert len(df) == 0
    assert config.output_csv.exists()
    assert any("missing.png" in r.message for r in caplog.records)


def test_bbox_past_image_bounds_does_not_crash(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png", size=(100, 100))
    _write_detections(
        config.detections_csv,
        [
            _detection_row(
                "a.png",
                category="short_sleeved_shirt",
                bbox=(50, 50, 200, 200),
            )
        ],
    )

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.9)
    )

    assert len(df) == 1
    assert df["category_yolo"].iloc[0] == "short_sleeved_shirt"


def test_corrupt_image_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    bad = config.images_dir / "bad.png"
    bad.write_bytes(b"not a real image")
    _write_detections(
        config.detections_csv,
        [_detection_row("bad.png", category="short_sleeved_shirt")],
    )

    with caplog.at_level(logging.WARNING, logger="src.clip_refine"):
        df = clip_refine.run_clip_refinement(config, infer_fn=_uniform_infer_fn)

    assert len(df) == 0
    assert any("bad.png" in r.message for r in caplog.records)


# --- upstream protection --------------------------------------------------


def test_detections_csv_is_not_modified(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [_detection_row("a.png", category="short_sleeved_shirt")],
    )
    before = config.detections_csv.read_bytes()

    clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.9)
    )

    assert config.detections_csv.read_bytes() == before


def test_row_count_matches_input_minus_skipped(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.images_dir.mkdir()
    _write_solid(config.images_dir / "a.png")
    _write_detections(
        config.detections_csv,
        [
            _detection_row(
                "a.png", category="short_sleeved_shirt", garment_id=0
            ),
            _detection_row(
                "missing.png", category="short_sleeved_shirt", garment_id=0
            ),
        ],
    )

    df = clip_refine.run_clip_refinement(
        config, infer_fn=_top_label_infer_fn(top_prob=0.9)
    )
    on_disk = pd.read_csv(config.output_csv)

    assert len(df) == 1
    assert len(on_disk) == 1


# --- production config sanity ---------------------------------------------


def test_starter_yaml_loads_and_covers_every_yolo_class() -> None:
    """The shipped config must parse cleanly and have an entry for every
    DeepFashion2 class the YOLO detector emits."""
    config = load_clip_refine_config()
    yolo_classes = (
        "short_sleeved_shirt",
        "long_sleeved_shirt",
        "short_sleeved_outwear",
        "long_sleeved_outwear",
        "vest",
        "sling",
        "shorts",
        "trousers",
        "skirt",
        "short_sleeved_dress",
        "long_sleeved_dress",
        "vest_dress",
        "sling_dress",
    )
    for yolo in yolo_classes:
        key = clip_refine.normalize_yolo_category(yolo)
        assert key in config.taxonomy, f"taxonomy missing entry for {yolo}"
        assert len(config.taxonomy[key]) >= 2
