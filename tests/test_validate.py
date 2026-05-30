"""Tests for the manual validation app (src/validate.py).

The Tkinter GUI itself is exercised by hand; these tests pin down every
pure-function helper and the disk I/O the app performs so the validation
session output stays well-defined.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.utils import ValidateConfig, load_validate_config, project_root
from src.validate import (
    NO_CLOTHES_LABEL,
    PER_CATEGORY_FILENAME,
    PER_SUBCATEGORY_FILENAME,
    SUMMARY_FILENAME,
    UNCERTAIN_DISTRIBUTION_FILENAME,
    UNCERTAIN_LABEL,
    UNCERTAIN_USER_LABELS_FILENAME,
    VALIDATION_COLUMNS,
    VALIDATIONS_FILENAME,
    _append_validation,
    build_category_choices,
    build_subcategory_choices,
    build_validation_items,
    compute_accuracy,
    make_validation_dir,
    write_accuracy_summary,
)


# --- fixtures -------------------------------------------------------------


def _synthetic_attributes_df() -> pd.DataFrame:
    """Two parent categories with distinct refined sub-labels plus a NaN row."""
    return pd.DataFrame(
        [
            {
                "image_id": "a.jpg", "garment_id": 0, "category": "trousers",
                "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 10.0, "bbox_h": 10.0,
                "category_refined": "jeans",
            },
            {
                "image_id": "a.jpg", "garment_id": 1, "category": "trousers",
                "bbox_x": 1.0, "bbox_y": 1.0, "bbox_w": 5.0, "bbox_h": 5.0,
                "category_refined": "leggings",
            },
            {
                "image_id": "b.jpg", "garment_id": 0, "category": "skirt",
                "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 10.0, "bbox_h": 10.0,
                "category_refined": "miniskirt",
            },
            {
                "image_id": "c.jpg", "garment_id": 0, "category": "skirt",
                "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 10.0, "bbox_h": 10.0,
                "category_refined": "uncertain",
            },
            {
                "image_id": "d.jpg", "garment_id": 0, "category": None,
                "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 10.0, "bbox_h": 10.0,
                "category_refined": None,
            },
        ]
    )


# --- pure helpers ---------------------------------------------------------


def test_build_category_choices_prepends_no_clothes_then_sorted_parents():
    df = _synthetic_attributes_df()
    assert build_category_choices(df) == [NO_CLOTHES_LABEL, "skirt", "trousers"]


def test_build_subcategory_choices_is_parent_conditioned():
    df = _synthetic_attributes_df()
    sub = build_subcategory_choices(df)
    assert sub["trousers"] == ["jeans", "leggings"]
    assert sub["skirt"] == ["miniskirt", "uncertain"]
    assert sub[NO_CLOTHES_LABEL] == [NO_CLOTHES_LABEL]


def test_make_validation_dir_uses_iso_date(tmp_path: Path):
    out = make_validation_dir(tmp_path, today=date(2026, 5, 30))
    assert out == tmp_path / "2026-05-30-val"
    assert out.is_dir()


def test_make_validation_dir_idempotent(tmp_path: Path):
    a = make_validation_dir(tmp_path, today=date(2026, 5, 30))
    b = make_validation_dir(tmp_path, today=date(2026, 5, 30))
    assert a == b and a.is_dir()


# --- item construction & randomisation -----------------------------------


def test_build_validation_items_drops_rows_without_parent_category():
    items = build_validation_items(_synthetic_attributes_df(), random_seed=0, max_items=None)
    assert len(items) == 4  # the NaN-category row is excluded
    for item in items:
        assert item["model_category"] in {"trousers", "skirt"}


def test_build_validation_items_seeded_is_reproducible():
    df = _synthetic_attributes_df()
    a = build_validation_items(df, random_seed=42, max_items=None)
    b = build_validation_items(df, random_seed=42, max_items=None)
    assert [(i["image_id"], i["garment_id"]) for i in a] == [
        (i["image_id"], i["garment_id"]) for i in b
    ]


def test_build_validation_items_no_seed_still_runs():
    """random_seed=None must not raise — it just means OS-seeded randomness."""
    df = _synthetic_attributes_df()
    items = build_validation_items(df, random_seed=None, max_items=None)
    assert len(items) == 4


def test_build_validation_items_respects_max_items():
    df = _synthetic_attributes_df()
    items = build_validation_items(df, random_seed=42, max_items=2)
    assert len(items) == 2


# --- accuracy maths -------------------------------------------------------


def test_compute_accuracy_excludes_uncertain_from_subcategory_denominator():
    df = pd.DataFrame(
        [
            # match — both correct
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": "jeans", "user_subcategory": "jeans"},
            # category miss, subcategory miss
            {"model_category": "trousers", "user_category": "skirt",
             "model_subcategory": "leggings", "user_subcategory": "miniskirt"},
            # uncertain model — must NOT count toward subcategory accuracy
            {"model_category": "skirt", "user_category": "skirt",
             "model_subcategory": UNCERTAIN_LABEL, "user_subcategory": "miniskirt"},
            # another uncertain — user even labelled it as uncertain
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": UNCERTAIN_LABEL, "user_subcategory": "jeans"},
        ]
    )
    result = compute_accuracy(df)
    row = result["summary"].iloc[0]

    assert int(row["items_validated"]) == 4
    # category covers all 4 rows: 3 correct (rows 0, 2, 3)
    assert row["category_accuracy"] == pytest.approx(3 / 4)
    # only the 2 non-uncertain rows count: 1 correct, 1 wrong → 0.5
    assert int(row["subcategory_items_evaluated"]) == 2
    assert row["subcategory_accuracy_excl_uncertain"] == pytest.approx(0.5)
    # both-correct over the 2 non-uncertain rows
    assert row["both_correct_accuracy_excl_uncertain"] == pytest.approx(0.5)
    assert int(row["uncertain_items"]) == 2
    assert row["uncertain_rate"] == pytest.approx(0.5)


def test_compute_accuracy_per_subcategory_drops_uncertain():
    df = pd.DataFrame(
        [
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": "jeans", "user_subcategory": "jeans"},
            {"model_category": "skirt", "user_category": "skirt",
             "model_subcategory": UNCERTAIN_LABEL, "user_subcategory": "miniskirt"},
        ]
    )
    per_sub = compute_accuracy(df)["per_subcategory"]
    assert set(per_sub["model_subcategory"]) == {"jeans"}
    assert UNCERTAIN_LABEL not in set(per_sub["model_subcategory"])


def test_compute_accuracy_uncertain_distribution_and_user_labels():
    df = pd.DataFrame(
        [
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": "jeans", "user_subcategory": "jeans"},
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": UNCERTAIN_LABEL, "user_subcategory": "jeans"},
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": UNCERTAIN_LABEL, "user_subcategory": "leggings"},
            {"model_category": "skirt", "user_category": "skirt",
             "model_subcategory": UNCERTAIN_LABEL, "user_subcategory": "miniskirt"},
            {"model_category": "skirt", "user_category": "skirt",
             "model_subcategory": "miniskirt", "user_subcategory": "miniskirt"},
        ]
    )
    result = compute_accuracy(df)

    dist = result["uncertain_distribution"].set_index("model_category")
    assert int(dist.loc["trousers", "n_total"]) == 3
    assert int(dist.loc["trousers", "n_uncertain"]) == 2
    assert dist.loc["trousers", "uncertain_rate"] == pytest.approx(2 / 3)
    assert int(dist.loc["skirt", "n_uncertain"]) == 1
    assert dist.loc["skirt", "uncertain_rate"] == pytest.approx(0.5)

    labels = result["uncertain_user_labels"]
    # only rows where model said uncertain appear
    assert len(labels) == 3
    trousers_labels = labels[labels["model_category"] == "trousers"]
    assert set(trousers_labels["user_subcategory"]) == {"jeans", "leggings"}
    # all counts are 1 in this sample
    assert (labels["count"] == 1).all()


def test_compute_accuracy_no_clothes_counts_as_category_miss():
    df = pd.DataFrame(
        [
            # model thinks trousers — operator says no garment at all
            {"model_category": "trousers", "user_category": NO_CLOTHES_LABEL,
             "model_subcategory": "jeans", "user_subcategory": NO_CLOTHES_LABEL},
            # control row
            {"model_category": "trousers", "user_category": "trousers",
             "model_subcategory": "jeans", "user_subcategory": "jeans"},
        ]
    )
    result = compute_accuracy(df)
    summary = result["summary"].iloc[0]
    assert summary["category_accuracy"] == pytest.approx(0.5)
    # subcategory: row 0 wrong (no_clothes != jeans), row 1 correct → 0.5
    assert summary["subcategory_accuracy_excl_uncertain"] == pytest.approx(0.5)


def test_compute_accuracy_empty_input_returns_empty_frames():
    result = compute_accuracy(pd.DataFrame())
    for key in (
        "summary", "per_category", "per_subcategory",
        "uncertain_distribution", "uncertain_user_labels",
    ):
        assert result[key].empty


# --- streaming validations to disk ---------------------------------------


def test_append_validation_writes_header_once(tmp_path: Path):
    csv = tmp_path / "v.csv"
    row = {col: "x" for col in VALIDATION_COLUMNS}
    _append_validation(csv, row)
    _append_validation(csv, row)
    text = csv.read_text(encoding="utf-8")
    assert text.count("image_id,garment_id") == 1
    assert len(text.strip().splitlines()) == 3  # header + 2 rows


def test_append_validation_preserves_column_order(tmp_path: Path):
    csv = tmp_path / "v.csv"
    _append_validation(csv, {col: i for i, col in enumerate(VALIDATION_COLUMNS)})
    first_line = csv.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == ",".join(VALIDATION_COLUMNS)


# --- accuracy summary I/O -------------------------------------------------


def test_write_accuracy_summary_creates_all_five_csvs(tmp_path: Path):
    val_csv = tmp_path / VALIDATIONS_FILENAME
    pd.DataFrame(
        [
            {
                "image_id": "a.jpg", "garment_id": 0,
                "model_category": "trousers", "model_subcategory": "jeans",
                "user_category": "trousers", "user_subcategory": "jeans",
                "category_correct": 1, "subcategory_correct": 1,
                "timestamp": "2026-05-30T10:00:00",
            },
            {
                "image_id": "b.jpg", "garment_id": 0,
                "model_category": "skirt", "model_subcategory": UNCERTAIN_LABEL,
                "user_category": "skirt", "user_subcategory": "miniskirt",
                "category_correct": 1, "subcategory_correct": "",
                "timestamp": "2026-05-30T10:01:00",
            },
        ]
    ).to_csv(val_csv, index=False)

    paths = write_accuracy_summary(val_csv, tmp_path)
    assert paths is not None
    expected_keys = {
        "summary", "per_category", "per_subcategory",
        "uncertain_distribution", "uncertain_user_labels",
    }
    assert set(paths) == expected_keys
    for key in expected_keys:
        assert paths[key].exists() and paths[key].stat().st_size > 0

    summary = pd.read_csv(tmp_path / SUMMARY_FILENAME).iloc[0]
    assert int(summary["items_validated"]) == 2
    assert summary["category_accuracy"] == pytest.approx(1.0)
    # only the non-uncertain row counts toward subcategory accuracy
    assert int(summary["subcategory_items_evaluated"]) == 1
    assert summary["subcategory_accuracy_excl_uncertain"] == pytest.approx(1.0)
    assert int(summary["uncertain_items"]) == 1

    per_cat = pd.read_csv(tmp_path / PER_CATEGORY_FILENAME)
    assert set(per_cat["model_category"]) == {"trousers", "skirt"}

    per_sub = pd.read_csv(tmp_path / PER_SUBCATEGORY_FILENAME)
    # uncertain row is excluded from per-subcategory accuracy
    assert set(per_sub["model_subcategory"]) == {"jeans"}

    dist = pd.read_csv(tmp_path / UNCERTAIN_DISTRIBUTION_FILENAME).set_index("model_category")
    assert int(dist.loc["skirt", "n_uncertain"]) == 1
    assert int(dist.loc["trousers", "n_uncertain"]) == 0

    user_labels = pd.read_csv(tmp_path / UNCERTAIN_USER_LABELS_FILENAME)
    assert len(user_labels) == 1
    assert user_labels.iloc[0]["model_category"] == "skirt"
    assert user_labels.iloc[0]["user_subcategory"] == "miniskirt"


def test_write_accuracy_summary_skips_missing_file(tmp_path: Path):
    result = write_accuracy_summary(tmp_path / "nope.csv", tmp_path)
    assert result is None
    assert not (tmp_path / SUMMARY_FILENAME).exists()


def test_write_accuracy_summary_skips_empty_file(tmp_path: Path):
    val_csv = tmp_path / VALIDATIONS_FILENAME
    pd.DataFrame(columns=list(VALIDATION_COLUMNS)).to_csv(val_csv, index=False)
    result = write_accuracy_summary(val_csv, tmp_path)
    assert result is None


# --- config loader --------------------------------------------------------


def test_load_validate_config_real_yaml_parses():
    cfg = load_validate_config(project_root() / "config" / "validate.yaml")
    assert isinstance(cfg, ValidateConfig)
    assert cfg.attributes_csv.name == "yolo_fashion_attributes.csv"
    assert cfg.images_dir.name == "sample_images"
    assert cfg.output_root.name == "validations"
    assert cfg.display_max_dim >= 64


def test_load_validate_config_accepts_null_seed(tmp_path: Path):
    yaml_path = tmp_path / "v.yaml"
    yaml_path.write_text(
        "attributes_csv: a.csv\n"
        "images_dir: imgs\n"
        "output_root: out\n"
        "random_seed: null\n"
        "max_items: null\n"
        "display_max_dim: 720\n",
        encoding="utf-8",
    )
    cfg = load_validate_config(yaml_path)
    assert cfg.random_seed is None
    assert cfg.max_items is None
