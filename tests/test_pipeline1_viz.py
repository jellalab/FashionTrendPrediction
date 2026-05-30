"""Tests for Pipeline 1 visualisations (src/pipeline1_viz.py).

Synthetic CSV + fake accepted/rejected image folders are built in a tmp dir
so the suite runs with no network access and no access to ``data/``. The
real palette YAML is reused so the palette-lookup test exercises every
entry as the spec requires.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import pipeline1_viz
from src.pipeline1_viz import (
    PATTERN_ORDER,
    PLOT_NAMES,
    UNCERTAIN_LABEL,
    apply_style,
    count_images,
    is_pale,
    palette_rgb_map,
    parents_above_threshold,
    plot_acceptance_rejection,
    plot_clip_uncertainty_per_parent,
    plot_dominant_color_distribution,
    plot_garment_category_distribution,
    plot_pattern_class_distribution,
    plot_refined_garment_type_per_parent,
    run_pipeline1_viz,
)
from src.utils import (
    Pipeline1VizConfig,
    load_color_config,
    project_root,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --- fixtures -------------------------------------------------------------


def _palette_path() -> Path:
    return project_root() / "config" / "color.yaml"


def _synthetic_df(
    rare_parent_count: int = 3,
    common_parent_counts: tuple[int, int] = (10, 8),
    uncertainty_pcts: tuple[float, float] = (0.4, 0.0),
) -> pd.DataFrame:
    """Build a small ``yolo_fashion_attributes.csv``-shaped DataFrame.

    Two common parents (above the default threshold), one rare parent (≤5
    rows). Real palette names are used so the dominant-colour plot exercises
    the palette lookup. Uncertainty rates per parent let plot 6 produce a
    deterministic ordering for assertions.
    """
    palette_cycle = [
        "black", "white", "gray", "beige", "cream",
        "brown", "navy", "blue", "red", "green",
    ]
    pattern_cycle = list(PATTERN_ORDER)

    rows = []
    garment_id = 0

    # common parent 1: trousers — high uncertainty
    n1 = common_parent_counts[0]
    uncertain_n1 = int(round(n1 * uncertainty_pcts[0]))
    for i in range(n1):
        refined = UNCERTAIN_LABEL if i < uncertain_n1 else f"sublabel_t{i % 3}"
        rows.append({
            "image_id": f"img_t_{i}.jpg",
            "garment_id": garment_id,
            "category": "trousers",
            "dominant_r": 40, "dominant_g": 80, "dominant_b": 180,
            "dominant_color_name": palette_cycle[i % len(palette_cycle)],
            "laplacian_variance": 100.0 + i,
            "pattern_class": pattern_cycle[i % len(pattern_cycle)],
            "category_yolo": "trousers",
            "category_refined": refined,
            "refined_confidence": 0.6,
        })
        garment_id += 1

    # common parent 2: long_sleeved_shirt — no uncertainty
    n2 = common_parent_counts[1]
    for i in range(n2):
        rows.append({
            "image_id": f"img_s_{i}.jpg",
            "garment_id": garment_id,
            "category": "long_sleeved_shirt",
            "dominant_r": 59, "dominant_g": 65, "dominant_b": 57,
            "dominant_color_name": palette_cycle[(i + 3) % len(palette_cycle)],
            "laplacian_variance": 200.0 + i,
            "pattern_class": pattern_cycle[(i + 1) % len(pattern_cycle)],
            "category_yolo": "long_sleeved_shirt",
            "category_refined": f"sublabel_s{i % 2}",
            "refined_confidence": 0.7,
        })
        garment_id += 1

    # rare parent: skirt — only N rows (below threshold)
    for i in range(rare_parent_count):
        rows.append({
            "image_id": f"img_k_{i}.jpg",
            "garment_id": garment_id,
            "category": "skirt",
            "dominant_r": 200, "dominant_g": 30, "dominant_b": 30,
            "dominant_color_name": "red",
            "laplacian_variance": 50.0 + i,
            "pattern_class": "plain",
            "category_yolo": "skirt",
            "category_refined": "miniskirt",
            "refined_confidence": 0.55,
        })
        garment_id += 1

    return pd.DataFrame(rows)


def _make_image_dir(parent: Path, n_files: int, suffix: str = ".jpg") -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (parent / f"img_{i}{suffix}").write_bytes(b"\x00")
    # control files that must be ignored by count_images
    (parent / ".DS_Store").write_bytes(b"\x00")
    (parent / "notes.txt").write_text("ignored", encoding="utf-8")
    return parent


def _make_config(tmp_path: Path, df: pd.DataFrame, min_detections: int = 5) -> Pipeline1VizConfig:
    input_csv = tmp_path / "yolo_fashion_attributes.csv"
    df.to_csv(input_csv, index=False)
    return Pipeline1VizConfig(
        input_csv=input_csv,
        accepted_dir=_make_image_dir(tmp_path / "accepted", 7),
        rejected_dir=_make_image_dir(tmp_path / "rejected", 3),
        output_dir=tmp_path / "figures",
        color_config=_palette_path(),
        min_detections_per_parent=min_detections,
        suppress=(),
    )


# --- acceptance criteria --------------------------------------------------


def test_run_produces_all_six_files(tmp_path: Path) -> None:
    """All six PNG files are produced on a single run with no errors."""
    df = _synthetic_df()
    config = _make_config(tmp_path, df)

    written = run_pipeline1_viz(config)

    expected = {config.output_dir / f"{n}.png" for n in PLOT_NAMES}
    assert set(written) == expected
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0


def test_every_png_starts_with_valid_signature(tmp_path: Path) -> None:
    """Basic header check: every output is a real PNG, not an empty / corrupt file."""
    config = _make_config(tmp_path, _synthetic_df())
    written = run_pipeline1_viz(config)
    for path in written:
        with open(path, "rb") as f:
            assert f.read(8) == PNG_SIGNATURE


def test_palette_lookup_covers_every_palette_name() -> None:
    """The colour-lookup helper maps every palette name to a valid RGB triple."""
    color_cfg = load_color_config(_palette_path())
    palette_map = palette_rgb_map(color_cfg.palette)

    # one entry per palette item, no drops
    assert len(palette_map) == len(color_cfg.palette)
    for entry in color_cfg.palette:
        assert entry.name in palette_map
        r, g, b = palette_map[entry.name]
        assert all(isinstance(c, int) for c in (r, g, b))
        assert all(0 <= c <= 255 for c in (r, g, b))


def test_parents_above_threshold_skips_rare(tmp_path: Path) -> None:
    """Rule: skip parents with <= 5 detections in plot 5."""
    df = _synthetic_df(rare_parent_count=3, common_parent_counts=(10, 8))
    kept = parents_above_threshold(df, min_count=5)

    assert "trousers" in kept            # 10 > 5
    assert "long_sleeved_shirt" in kept  # 8 > 5
    assert "skirt" not in kept           # 3 <= 5

    # And the rule is honoured end-to-end via the plot orchestrator:
    config = _make_config(tmp_path, df, min_detections=5)
    run_pipeline1_viz(config)
    plot5 = config.output_dir / "refined_garment_type_per_parent.png"
    assert plot5.exists() and plot5.stat().st_size > 0


# --- determinism and re-run behaviour -------------------------------------


def test_rerun_overwrites_cleanly(tmp_path: Path) -> None:
    """Re-running overwrites cleanly and yields visually identical bytes."""
    config = _make_config(tmp_path, _synthetic_df())
    first = run_pipeline1_viz(config)
    first_bytes = {p.name: p.read_bytes() for p in first}

    second = run_pipeline1_viz(config)
    for path in second:
        assert path.read_bytes() == first_bytes[path.name], (
            f"{path.name} bytes differ on re-run — plot is non-deterministic"
        )


# --- helper coverage ------------------------------------------------------


def test_count_images_ignores_hidden_and_non_image_files(tmp_path: Path) -> None:
    folder = _make_image_dir(tmp_path / "imgs", n_files=4, suffix=".png")
    assert count_images(folder) == 4


def test_count_images_handles_missing_directory(tmp_path: Path) -> None:
    assert count_images(tmp_path / "nope") == 0


def test_is_pale_for_white_and_cream() -> None:
    # spec-named examples: white and cream both need an outline against #FAFAFA
    assert is_pale((255, 255, 255)) is True
    assert is_pale((250, 240, 225)) is True       # cream
    # mid- and dark-toned palette entries do not need outlining
    assert is_pale((220, 210, 190)) is False      # beige (luminance ~0.83)
    assert is_pale((0, 0, 0)) is False
    assert is_pale((40, 80, 180)) is False        # blue


def test_pattern_order_constant_matches_spec() -> None:
    assert PATTERN_ORDER == ("plain", "subtle", "patterned")


# --- suppression flag -----------------------------------------------------


def test_suppress_skips_named_plots(tmp_path: Path) -> None:
    config = _make_config(tmp_path, _synthetic_df())
    config = Pipeline1VizConfig(
        input_csv=config.input_csv,
        accepted_dir=config.accepted_dir,
        rejected_dir=config.rejected_dir,
        output_dir=config.output_dir,
        color_config=config.color_config,
        min_detections_per_parent=config.min_detections_per_parent,
        suppress=("clip_uncertainty_per_parent",),
    )
    written = run_pipeline1_viz(config)

    names = {p.name for p in written}
    assert "clip_uncertainty_per_parent.png" not in names
    assert "acceptance_rejection.png" in names
    assert (config.output_dir / "clip_uncertainty_per_parent.png").exists() is False


# --- individual plot functions stay callable -----------------------------


def test_individual_plot_functions(tmp_path: Path) -> None:
    """Sanity check that each plot function can run in isolation."""
    apply_style()
    df = _synthetic_df()
    palette_map = palette_rgb_map(load_color_config(_palette_path()).palette)
    out = tmp_path / "out"
    out.mkdir()

    plot_acceptance_rejection(289, 108, out / "1.png")
    plot_garment_category_distribution(df, out / "2.png")
    plot_dominant_color_distribution(df, palette_map, out / "3.png")
    plot_pattern_class_distribution(df, out / "4.png")
    plot_refined_garment_type_per_parent(df, 5, out / "5.png")
    plot_clip_uncertainty_per_parent(df, out / "6.png")

    for i in range(1, 7):
        path = out / f"{i}.png"
        assert path.exists() and path.stat().st_size > 0
