"""Pipeline 1 — descriptive visualisations of the joined attribute CSV.

Produces six static PNG plots used in the Methodology and Results chapters:

1. ``acceptance_rejection.png`` — accepted vs rejected image counts.
2. ``garment_category_distribution.png`` — DeepFashion2 parent counts.
3. ``dominant_color_distribution.png`` — dominant palette-name counts,
   filled with the actual palette RGB.
4. ``pattern_class_distribution.png`` — plain / subtle / patterned counts.
5. ``refined_garment_type_per_parent.png`` — small multiples of CLIP
   refined sub-label counts within each frequent parent.
6. ``clip_uncertainty_per_parent.png`` — per-parent share of ``uncertain``
   CLIP refinements.

This module is read-only with respect to ``yolo_fashion_attributes.csv`` and
the upstream Step-1 accepted/rejected image folders.

Run as a module from the project root::

    uv run python -m src.pipeline1_viz
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no-display backend for deterministic file output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import (
    PaletteEntry,
    Pipeline1VizConfig,
    load_color_config,
    load_pipeline1_viz_config,
)

logger = logging.getLogger(__name__)


__all__ = [
    "PLOT_NAMES",
    "apply_style",
    "count_images",
    "is_pale",
    "palette_rgb_map",
    "parents_above_threshold",
    "plot_acceptance_rejection",
    "plot_clip_uncertainty_per_parent",
    "plot_dominant_color_distribution",
    "plot_garment_category_distribution",
    "plot_pattern_class_distribution",
    "plot_refined_garment_type_per_parent",
    "run_pipeline1_viz",
]


# Plot stems in the order the orchestrator emits them. Suppression in the
# config references these names verbatim.
PLOT_NAMES: tuple[str, ...] = (
    "acceptance_rejection",
    "garment_category_distribution",
    "dominant_color_distribution",
    "pattern_class_distribution",
    "refined_garment_type_per_parent",
    "clip_uncertainty_per_parent",
)

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

# Pattern bucketing order from src/pattern.py. Plot 4 enforces this left-to-
# right ordering regardless of count so the chart reads as a complexity ramp.
PATTERN_ORDER: tuple[str, ...] = ("plain", "subtle", "patterned")

# Literal string written by src/clip_refine.py when the top softmax score is
# below the threshold. Plots 5 and 6 hinge on this exact label.
UNCERTAIN_LABEL: str = "uncertain"

# Neutral palette for non-coloured charts.
_NEUTRAL_BAR = "#3F4A5C"
_UNCERTAIN_BAR = "#9A9A9A"
_PALE_OUTLINE = "#444444"
_SPINE_COLOR = "#444444"
_SUBTITLE_COLOR = "#666666"
_BACKGROUND = "#FAFAFA"
_PATTERN_BARS = ("#7FA6C9", "#E0B074", "#C16A6A")  # plain / subtle / patterned


# --- style ----------------------------------------------------------------


def apply_style() -> None:
    """Apply the academic-minimal matplotlib rcParams used across all plots.

    Idempotent — safe to call before each plot. Title fonts switch to serif
    per-axes; the global default keeps tick / label fonts sans-serif.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": _BACKGROUND,
            "axes.facecolor": _BACKGROUND,
            "savefig.facecolor": _BACKGROUND,
            "savefig.edgecolor": _BACKGROUND,
            "axes.edgecolor": _SPINE_COLOR,
            "axes.labelcolor": _SPINE_COLOR,
            "xtick.color": _SPINE_COLOR,
            "ytick.color": _SPINE_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
            "font.family": "sans-serif",
            "axes.titleweight": "normal",
            "axes.titlepad": 16,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


# --- helpers --------------------------------------------------------------


def count_images(directory: Path) -> int:
    """Return the number of image files at the top level of *directory*.

    Hidden files (``.DS_Store``) and sentinels (``.gitkeep``) are skipped by
    requiring a recognised image extension.
    """
    if not directory.exists():
        return 0
    return sum(
        1
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def palette_rgb_map(
    palette: tuple[PaletteEntry, ...],
) -> dict[str, tuple[int, int, int]]:
    """Map every palette name to its (r, g, b) triple."""
    return {entry.name: entry.rgb for entry in palette}


def is_pale(rgb: tuple[int, int, int], threshold: float = 0.85) -> bool:
    """Return True when *rgb* is light enough to need an outline on a pale bg.

    Uses Rec. 709 relative luminance on sRGB-normalised channels — close
    enough to perceptual lightness for the "white / cream stays visible on
    #FAFAFA" use-case without pulling in an extra colour library.
    """
    r, g, b = (c / 255.0 for c in rgb)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > threshold


def parents_above_threshold(df: pd.DataFrame, min_count: int) -> list[str]:
    """Return parent categories with strictly more than *min_count* detections.

    Sorted alphabetically so the small-multiples layout is deterministic.
    """
    counts = df["category"].value_counts()
    kept = counts[counts > min_count].index.tolist()
    return sorted(kept)


def _normalise_rgb_unit(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _stable_value_counts(series: pd.Series) -> pd.Series:
    """value_counts sorted by (count desc, label asc) for deterministic plots."""
    counts = series.value_counts(dropna=True)
    counts_df = counts.rename_axis("label").reset_index(name="count")
    counts_df = counts_df.sort_values(
        by=["count", "label"], ascending=[False, True], kind="stable"
    )
    return pd.Series(
        counts_df["count"].to_numpy(),
        index=counts_df["label"].to_numpy(),
        name="count",
    )


def _set_axes_style(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.tick_params(axis="both", colors=_SPINE_COLOR, length=4)


def _set_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Set a serif title with an optional italic subtitle below it.

    Both positions are computed in offset points so the spacing is constant
    regardless of figure size — axes-fractional y-coords drifted on taller
    plots and caused the subtitle to overlap the title.
    """
    if subtitle:
        ax.set_title(title, fontfamily="serif", fontsize=14, color=_SPINE_COLOR, pad=30)
        ax.annotate(
            subtitle,
            xy=(0.5, 1.0),
            xytext=(0, 6),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontfamily="serif",
            fontstyle="italic",
            fontsize=10,
            color=_SUBTITLE_COLOR,
        )
    else:
        ax.set_title(title, fontfamily="serif", fontsize=14, color=_SPINE_COLOR)


def _save(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# --- plot 1: acceptance_rejection ----------------------------------------


def plot_acceptance_rejection(
    accepted_count: int,
    rejected_count: int,
    output_path: Path,
) -> Path:
    """Horizontal two-bar chart with counts and percentages."""
    total = accepted_count + rejected_count
    labels = ["Accepted", "Rejected"]
    counts = [accepted_count, rejected_count]
    colors = [_NEUTRAL_BAR, "#B5B5B5"]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(labels, counts, color=colors, edgecolor="none")

    for bar, count in zip(bars, counts):
        pct = (100.0 * count / total) if total else 0.0
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count}  ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color=_SPINE_COLOR,
        )

    ax.set_xlabel("Number of images")
    ax.set_ylabel("")
    ax.invert_yaxis()  # "Accepted" on top
    ax.set_xlim(0, max(counts) * 1.18 if max(counts) > 0 else 1)
    _set_axes_style(ax)
    _set_title(
        ax,
        "Step 1 Fashion Filter — Image Acceptance vs Rejection",
        subtitle=f"Total images screened: {total}",
    )

    fig.tight_layout()
    return _save(fig, output_path)


# --- plot 2: garment_category_distribution -------------------------------


def plot_garment_category_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Horizontal bar of DeepFashion2 parent counts, sorted descending."""
    counts = _stable_value_counts(df["category"])
    labels = [str(x).replace("_", " ") for x in counts.index]
    values = counts.to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.4 * len(labels) + 2)))
    bars = ax.barh(labels, values, color=_NEUTRAL_BAR, edgecolor="none")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            ha="left",
            fontsize=9,
            color=_SPINE_COLOR,
        )

    ax.set_xlabel("Number of garments")
    ax.set_ylabel("Garment category")
    ax.invert_yaxis()  # largest at top
    ax.set_xlim(0, values.max() * 1.12)
    _set_axes_style(ax)
    _set_title(ax, "Garment Category Distribution (DeepFashion2 Parent)")

    fig.tight_layout()
    return _save(fig, output_path)


# --- plot 3: dominant_color_distribution ---------------------------------


def plot_dominant_color_distribution(
    df: pd.DataFrame,
    palette_map: dict[str, tuple[int, int, int]],
    output_path: Path,
) -> Path:
    """Horizontal bar of dominant palette-name counts, filled with palette RGB.

    Pale fills (luminance > 0.85) get a thin dark outline so they stay visible
    against the off-white background. Names absent from the palette config
    fall back to the per-group mean of dominant_r/g/b, matching the spec.
    """
    counts = _stable_value_counts(df["dominant_color_name"])
    names = [str(n) for n in counts.index]
    values = counts.to_numpy()

    bar_rgbs: list[tuple[int, int, int]] = []
    for name in names:
        if name in palette_map:
            bar_rgbs.append(palette_map[name])
            continue
        # fallback: mean RGB of detections labelled with this name
        subset = df.loc[df["dominant_color_name"] == name, ["dominant_r", "dominant_g", "dominant_b"]]
        mean = subset.mean(numeric_only=True).round().astype(int)
        bar_rgbs.append((int(mean["dominant_r"]), int(mean["dominant_g"]), int(mean["dominant_b"])))

    fig, ax = plt.subplots(figsize=(8, max(5.0, 0.4 * len(names) + 2)))
    bars = ax.barh(names, values, color="#FFFFFF")
    for bar, rgb in zip(bars, bar_rgbs):
        bar.set_facecolor(_normalise_rgb_unit(rgb))
        if is_pale(rgb):
            bar.set_edgecolor(_PALE_OUTLINE)
            bar.set_linewidth(0.6)
        else:
            bar.set_edgecolor("none")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            ha="left",
            fontsize=9,
            color=_SPINE_COLOR,
        )

    ax.set_xlabel("Number of garments")
    ax.set_ylabel("Dominant colour")
    ax.invert_yaxis()
    ax.set_xlim(0, values.max() * 1.12)
    _set_axes_style(ax)
    _set_title(
        ax,
        "Dominant Colour Distribution",
        subtitle="Bars filled with the curated fashion-palette RGB",
    )

    fig.tight_layout()
    return _save(fig, output_path)


# --- plot 4: pattern_class_distribution ----------------------------------


def plot_pattern_class_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Vertical bar of plain / subtle / patterned counts in fixed order."""
    series = df["pattern_class"].dropna()
    total = len(series)
    counts = {cls: int((series == cls).sum()) for cls in PATTERN_ORDER}
    values = [counts[c] for c in PATTERN_ORDER]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(
        list(PATTERN_ORDER),
        values,
        color=list(_PATTERN_BARS),
        edgecolor="none",
        width=0.6,
    )

    max_value = max(values) if max(values) > 0 else 1
    for bar, value in zip(bars, values):
        pct = (100.0 * value / total) if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.02,
            f"{value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color=_SPINE_COLOR,
        )

    ax.set_xlabel("Pattern class")
    ax.set_ylabel("Number of garments")
    ax.set_ylim(0, max_value * 1.18)
    _set_axes_style(ax)
    _set_title(ax, "Pattern Complexity Distribution")

    fig.tight_layout()
    return _save(fig, output_path)


# --- plot 5: refined_garment_type_per_parent -----------------------------


def _empty_placeholder(title: str, message: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.7,
        title,
        ha="center",
        va="center",
        fontfamily="serif",
        fontsize=14,
        color=_SPINE_COLOR,
    )
    ax.text(
        0.5,
        0.4,
        message,
        ha="center",
        va="center",
        fontsize=11,
        color=_SUBTITLE_COLOR,
    )
    return _save(fig, output_path)


def plot_refined_garment_type_per_parent(
    df: pd.DataFrame,
    min_detections_per_parent: int,
    output_path: Path,
) -> Path:
    """Small multiples of CLIP refined sub-label counts within each parent.

    Parents with at most *min_detections_per_parent* detections are skipped.
    Within each subplot the ``uncertain`` bar is drawn in grey; the named
    sub-labels share the neutral charcoal so the visual hierarchy reads as
    "confident vs unconfident".
    """
    parents = parents_above_threshold(df, min_detections_per_parent)
    if not parents:
        return _empty_placeholder(
            "Refined Garment Type per Parent",
            f"No parent category exceeds {min_detections_per_parent} detections.",
            output_path,
        )

    n_cols = 2 if len(parents) > 1 else 1
    n_rows = math.ceil(len(parents) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, max(3, 2.2 * n_rows)),
        squeeze=False,
    )

    for idx, parent in enumerate(parents):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = df.loc[df["category"] == parent, "category_refined"].dropna()
        counts = _stable_value_counts(sub)
        labels = [str(x) for x in counts.index]
        values = counts.to_numpy()

        bar_colors = [
            _UNCERTAIN_BAR if label == UNCERTAIN_LABEL else _NEUTRAL_BAR
            for label in labels
        ]
        bars = ax.barh(labels, values, color=bar_colors, edgecolor="none")

        max_value = max(values) if len(values) else 1
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + max_value * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)}",
                va="center",
                ha="left",
                fontsize=8,
                color=_SPINE_COLOR,
            )

        ax.set_xlim(0, max_value * 1.20 if max_value else 1)
        ax.invert_yaxis()
        ax.set_xlabel("Number of garments")
        ax.set_ylabel("")
        ax.set_title(
            parent.replace("_", " "),
            fontfamily="serif",
            fontsize=11,
            color=_SPINE_COLOR,
            pad=8,
        )
        _set_axes_style(ax)

    # Hide unused subplot cells (when len(parents) is odd and n_cols == 2).
    for empty_idx in range(len(parents), n_rows * n_cols):
        axes[empty_idx // n_cols][empty_idx % n_cols].axis("off")

    fig.suptitle(
        "Refined Garment Type per Parent (CLIP Sub-label Counts)",
        fontfamily="serif",
        fontsize=15,
        color=_SPINE_COLOR,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output_path)


# --- plot 6: clip_uncertainty_per_parent ---------------------------------


def plot_clip_uncertainty_per_parent(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-parent share of ``uncertain`` CLIP refinements, sorted desc."""
    subset = df[["category", "category_refined"]].dropna(subset=["category"])
    grouped = subset.groupby("category", sort=False)["category_refined"].apply(
        lambda s: (s == UNCERTAIN_LABEL).mean() * 100.0 if len(s) else 0.0
    )
    grouped_df = grouped.rename("pct").reset_index()
    grouped_df = grouped_df.sort_values(
        by=["pct", "category"], ascending=[False, True], kind="stable"
    )

    if grouped_df.empty:
        return _empty_placeholder(
            "CLIP Uncertainty per Parent",
            "No parent categories present in the input.",
            output_path,
        )

    labels = [str(c).replace("_", " ") for c in grouped_df["category"]]
    values = grouped_df["pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.4 * len(labels) + 2)))
    bars = ax.barh(labels, values, color=_UNCERTAIN_BAR, edgecolor="none")

    max_value = max(values) if len(values) else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(max_value, 1.0) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color=_SPINE_COLOR,
        )

    ax.set_xlabel("Percentage of garments marked uncertain")
    ax.set_ylabel("Garment category")
    ax.invert_yaxis()
    ax.set_xlim(0, max(max_value * 1.15, 5.0))
    _set_axes_style(ax)
    _set_title(ax, "CLIP Refinement Uncertainty per Parent Category")

    fig.tight_layout()
    return _save(fig, output_path)


# --- orchestrator ---------------------------------------------------------


def _emit(
    name: str,
    suppress: set[str],
    output_dir: Path,
    fn: Callable[[Path], Path],
    written: list[Path],
) -> None:
    if name in suppress:
        logger.info("Skipping suppressed plot: %s", name)
        return
    path = output_dir / f"{name}.png"
    fn(path)
    written.append(path)


def run_pipeline1_viz(config: Pipeline1VizConfig) -> list[Path]:
    """Run the full plot generation. Returns the list of PNGs written.

    Re-running overwrites cleanly: matplotlib opens each path in binary write
    mode under the hood. The output directory is created if absent.
    """
    apply_style()
    df = pd.read_csv(config.input_csv)
    color_cfg = load_color_config(config.color_config)
    palette_map = palette_rgb_map(color_cfg.palette)
    accepted = count_images(config.accepted_dir)
    rejected = count_images(config.rejected_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    suppress = set(config.suppress)
    written: list[Path] = []

    _emit(
        "acceptance_rejection",
        suppress,
        config.output_dir,
        lambda p: plot_acceptance_rejection(accepted, rejected, p),
        written,
    )
    _emit(
        "garment_category_distribution",
        suppress,
        config.output_dir,
        lambda p: plot_garment_category_distribution(df, p),
        written,
    )
    _emit(
        "dominant_color_distribution",
        suppress,
        config.output_dir,
        lambda p: plot_dominant_color_distribution(df, palette_map, p),
        written,
    )
    _emit(
        "pattern_class_distribution",
        suppress,
        config.output_dir,
        lambda p: plot_pattern_class_distribution(df, p),
        written,
    )
    _emit(
        "refined_garment_type_per_parent",
        suppress,
        config.output_dir,
        lambda p: plot_refined_garment_type_per_parent(
            df, config.min_detections_per_parent, p
        ),
        written,
    )
    _emit(
        "clip_uncertainty_per_parent",
        suppress,
        config.output_dir,
        lambda p: plot_clip_uncertainty_per_parent(df, p),
        written,
    )

    _print_summary(config, accepted, rejected, df, written)
    return written


def _print_summary(
    config: Pipeline1VizConfig,
    accepted: int,
    rejected: int,
    df: pd.DataFrame,
    written: list[Path],
) -> None:
    print()
    print("=== Pipeline 1: visualisation summary ===")
    print(f"Input CSV:       {config.input_csv}  ({len(df)} garment rows)")
    print(f"Accepted images: {accepted}")
    print(f"Rejected images: {rejected}")
    print(f"Output dir:      {config.output_dir}")
    print(f"Plots written:   {len(written)} / {len(PLOT_NAMES)}")
    for path in written:
        print(f"  {path.name}")
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_pipeline1_viz_config()
    run_pipeline1_viz(config)


if __name__ == "__main__":
    main()
