"""Shared utilities: project paths and YAML config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root, located two levels above this file."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root, if not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    filename: str
    cache_dir: Path


@dataclass(frozen=True)
class DetectionConfig:
    input_dir: Path
    accepted_dir: Path
    rejected_dir: Path
    detections_csv: Path
    confidence_threshold: float
    model: ModelConfig


@dataclass(frozen=True)
class PatternConfig:
    detections_csv: Path
    images_dir: Path
    output_csv: Path
    center_crop_fraction: float
    quantile_low: float
    quantile_high: float


@dataclass(frozen=True)
class PaletteEntry:
    name: str
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ColorConfig:
    detections_csv: Path
    images_dir: Path
    output_csv: Path
    center_crop_fraction: float
    kmeans_k: int
    random_state: int
    palette: tuple[PaletteEntry, ...]


@dataclass(frozen=True)
class JoinConfig:
    detections_csv: Path
    color_csv: Path
    pattern_csv: Path
    clip_csv: Path
    output_csv: Path


@dataclass(frozen=True)
class ClipRefineConfig:
    detections_csv: Path
    images_dir: Path
    output_csv: Path
    center_crop_fraction: float
    model_id: str
    model_cache_dir: Path
    prompt_template: str
    threshold: float
    batch_size: int
    taxonomy: dict[str, tuple[str, ...]]


def load_detection_config(
    config_path: str | Path = "config/detection.yaml",
) -> DetectionConfig:
    """Load and validate the detection pipeline config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model_raw = raw["model"]
    return DetectionConfig(
        input_dir=resolve_path(raw["input_dir"]),
        accepted_dir=resolve_path(raw["accepted_dir"]),
        rejected_dir=resolve_path(raw["rejected_dir"]),
        detections_csv=resolve_path(raw["detections_csv"]),
        confidence_threshold=float(raw["confidence_threshold"]),
        model=ModelConfig(
            repo_id=str(model_raw["repo_id"]),
            filename=str(model_raw["filename"]),
            cache_dir=resolve_path(model_raw["cache_dir"]),
        ),
    )


def load_pattern_config(
    config_path: str | Path = "config/pattern.yaml",
) -> PatternConfig:
    """Load and validate the pattern-complexity pipeline config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return PatternConfig(
        detections_csv=resolve_path(raw["detections_csv"]),
        images_dir=resolve_path(raw["images_dir"]),
        output_csv=resolve_path(raw["output_csv"]),
        center_crop_fraction=float(raw["center_crop_fraction"]),
        quantile_low=float(raw["quantile_low"]),
        quantile_high=float(raw["quantile_high"]),
    )


def _parse_palette(raw_palette: list[dict[str, Any]]) -> tuple[PaletteEntry, ...]:
    """Validate and normalize a YAML palette block.

    Each entry must have a ``name`` (str) and ``rgb`` (sequence of three ints
    in ``[0, 255]``). Duplicate names are rejected so the nearest-neighbor
    lookup remains unambiguous.
    """
    if not raw_palette:
        raise ValueError("palette must contain at least one entry")

    seen: set[str] = set()
    entries: list[PaletteEntry] = []
    for item in raw_palette:
        name = str(item["name"])
        rgb_raw = item["rgb"]
        if len(rgb_raw) != 3:
            raise ValueError(f"palette entry {name!r} rgb must have 3 channels")
        rgb = tuple(int(c) for c in rgb_raw)
        for c in rgb:
            if not 0 <= c <= 255:
                raise ValueError(
                    f"palette entry {name!r} has out-of-range channel: {rgb}"
                )
        if name in seen:
            raise ValueError(f"duplicate palette name: {name!r}")
        seen.add(name)
        entries.append(PaletteEntry(name=name, rgb=rgb))
    return tuple(entries)


def load_color_config(
    config_path: str | Path = "config/color.yaml",
) -> ColorConfig:
    """Load and validate the color-extraction pipeline config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return ColorConfig(
        detections_csv=resolve_path(raw["detections_csv"]),
        images_dir=resolve_path(raw["images_dir"]),
        output_csv=resolve_path(raw["output_csv"]),
        center_crop_fraction=float(raw["center_crop_fraction"]),
        kmeans_k=int(raw["kmeans_k"]),
        random_state=int(raw["random_state"]),
        palette=_parse_palette(raw["palette"]),
    )


def load_join_config(
    config_path: str | Path = "config/join.yaml",
) -> JoinConfig:
    """Load and validate the Pipeline 1 join config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return JoinConfig(
        detections_csv=resolve_path(raw["detections_csv"]),
        color_csv=resolve_path(raw["color_csv"]),
        pattern_csv=resolve_path(raw["pattern_csv"]),
        clip_csv=resolve_path(raw["clip_csv"]),
        output_csv=resolve_path(raw["output_csv"]),
    )


def _parse_taxonomy(
    raw_taxonomy: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Validate and normalize a YAML taxonomy block.

    Each key is a parent category and each value must be a list of sub-label
    strings. Duplicates within a single parent's list are rejected so the
    softmax distribution stays well-defined; empty lists are allowed (they
    trigger the no-refinement passthrough at runtime).
    """
    if not isinstance(raw_taxonomy, dict):
        raise ValueError("taxonomy must be a mapping of parent -> [sub-labels]")

    parsed: dict[str, tuple[str, ...]] = {}
    for parent, labels in raw_taxonomy.items():
        if not isinstance(labels, list):
            raise ValueError(
                f"taxonomy entry {parent!r} must be a list of sub-labels"
            )
        cleaned: list[str] = []
        seen: set[str] = set()
        for label in labels:
            text = str(label).strip()
            if not text:
                raise ValueError(
                    f"taxonomy entry {parent!r} contains an empty sub-label"
                )
            if text in seen:
                raise ValueError(
                    f"taxonomy entry {parent!r} has duplicate sub-label: {text!r}"
                )
            seen.add(text)
            cleaned.append(text)
        parsed[str(parent)] = tuple(cleaned)
    return parsed


def load_clip_refine_config(
    config_path: str | Path = "config/clip_refine.yaml",
) -> ClipRefineConfig:
    """Load and validate the CLIP-refinement pipeline config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    threshold = float(raw["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    batch_size = int(raw["batch_size"])
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    model_raw = raw["model"]
    return ClipRefineConfig(
        detections_csv=resolve_path(raw["detections_csv"]),
        images_dir=resolve_path(raw["images_dir"]),
        output_csv=resolve_path(raw["output_csv"]),
        center_crop_fraction=float(raw["center_crop_fraction"]),
        model_id=str(model_raw["id"]),
        model_cache_dir=resolve_path(model_raw["cache_dir"]),
        prompt_template=str(raw["prompt_template"]),
        threshold=threshold,
        batch_size=batch_size,
        taxonomy=_parse_taxonomy(raw["taxonomy"]),
    )