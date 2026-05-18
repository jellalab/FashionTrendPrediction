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