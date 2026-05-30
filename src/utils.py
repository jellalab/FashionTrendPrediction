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
class PopularityFormulaConfig:
    likes_column: str
    comments_column: str
    weight: float


@dataclass(frozen=True)
class HashtagsConfig:
    column: str
    top_n: int


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int
    stratify: bool


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    random_state: int
    class_weight: str | None
    n_jobs: int


@dataclass(frozen=True)
class PlotsConfig:
    top_k_importances: int
    figure_dpi: int


@dataclass(frozen=True)
class PopularityConfig:
    input_xlsx: Path
    output_dir: Path
    popularity_csv: Path
    ablation_results_csv: Path
    hashtag_features_csv: Path
    combined_model_path: Path
    target_column: str
    excluded_columns: tuple[str, ...]
    popularity: PopularityFormulaConfig
    group_a_engagement: tuple[str, ...]
    group_c_behavioral: tuple[str, ...]
    hashtags: HashtagsConfig
    split: SplitConfig
    model: RandomForestConfig
    plots: PlotsConfig


@dataclass(frozen=True)
class Pipeline1VizConfig:
    input_csv: Path
    accepted_dir: Path
    rejected_dir: Path
    output_dir: Path
    color_config: Path
    min_detections_per_parent: int
    suppress: tuple[str, ...]


@dataclass(frozen=True)
class ValidateConfig:
    attributes_csv: Path
    images_dir: Path
    output_root: Path
    random_seed: int | None
    max_items: int | None
    display_max_dim: int


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


def load_pipeline1_viz_config(
    config_path: str | Path = "config/pipeline1_viz.yaml",
) -> Pipeline1VizConfig:
    """Load and validate the Pipeline 1 visualisation config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    min_detections = int(raw["min_detections_per_parent"])
    if min_detections < 0:
        raise ValueError(
            f"min_detections_per_parent must be >= 0, got {min_detections}"
        )

    suppress_raw = raw.get("suppress") or ()
    if not isinstance(suppress_raw, (list, tuple)):
        raise ValueError("suppress must be a list of plot-name strings")
    suppress = tuple(str(name) for name in suppress_raw)

    return Pipeline1VizConfig(
        input_csv=resolve_path(raw["input_csv"]),
        accepted_dir=resolve_path(raw["accepted_dir"]),
        rejected_dir=resolve_path(raw["rejected_dir"]),
        output_dir=resolve_path(raw["output_dir"]),
        color_config=resolve_path(raw["color_config"]),
        min_detections_per_parent=min_detections,
        suppress=suppress,
    )


def load_validate_config(
    config_path: str | Path = "config/validate.yaml",
) -> ValidateConfig:
    """Load and validate the manual-validation tool config from YAML."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    max_items_raw = raw.get("max_items")
    if max_items_raw in (None, "null", "None"):
        max_items: int | None = None
    else:
        max_items = int(max_items_raw)
        if max_items < 1:
            raise ValueError(f"max_items must be >= 1 when set, got {max_items}")

    seed_raw = raw.get("random_seed")
    if seed_raw in (None, "null", "None"):
        random_seed: int | None = None
    else:
        random_seed = int(seed_raw)

    display_max_dim = int(raw.get("display_max_dim", 720))
    if display_max_dim < 64:
        raise ValueError(
            f"display_max_dim must be >= 64 px, got {display_max_dim}"
        )

    return ValidateConfig(
        attributes_csv=resolve_path(raw["attributes_csv"]),
        images_dir=resolve_path(raw["images_dir"]),
        output_root=resolve_path(raw["output_root"]),
        random_seed=random_seed,
        max_items=max_items,
        display_max_dim=display_max_dim,
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


def _parse_str_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    out: list[str] = []
    for item in value:
        text = str(item)
        if not text:
            raise ValueError(f"{field} contains an empty entry")
        out.append(text)
    return tuple(out)


def load_popularity_config(
    config_path: str | Path = "config/popularity.yaml",
) -> PopularityConfig:
    """Load and validate the Pipeline 2 popularity / classification config."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    pop_raw = raw["popularity"]
    fg_raw = raw["feature_groups"]
    hs_raw = raw["hashtags"]
    split_raw = raw["split"]
    model_raw = raw["model"]
    plots_raw = raw["plots"]

    test_size = float(split_raw["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"split.test_size must be in (0, 1), got {test_size}")
    top_n = int(hs_raw["top_n"])
    if top_n < 1:
        raise ValueError(f"hashtags.top_n must be >= 1, got {top_n}")
    n_estimators = int(model_raw["n_estimators"])
    if n_estimators < 1:
        raise ValueError(
            f"model.n_estimators must be >= 1, got {n_estimators}"
        )
    weight = float(pop_raw["weight"])
    if weight <= 0.0:
        raise ValueError(f"popularity.weight must be > 0, got {weight}")

    cw_raw = model_raw.get("class_weight", None)
    class_weight = None if cw_raw in (None, "null", "None") else str(cw_raw)

    return PopularityConfig(
        input_xlsx=resolve_path(raw["input_xlsx"]),
        output_dir=resolve_path(raw["output_dir"]),
        popularity_csv=resolve_path(raw["popularity_csv"]),
        ablation_results_csv=resolve_path(raw["ablation_results_csv"]),
        hashtag_features_csv=resolve_path(raw["hashtag_features_csv"]),
        combined_model_path=resolve_path(raw["combined_model_path"]),
        target_column=str(raw["target_column"]),
        excluded_columns=_parse_str_tuple(
            raw["excluded_columns"], "excluded_columns"
        ),
        popularity=PopularityFormulaConfig(
            likes_column=str(pop_raw["likes_column"]),
            comments_column=str(pop_raw["comments_column"]),
            weight=weight,
        ),
        group_a_engagement=_parse_str_tuple(
            fg_raw["group_a_engagement"], "feature_groups.group_a_engagement"
        ),
        group_c_behavioral=_parse_str_tuple(
            fg_raw["group_c_behavioral"], "feature_groups.group_c_behavioral"
        ),
        hashtags=HashtagsConfig(
            column=str(hs_raw["column"]),
            top_n=top_n,
        ),
        split=SplitConfig(
            test_size=test_size,
            random_state=int(split_raw["random_state"]),
            stratify=bool(split_raw["stratify"]),
        ),
        model=RandomForestConfig(
            n_estimators=n_estimators,
            random_state=int(model_raw["random_state"]),
            class_weight=class_weight,
            n_jobs=int(model_raw.get("n_jobs", -1)),
        ),
        plots=PlotsConfig(
            top_k_importances=int(plots_raw["top_k_importances"]),
            figure_dpi=int(plots_raw["figure_dpi"]),
        ),
    )