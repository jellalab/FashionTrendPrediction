"""End-to-end integration test for Pipeline 1.

Exercises the full detect → color → pattern → clip_refine → join chain on a
tiny synthetic fixture (3 solid-color images, 1 corrupt, 1 referenced but
missing). The YOLO model and CLIP model are both stubbed out so the test
runs offline with no weight downloads.

The point isn't ML accuracy — it's that the modules' shape/dtype/path
contracts agree with each other and that the joined output preserves every
detection row even when downstream extractors skip some.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import clip_refine, color, detect, join, pattern
from src.utils import (
    ClipRefineConfig,
    ColorConfig,
    DetectionConfig,
    JoinConfig,
    ModelConfig,
    PatternConfig,
    load_color_config,
)


DEEPFASHION2_NAMES: dict[int, str] = {
    0: "short_sleeved_shirt",
    1: "long_sleeved_shirt",
    7: "trousers",
}


def _write_solid(path: Path, rgb: tuple[int, int, int]) -> None:
    bgr = np.full((200, 200, 3), (rgb[2], rgb[1], rgb[0]), dtype=np.uint8)
    cv2.imwrite(str(path), bgr)


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.conf.shape[0])


def _fake_result(xyxy: list[list[float]], conf: list[float], cls: list[int]) -> Any:
    boxes = _FakeBoxes(
        xyxy=np.array(xyxy, dtype=np.float32),
        conf=np.array(conf, dtype=np.float32),
        cls=np.array(cls, dtype=np.float32),
    )
    return SimpleNamespace(boxes=boxes, names=DEEPFASHION2_NAMES)


def _uniform_infer_fn(
    images: list[Image.Image], labels: list[str]
) -> np.ndarray:
    n, L = len(images), len(labels)
    return np.full((n, L), 1.0 / L, dtype=np.float64)


def test_pipeline1_end_to_end_preserves_detections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_solid(raw / "red.jpg", (200, 30, 30))
    _write_solid(raw / "navy.jpg", (28, 38, 65))
    _write_solid(raw / "black.jpg", (10, 10, 10))

    plan = {
        "red.jpg": _fake_result(
            xyxy=[[10, 10, 180, 180]], conf=[0.9], cls=[0]
        ),
        "navy.jpg": _fake_result(
            xyxy=[[20, 20, 170, 170]], conf=[0.8], cls=[1]
        ),
        "black.jpg": _fake_result(
            xyxy=[[15, 15, 175, 175]], conf=[0.85], cls=[7]
        ),
    }

    detect_cfg = DetectionConfig(
        input_dir=raw,
        accepted_dir=tmp_path / "accepted",
        rejected_dir=tmp_path / "rejected",
        detections_csv=tmp_path / "detections.csv",
        confidence_threshold=0.5,
        model=ModelConfig(
            repo_id="fake/repo",
            filename="fake.pt",
            cache_dir=tmp_path / "models",
        ),
    )

    class FakeYolo:
        def predict(self, source: str, conf: float, verbose: bool) -> list[Any]:
            return [plan[Path(source).name]]

    monkeypatch.setattr(detect, "download_weights", lambda mc: Path("/tmp/fake.pt"))
    monkeypatch.setattr(detect, "load_model", lambda p: FakeYolo())

    detections = detect.run_detection(detect_cfg)
    assert len(detections) == 3

    # color
    color_cfg = ColorConfig(
        detections_csv=detect_cfg.detections_csv,
        images_dir=detect_cfg.accepted_dir,
        output_csv=tmp_path / "color_attributes.csv",
        center_crop_fraction=0.6,
        kmeans_k=3,
        random_state=42,
        palette=load_color_config().palette,
    )
    color_df = color.run_color_extraction(color_cfg)
    assert len(color_df) == 3

    # pattern
    pattern_cfg = PatternConfig(
        detections_csv=detect_cfg.detections_csv,
        images_dir=detect_cfg.accepted_dir,
        output_csv=tmp_path / "pattern_attributes.csv",
        center_crop_fraction=0.6,
        quantile_low=0.33,
        quantile_high=0.66,
    )
    pattern_df = pattern.run_pattern_detection(pattern_cfg)
    assert len(pattern_df) == 3

    # clip_refine
    clip_cfg = ClipRefineConfig(
        detections_csv=detect_cfg.detections_csv,
        images_dir=detect_cfg.accepted_dir,
        output_csv=tmp_path / "clip_refinement.csv",
        center_crop_fraction=0.6,
        model_id="dummy/clip",
        model_cache_dir=tmp_path / "clip_cache",
        prompt_template="a photo of a {label}",
        threshold=0.4,
        batch_size=4,
        taxonomy={
            "short_sleeved_shirt": ("t-shirt", "polo shirt", "blouse"),
            "long_sleeved_shirt": ("sweater", "blazer", "cardigan"),
            "trousers": ("jeans", "tailored trousers", "leggings"),
        },
    )
    clip_df = clip_refine.run_clip_refinement(clip_cfg, infer_fn=_uniform_infer_fn)
    assert len(clip_df) == 3

    # join
    join_cfg = JoinConfig(
        detections_csv=detect_cfg.detections_csv,
        color_csv=color_cfg.output_csv,
        pattern_csv=pattern_cfg.output_csv,
        clip_csv=clip_cfg.output_csv,
        output_csv=tmp_path / "yolo_fashion_attributes.csv",
    )
    joined = join.run_join(join_cfg)

    assert len(joined) == 3
    assert list(joined.columns) == list(join.CSV_COLUMNS)
    # every detection survived end-to-end with no NaN attribute columns
    for col in (
        "dominant_color_name",
        "laplacian_variance",
        "pattern_class",
        "category_refined",
    ):
        assert joined[col].notna().all(), f"NaN in {col}: {joined[col].tolist()}"


def test_pipeline1_join_preserves_detections_when_extractor_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detection row whose image is missing must still appear in the join
    output with NaN attribute cells — never silently dropped."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_solid(raw / "ok.jpg", (200, 30, 30))

    plan = {
        "ok.jpg": _fake_result(
            xyxy=[[10, 10, 180, 180]], conf=[0.9], cls=[0]
        ),
    }

    detect_cfg = DetectionConfig(
        input_dir=raw,
        accepted_dir=tmp_path / "accepted",
        rejected_dir=tmp_path / "rejected",
        detections_csv=tmp_path / "detections.csv",
        confidence_threshold=0.5,
        model=ModelConfig(
            repo_id="fake/repo",
            filename="fake.pt",
            cache_dir=tmp_path / "models",
        ),
    )

    class FakeYolo:
        def predict(self, source: str, conf: float, verbose: bool) -> list[Any]:
            return [plan[Path(source).name]]

    monkeypatch.setattr(detect, "download_weights", lambda mc: Path("/tmp/fake.pt"))
    monkeypatch.setattr(detect, "load_model", lambda p: FakeYolo())

    detect.run_detection(detect_cfg)

    # Inject a phantom detection row pointing at a file the extractors will
    # fail to open. The join must still surface it.
    df = pd.read_csv(detect_cfg.detections_csv)
    phantom = df.iloc[0].to_dict()
    phantom["image_id"] = "ghost.jpg"
    phantom["garment_id"] = 0
    pd.concat([df, pd.DataFrame([phantom])], ignore_index=True).to_csv(
        detect_cfg.detections_csv, index=False
    )

    color_cfg = ColorConfig(
        detections_csv=detect_cfg.detections_csv,
        images_dir=detect_cfg.accepted_dir,
        output_csv=tmp_path / "color_attributes.csv",
        center_crop_fraction=0.6,
        kmeans_k=3,
        random_state=42,
        palette=load_color_config().palette,
    )
    color.run_color_extraction(color_cfg)

    pattern_cfg = PatternConfig(
        detections_csv=detect_cfg.detections_csv,
        images_dir=detect_cfg.accepted_dir,
        output_csv=tmp_path / "pattern_attributes.csv",
        center_crop_fraction=0.6,
        quantile_low=0.33,
        quantile_high=0.66,
    )
    pattern.run_pattern_detection(pattern_cfg)

    clip_cfg = ClipRefineConfig(
        detections_csv=detect_cfg.detections_csv,
        images_dir=detect_cfg.accepted_dir,
        output_csv=tmp_path / "clip_refinement.csv",
        center_crop_fraction=0.6,
        model_id="dummy/clip",
        model_cache_dir=tmp_path / "clip_cache",
        prompt_template="a photo of a {label}",
        threshold=0.4,
        batch_size=4,
        taxonomy={
            "short_sleeved_shirt": ("t-shirt", "polo shirt", "blouse"),
        },
    )
    clip_refine.run_clip_refinement(clip_cfg, infer_fn=_uniform_infer_fn)

    join_cfg = JoinConfig(
        detections_csv=detect_cfg.detections_csv,
        color_csv=color_cfg.output_csv,
        pattern_csv=pattern_cfg.output_csv,
        clip_csv=clip_cfg.output_csv,
        output_csv=tmp_path / "yolo_fashion_attributes.csv",
    )
    joined = join.run_join(join_cfg)

    assert len(joined) == 2
    ghost = joined[joined["image_id"] == "ghost.jpg"]
    assert len(ghost) == 1
    assert pd.isna(ghost["dominant_color_name"].iloc[0])
    assert pd.isna(ghost["laplacian_variance"].iloc[0])
