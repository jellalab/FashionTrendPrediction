"""Tests for the Pipeline 1 garment detection module.

The YOLO model itself is never invoked: ``run_detection`` is exercised via
``monkeypatch`` so the tests have no network access, no weight download, and
no torch/ultralytics runtime requirement beyond import.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import detect
from src.utils import DetectionConfig, ModelConfig


DEEPFASHION2_NAMES: dict[int, str] = {
    0: "short_sleeved_shirt",
    1: "long_sleeved_shirt",
    2: "short_sleeved_outwear",
    3: "long_sleeved_outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short_sleeved_dress",
    10: "long_sleeved_dress",
    11: "vest_dress",
    12: "sling_dress",
}


class _FakeBoxes:
    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.conf.shape[0])


def _make_result(
    xyxy: list[list[float]] | None,
    conf: list[float] | None,
    cls: list[int] | None,
) -> Any:
    if xyxy is None:
        boxes = _FakeBoxes(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
            cls=np.empty((0,), dtype=np.float32),
        )
    else:
        boxes = _FakeBoxes(
            xyxy=np.array(xyxy, dtype=np.float32),
            conf=np.array(conf or [], dtype=np.float32),
            cls=np.array(cls or [], dtype=np.float32),
        )
    return SimpleNamespace(boxes=boxes, names=DEEPFASHION2_NAMES)


# --- extract_detection_rows: the pure accept/reject logic ----------------


def test_two_detections_above_threshold_produces_two_rows() -> None:
    result = _make_result(
        xyxy=[[10, 20, 110, 220], [50, 60, 150, 260]],
        conf=[0.9, 0.75],
        cls=[1, 7],
    )
    rows = detect.extract_detection_rows(result, "img_a.jpg", 0.5)

    assert len(rows) == 2
    assert [r["garment_id"] for r in rows] == [0, 1]
    assert [r["category"] for r in rows] == ["long_sleeved_shirt", "trousers"]
    assert rows[0]["bbox_x"] == 10 and rows[0]["bbox_y"] == 20
    assert rows[0]["bbox_w"] == 100 and rows[0]["bbox_h"] == 200


def test_zero_detections_produces_no_rows() -> None:
    result = _make_result(xyxy=None, conf=None, cls=None)
    assert detect.extract_detection_rows(result, "img_b.jpg", 0.5) == []


def test_detections_below_threshold_are_dropped() -> None:
    result = _make_result(
        xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
        conf=[0.2, 0.49],
        cls=[0, 1],
    )
    assert detect.extract_detection_rows(result, "img_c.jpg", 0.5) == []


def test_mixed_detections_renumber_garment_ids_over_kept_only() -> None:
    result = _make_result(
        xyxy=[[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]],
        conf=[0.9, 0.3, 0.6],
        cls=[0, 1, 8],
    )
    rows = detect.extract_detection_rows(result, "img_d.jpg", 0.5)
    assert [r["garment_id"] for r in rows] == [0, 1]
    assert [r["category"] for r in rows] == ["short_sleeved_shirt", "skirt"]


# --- end-to-end run_detection with mocked YOLO ----------------------------


def _write_jpg(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    Image.new("RGB", size, color=(128, 128, 128)).save(path, "JPEG")


@pytest.fixture
def fake_dataset(tmp_path: Path) -> dict[str, Any]:
    """Three images plus a config pointing at tmp dirs."""
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _write_jpg(input_dir / name)

    config = DetectionConfig(
        input_dir=input_dir,
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

    plan: dict[str, Any] = {
        "a.jpg": _make_result(
            xyxy=[[0, 0, 10, 10], [5, 5, 25, 35]],
            conf=[0.9, 0.7],
            cls=[1, 7],
        ),
        "b.jpg": _make_result(xyxy=None, conf=None, cls=None),
        "c.jpg": _make_result(
            xyxy=[[0, 0, 10, 10]],
            conf=[0.3],
            cls=[0],
        ),
    }

    return {"config": config, "plan": plan}


def _install_fakes(monkeypatch: pytest.MonkeyPatch, plan: dict[str, Any]) -> None:
    class FakeModel:
        def predict(self, source: str, conf: float, verbose: bool) -> list[Any]:
            return [plan[Path(source).name]]

    monkeypatch.setattr(detect, "download_weights", lambda mc: Path("/tmp/fake.pt"))
    monkeypatch.setattr(detect, "load_model", lambda p: FakeModel())


def test_run_detection_routes_and_writes_csv(
    fake_dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config: DetectionConfig = fake_dataset["config"]
    _install_fakes(monkeypatch, fake_dataset["plan"])

    df = detect.run_detection(config)

    assert sorted(p.name for p in config.accepted_dir.iterdir()) == ["a.jpg"]
    assert sorted(p.name for p in config.rejected_dir.iterdir()) == ["b.jpg", "c.jpg"]

    assert config.detections_csv.exists()
    on_disk = pd.read_csv(config.detections_csv)
    assert len(on_disk) == 2
    assert len(df) == 2
    assert (on_disk["image_id"] == "a.jpg").all()


def test_csv_schema_columns_and_dtypes(
    fake_dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config: DetectionConfig = fake_dataset["config"]
    _install_fakes(monkeypatch, fake_dataset["plan"])

    df = detect.run_detection(config)

    assert list(df.columns) == list(detect.CSV_COLUMNS)
    assert df["garment_id"].dtype == np.int64
    for col in ("confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        assert df[col].dtype == np.float64
    assert str(df["image_id"].dtype) == "string"
    assert str(df["category"].dtype) == "string"


def test_run_detection_is_idempotent(
    fake_dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config: DetectionConfig = fake_dataset["config"]
    _install_fakes(monkeypatch, fake_dataset["plan"])

    detect.run_detection(config)
    first = config.detections_csv.read_bytes()
    first_accepted = sorted(p.name for p in config.accepted_dir.iterdir())
    first_rejected = sorted(p.name for p in config.rejected_dir.iterdir())

    detect.run_detection(config)
    second = config.detections_csv.read_bytes()
    second_accepted = sorted(p.name for p in config.accepted_dir.iterdir())
    second_rejected = sorted(p.name for p in config.rejected_dir.iterdir())

    assert first == second
    assert first_accepted == second_accepted
    assert first_rejected == second_rejected


def test_total_rows_equal_sum_of_accepted_garments(
    fake_dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config: DetectionConfig = fake_dataset["config"]
    _install_fakes(monkeypatch, fake_dataset["plan"])

    df = detect.run_detection(config)
    per_image = df.groupby("image_id").size().sum()
    assert per_image == len(df)


def test_corrupt_image_is_skipped_and_not_classified(
    fake_dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config: DetectionConfig = fake_dataset["config"]
    bad = config.input_dir / "broken.jpg"
    bad.write_bytes(b"not a real image")
    fake_dataset["plan"]["broken.jpg"] = _make_result(None, None, None)

    _install_fakes(monkeypatch, fake_dataset["plan"])

    detect.run_detection(config)
    accepted_names = {p.name for p in config.accepted_dir.iterdir()}
    rejected_names = {p.name for p in config.rejected_dir.iterdir()}
    assert "broken.jpg" not in accepted_names
    assert "broken.jpg" not in rejected_names
