"""Tests for the shared crop helpers used by both attribute pipelines."""

from __future__ import annotations

import numpy as np
import pytest

from src.crop_utils import center_crop, clip_bbox_to_image


def test_bbox_inside_image_is_unchanged() -> None:
    assert clip_bbox_to_image((10, 20, 30, 40), (100, 100)) == (10, 20, 40, 60)


def test_bbox_partially_outside_is_clipped() -> None:
    assert clip_bbox_to_image((80, 80, 50, 50), (100, 100)) == (80, 80, 100, 100)


def test_bbox_negative_origin_is_clipped() -> None:
    assert clip_bbox_to_image((-20, -10, 50, 40), (100, 100)) == (0, 0, 30, 30)


def test_bbox_entirely_outside_collapses_to_zero_area() -> None:
    x1, y1, x2, y2 = clip_bbox_to_image((150, 150, 50, 50), (100, 100))
    assert x2 - x1 == 0 and y2 - y1 == 0


def test_center_crop_extracts_middle_fraction() -> None:
    image = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    cropped = center_crop(image, 0.6)
    assert cropped.shape == (60, 60)
    # The view starts 20px in on each axis.
    assert cropped[0, 0] == image[20, 20]


def test_center_crop_full_fraction_returns_full_image() -> None:
    image = np.zeros((30, 50, 3), dtype=np.uint8)
    cropped = center_crop(image, 1.0)
    assert cropped.shape == image.shape


def test_center_crop_rejects_invalid_fraction() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        center_crop(image, 0.0)
    with pytest.raises(ValueError):
        center_crop(image, 1.5)
