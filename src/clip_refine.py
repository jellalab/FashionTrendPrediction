"""Pipeline 1 Step 2C — CLIP zero-shot garment refinement.

For each garment in ``detections.csv``, this module refines the broad YOLO
parent category into a fine-grained sub-label drawn from a hierarchical
taxonomy. Refinement is strictly zero-shot: CLIP scores the garment crop
against a parent-conditioned list of sub-labels (wrapped in a configurable
prompt template) and picks the highest-probability one — unless that
probability is below a confidence threshold, in which case the row is
labelled ``uncertain``.

The same inner center-crop fraction used by Steps 2A and 2B is applied so
all three per-garment modules see the same region of pixels. CLIP's own
``CLIPProcessor`` does the resize / normalize / token-pad; we do not
touch the crop ourselves beyond the bbox slice + center crop.

This module is read-only with respect to ``detections.csv``,
``pattern_attributes.csv``, and ``color_attributes.csv``.

Run as a module from the project root::

    uv run python -m src.clip_refine
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.crop_utils import center_crop, clip_bbox_to_image
from src.utils import ClipRefineConfig, load_clip_refine_config

logger = logging.getLogger(__name__)


__all__ = [
    "CSV_COLUMNS",
    "CSV_DTYPES",
    "DEFAULT_BATCH_SIZE",
    "InferFn",
    "UNCERTAIN_LABEL",
    "build_default_infer_fn",
    "clear_model_cache",
    "crop_for_clip",
    "load_clip_model",
    "normalize_yolo_category",
    "process_detections",
    "refine_batch",
    "run_clip_refinement",
    "select_refined_label",
    "write_clip_csv",
]


CSV_COLUMNS: tuple[str, ...] = (
    "image_id",
    "garment_id",
    "category_yolo",
    "category_refined",
    "refined_confidence",
    "all_scores",
)

CSV_DTYPES: dict[str, str] = {
    "image_id": "string",
    "garment_id": "int64",
    "category_yolo": "string",
    "category_refined": "string",
    "refined_confidence": "float64",
    "all_scores": "string",
}

UNCERTAIN_LABEL: str = "uncertain"
DEFAULT_BATCH_SIZE: int = 8


# Cache loaded (model, processor) pairs per model id so multiple
# ``run_clip_refinement`` calls in the same Python session don't reload.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


# A pure-Python signature so tests can swap in a fake without importing torch
# or transformers. Returns an ``(N, L)`` array of softmax probabilities.
InferFn = Callable[[list[Image.Image], list[str]], np.ndarray]


# --- taxonomy normalization -----------------------------------------------


def normalize_yolo_category(category: str) -> str:
    """Map a YOLO category string to the taxonomy-key form.

    DeepFashion2 weights shipped via ``Bingsu/adetailer`` use class names
    like ``short_sleeved_shirt``; the DeepFashion2 paper (and the starter
    taxonomy in ``config/clip_refine.yaml``) uses ``short sleeve top``.
    This helper bridges the two so taxonomy lookups succeed regardless of
    which form the detector produces. The mapping is idempotent — strings
    already in the human form pass through unchanged.
    """
    out = category.strip().lower().replace("_", " ").replace("sleeved", "sleeve")
    if out.endswith(" shirt"):
        out = out[: -len(" shirt")] + " top"
    return out


# --- crop -----------------------------------------------------------------


def crop_for_clip(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    center_crop_fraction: float,
    image_id: str,
    garment_id: int,
) -> Image.Image | None:
    """Open ``image_path``, slice the bbox, take the inner center crop, and
    return a PIL ``Image`` (RGB) ready for ``CLIPProcessor``.

    Returns ``None`` (and logs a warning) on any failure: missing file,
    undecodable image, bbox that clips to zero area, or zero-area center
    crop. Pixel values are not otherwise modified — CLIP's processor owns
    all resize / normalize / channel-scaling concerns.
    """
    if not image_path.exists():
        logger.warning(
            "Skipping %s (garment %s): image file not found at %s",
            image_id,
            garment_id,
            image_path,
        )
        return None

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        logger.warning(
            "Skipping %s (garment %s): cv2 failed to decode image",
            image_id,
            garment_id,
        )
        return None

    x1, y1, x2, y2 = clip_bbox_to_image(bbox, bgr.shape[:2])
    if x2 <= x1 or y2 <= y1:
        logger.warning(
            "Skipping %s (garment %s): bbox clipped to zero area",
            image_id,
            garment_id,
        )
        return None

    garment = bgr[y1:y2, x1:x2]
    inner = center_crop(garment, center_crop_fraction)
    if inner.size == 0 or inner.shape[0] == 0 or inner.shape[1] == 0:
        logger.warning(
            "Skipping %s (garment %s): zero-area center crop", image_id, garment_id
        )
        return None

    rgb = cv2.cvtColor(inner, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# --- CLIP model loading ---------------------------------------------------


def clear_model_cache() -> None:
    """Drop any cached (model, processor) pairs. Useful in tests."""
    _MODEL_CACHE.clear()


def load_clip_model(model_id: str, cache_dir: Path) -> tuple[Any, Any]:
    """Return ``(model, processor)``, loaded once per process per model id.

    The Hugging Face cache lives under ``cache_dir`` so weights are only
    downloaded on first run. Torch's manual seed is fixed and the model is
    put in ``eval`` mode so inference is deterministic across re-runs.
    """
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    import torch
    from transformers import CLIPModel, CLIPProcessor

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = CLIPModel.from_pretrained(model_id, cache_dir=str(cache_dir))
    model.eval()

    _MODEL_CACHE[model_id] = (model, processor)
    return model, processor


# --- inference ------------------------------------------------------------


def refine_batch(
    images: list[Image.Image],
    labels: list[str],
    prompt_template: str,
    model: Any,
    processor: Any,
) -> np.ndarray:
    """Run a CLIP forward pass over ``images`` against ``labels``.

    Returns an ``(N, L)`` array of softmax probabilities — each row is a
    distribution over ``labels``. All N images share the same L labels;
    callers must group by parent category before calling this.
    """
    import torch

    prompts = [prompt_template.format(label=label) for label in labels]
    inputs = processor(
        text=prompts, images=images, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)
    return probs.cpu().numpy()


def build_default_infer_fn(
    model_id: str,
    cache_dir: Path,
    prompt_template: str,
) -> InferFn:
    """Build an ``InferFn`` backed by a freshly loaded (or cached) CLIP model.

    Splitting this out from ``run_clip_refinement`` lets tests inject a fake
    ``InferFn`` and avoid importing torch / transformers entirely.
    """
    model, processor = load_clip_model(model_id, cache_dir)

    def infer(images: list[Image.Image], labels: list[str]) -> np.ndarray:
        return refine_batch(images, labels, prompt_template, model, processor)

    return infer


# --- thresholding ---------------------------------------------------------


def select_refined_label(
    scores: dict[str, float],
    threshold: float,
) -> tuple[str, float]:
    """Pick the top label if its probability is at or above ``threshold``,
    else return ``(UNCERTAIN_LABEL, top_probability)``.

    The returned confidence is always the top-label probability — even when
    the result is ``uncertain`` — so downstream consumers can sort by it.
    """
    if not scores:
        return UNCERTAIN_LABEL, 0.0
    top_label, top_prob = max(scores.items(), key=lambda kv: kv[1])
    top_prob = float(top_prob)
    if top_prob >= threshold:
        return top_label, top_prob
    return UNCERTAIN_LABEL, top_prob


# --- per-detection pipeline -----------------------------------------------


def _passthrough_row(
    image_id: str, garment_id: int, category_yolo: str
) -> dict[str, Any]:
    """Build a row for parents with no taxonomy entry or <2 sub-labels."""
    return {
        "image_id": image_id,
        "garment_id": garment_id,
        "category_yolo": category_yolo,
        "category_refined": category_yolo,
        "refined_confidence": 1.0,
        "all_scores": json.dumps({category_yolo: 1.0}),
    }


def process_detections(
    detections: pd.DataFrame,
    images_dir: Path,
    center_crop_fraction: float,
    taxonomy: dict[str, tuple[str, ...]] | dict[str, list[str]],
    threshold: float,
    batch_size: int,
    infer_fn: InferFn,
) -> tuple[list[dict[str, Any]], int]:
    """Run the full refinement over every detection row.

    Returns ``(rows, skipped)`` where ``rows`` is in the original input order
    (regardless of the batching grouping). Rows whose image is missing /
    unreadable / bbox-clipped are logged and excluded from ``rows``.
    """
    rows: dict[int, dict[str, Any]] = {}
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    records = detections.to_dict(orient="records")
    for idx, record in enumerate(
        tqdm(records, desc="Cropping garments", unit="garment")
    ):
        image_id = str(record["image_id"])
        garment_id = int(record["garment_id"])
        category_yolo = str(record["category"])
        taxonomy_key = normalize_yolo_category(category_yolo)
        sub_labels = taxonomy.get(taxonomy_key, ())

        if len(sub_labels) < 2:
            if taxonomy_key not in taxonomy:
                logger.info(
                    "No taxonomy entry for parent %r — passing %s through unrefined",
                    category_yolo,
                    image_id,
                )
            rows[idx] = _passthrough_row(image_id, garment_id, category_yolo)
            continue

        bbox = (
            float(record["bbox_x"]),
            float(record["bbox_y"]),
            float(record["bbox_w"]),
            float(record["bbox_h"]),
        )
        pil = crop_for_clip(
            images_dir / image_id, bbox, center_crop_fraction, image_id, garment_id
        )
        if pil is None:
            skipped += 1
            continue
        pending[taxonomy_key].append(
            {
                "idx": idx,
                "image": pil,
                "image_id": image_id,
                "garment_id": garment_id,
                "category_yolo": category_yolo,
            }
        )

    for taxonomy_key in sorted(pending.keys()):
        batch = pending[taxonomy_key]
        labels = list(taxonomy[taxonomy_key])
        for start in tqdm(
            range(0, len(batch), batch_size),
            desc=f"CLIP {taxonomy_key}",
            unit="batch",
        ):
            chunk = batch[start : start + batch_size]
            images = [item["image"] for item in chunk]
            probs = infer_fn(images, labels)
            if probs.shape != (len(images), len(labels)):
                raise ValueError(
                    f"infer_fn returned shape {probs.shape}; "
                    f"expected ({len(images)}, {len(labels)})"
                )
            for item, row_probs in zip(chunk, probs, strict=True):
                scores = {
                    label: float(p)
                    for label, p in zip(labels, row_probs, strict=True)
                }
                refined, confidence = select_refined_label(scores, threshold)
                rows[item["idx"]] = {
                    "image_id": item["image_id"],
                    "garment_id": item["garment_id"],
                    "category_yolo": item["category_yolo"],
                    "category_refined": refined,
                    "refined_confidence": confidence,
                    "all_scores": json.dumps(scores),
                }

    return [rows[i] for i in sorted(rows.keys())], skipped


def write_clip_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    """Write refinement rows to CSV with the canonical schema and dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    df = df.astype(CSV_DTYPES)
    df.to_csv(path, index=False)
    return df


def _print_summary(total_input: int, skipped: int, df: pd.DataFrame) -> None:
    print()
    print("=== Pipeline 1 Step 2C: CLIP zero-shot refinement summary ===")
    print(f"Total detections in input: {total_input}")
    print(f"Processed:                 {len(df)}")
    print(f"Skipped (errors):          {skipped}")
    if len(df) == 0:
        print()
        return

    refined_mask = df["category_refined"] != UNCERTAIN_LABEL
    refined_n = int(refined_mask.sum())
    uncertain_n = len(df) - refined_n
    total = len(df)
    print(
        f"Refined:                   {refined_n}  "
        f"({100.0 * refined_n / total:5.1f}%)"
    )
    print(
        f"Uncertain:                 {uncertain_n}  "
        f"({100.0 * uncertain_n / total:5.1f}%)"
    )

    print()
    print("Refined-label distribution per parent category:")
    for parent in sorted(df["category_yolo"].unique()):
        subset = df[df["category_yolo"] == parent]
        counts = Counter(subset["category_refined"].tolist())
        print(f"  {parent} ({len(subset)} garments):")
        for label, n in counts.most_common():
            pct = 100.0 * n / len(subset)
            print(f"    {label:<24} {n:>5}  ({pct:5.1f}%)")
    print()


def run_clip_refinement(
    config: ClipRefineConfig,
    infer_fn: InferFn | None = None,
) -> pd.DataFrame:
    """Execute the full refinement pipeline. Returns the result DataFrame.

    Passing ``infer_fn`` bypasses CLIP loading; this is how tests exercise
    the orchestration code without pulling in torch / transformers weights.
    """
    detections = pd.read_csv(config.detections_csv)

    if infer_fn is None:
        infer_fn = build_default_infer_fn(
            config.model_id, config.model_cache_dir, config.prompt_template
        )

    rows, skipped = process_detections(
        detections,
        config.images_dir,
        config.center_crop_fraction,
        config.taxonomy,
        config.threshold,
        config.batch_size,
        infer_fn,
    )

    df = write_clip_csv(rows, config.output_csv)
    _print_summary(len(detections), skipped, df)
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_clip_refine_config()
    run_clip_refinement(config)


if __name__ == "__main__":
    main()
