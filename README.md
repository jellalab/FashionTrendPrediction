# Fashion Trend Prediction

MSc thesis pipeline for extracting fashion attributes from Instagram images.

## Pipeline 1 — Garment detection / fashion filter

[src/detect.py](src/detect.py) splits a folder of raw images into accepted
(≥1 garment detected at or above the confidence threshold) and
`rejected_non_fashion`, and writes one row per detection to
`detections.csv`.

### Usage

```bash
uv run python -m src.detect
```

Configuration lives in [config/detection.yaml](config/detection.yaml)
(input/output paths, confidence threshold, model identifier). No CLI flags.

### Model

The detector uses **`deepfashion2_yolov8s-seg.pt`** — a YOLOv8s
segmentation model trained on DeepFashion2 by **Bingsu**, hosted on
Hugging Face:

- Repo: <https://huggingface.co/Bingsu/adetailer>
- File: `deepfashion2_yolov8s-seg.pt` (≈24 MB)
- Reported metrics on DeepFashion2 realistic clothes: bbox mAP@50 = 0.849,
  bbox mAP@50-95 = 0.763.
- Classes (13, read at runtime from `model.names`, never hardcoded):
  short_sleeved_shirt, long_sleeved_shirt, short_sleeved_outwear,
  long_sleeved_outwear, vest, sling, shorts, trousers, skirt,
  short_sleeved_dress, long_sleeved_dress, vest_dress, sling_dress.

The weights are downloaded via `huggingface_hub` to `data/models/` on first
run and reused on subsequent runs.

### Outputs

- `data/processed/accepted/` — copies of accepted images (originals untouched).
- `data/processed/rejected_non_fashion/` — copies of rejected images.
- `data/processed/detections.csv` — one row per detection. Columns:
  `image_id`, `garment_id`, `category`, `confidence`,
  `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` (pixel coordinates, top-left origin).

Re-running the script clears and rewrites both output folders and the CSV.

## Pipeline 1 Step 2A — Dominant color extraction

[src/color.py](src/color.py) reads `detections.csv`, crops each garment,
takes the same inner center crop as Step 2B, converts the crop to
CIELAB, runs K-means (K=3, `random_state=42`) over the pixels, and
records the largest cluster centroid as the garment's dominant color.
The dominant LAB centroid is then matched to the nearest entry in a
curated fashion palette (config-driven) by Euclidean distance in LAB.

This module is read-only with respect to `detections.csv`.

### Usage

```bash
uv run python -m src.color
```

Configuration lives in [config/color.yaml](config/color.yaml)
(input/output paths, `center_crop_fraction`, K-means parameters,
palette). No CLI flags.

### Outputs

- `data/processed/color_attributes.csv` — one row per input detection
  with columns: `image_id`, `garment_id`, `dominant_r`, `dominant_g`,
  `dominant_b`, `dominant_color_name`, `palette_rgb` (top-3 RGB
  centroids serialized as a JSON list).

Re-running rewrites the CSV; `random_state=42` makes output bit-stable
across runs. Rows whose image is missing, corrupt, or whose bbox clips
to zero area are logged and skipped (reported in the console summary,
absent from the output).

## Pipeline 1 Step 2B — Pattern complexity scoring

[src/pattern.py](src/pattern.py) reads `detections.csv`, crops each
garment from its source image, takes an inner center crop (default 60%
on each axis) to reduce skin/background contamination, and computes the
variance of the grayscale Laplacian as a scalar measure of visual
complexity. Garments are then bucketed into `plain` / `subtle` /
`patterned` using dataset-relative quantile thresholds.

This module measures complexity only — it does **not** classify pattern
*type* (stripes, florals, plaid). It does not modify `detections.csv`.

### Usage

```bash
uv run python -m src.pattern
```

Configuration lives in [config/pattern.yaml](config/pattern.yaml)
(input/output paths, `center_crop_fraction`, quantile thresholds).
No CLI flags.

### Outputs

- `data/processed/pattern_attributes.csv` — one row per input detection
  with columns: `image_id`, `garment_id`, `laplacian_variance`,
  `pattern_class` (`plain` / `subtle` / `patterned`).

Re-running the script rewrites the CSV. Rows whose image is missing,
corrupt, or whose bbox clips to zero area are logged and skipped
(reported in the console summary, absent from the output).

## Tests

```bash
uv run pytest
```
