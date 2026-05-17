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

### Tests

```bash
uv run pytest
```
