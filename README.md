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

## Pipeline 1 Step 2C — CLIP zero-shot garment refinement

[src/clip_refine.py](src/clip_refine.py) reads `detections.csv`, crops
each garment from its source image with the same inner center crop used
by Steps 2A/2B, and runs `openai/clip-vit-large-patch14` in zero-shot
mode against a parent-conditioned taxonomy of fine-grained sub-labels.
The top-probability sub-label is recorded — unless that probability is
below `threshold` (default 0.4), in which case the row is labelled
`uncertain`.

This module is read-only with respect to `detections.csv`,
`pattern_attributes.csv`, and `color_attributes.csv`.

### Usage

```bash
uv run python -m src.clip_refine
```

Configuration lives in [config/clip_refine.yaml](config/clip_refine.yaml)
(input/output paths, `center_crop_fraction`, CLIP model id and cache,
prompt template, confidence threshold, batch size, and the taxonomy).
The taxonomy is the user-editable list of candidate sub-labels per YOLO
parent category. No CLI flags.

### Outputs

- `data/processed/clip_refinement.csv` — one row per input detection
  with columns: `image_id`, `garment_id`, `category_yolo` (the parent
  category from Step 1), `category_refined` (CLIP top label or
  `uncertain`), `refined_confidence` (float in [0, 1]), `all_scores`
  (JSON-serialized dict of `{sub_label: probability}` for every
  candidate considered).

Refinement is strictly zero-shot — CLIP is not fine-tuned. Sub-labels
considered for a given row are exactly the ones listed under its YOLO
parent in the taxonomy; no other sub-labels can be assigned. Rows whose
image is missing, corrupt, or whose bbox clips to zero area are logged
and skipped (reported in the console summary, absent from the output).

## Pipeline 1 join — combined per-garment attribute table

[src/join.py](src/join.py) merges the four per-garment CSVs produced by
Steps 1, 2A, 2B, and 2C into a single
`data/processed/yolo_fashion_attributes.csv`. The join is
left-anchored on `detections.csv` (the source of truth for the garment
universe) and matches on `(image_id, garment_id)`, so any garment dropped
by a downstream step (missing image, decode failure, zero-area crop)
still appears in the output with empty attribute columns. Every column
from every input CSV is preserved.

This module is read-only with respect to all four input CSVs.

### Usage

```bash
uv run python -m src.join
```

Configuration lives in [config/join.yaml](config/join.yaml)
(input paths for the four upstream CSVs and the output path). No CLI flags.

### Outputs

- `data/processed/yolo_fashion_attributes.csv` — one row per detection
  with columns: `image_id`, `garment_id`, `category`, `confidence`,
  `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, `dominant_r`, `dominant_g`,
  `dominant_b`, `dominant_color_name`, `palette_rgb`,
  `laplacian_variance`, `pattern_class`, `category_yolo`,
  `category_refined`, `refined_confidence`, `all_scores`.

Re-running rewrites the CSV. The script logs a warning per downstream
CSV that is missing any detection row, and prints a per-source coverage
summary on completion.

## Pipeline 1 manual validation — sample-based label review

[src/validate.py](src/validate.py) launches a small Tkinter desktop app that
draws a uniform random sample of garment detections from
`yolo_fashion_attributes.csv` and asks the operator to label each one. For
every sampled row it shows the source image from `data/raw/sample_images/`
with the garment's bbox outlined and presents two drop-downs:

- **Category** — `no_clothes` (for bboxes that contain no garment at all —
  a model false positive) followed by the unique parent values present in
  `yolo_fashion_attributes.csv:category` (i.e. the DeepFashion2 YOLO classes
  the pipeline actually emitted). Picking `no_clothes` auto-fills the
  subcategory and disables that drop-down.
- **Subcategory** — the unique values of `category_refined` *for the chosen
  parent*. The list re-populates on category change so only valid CLIP
  sub-labels (plus `uncertain` where present) are offered.

The operator does not see the model's prediction during labelling.

### Usage

```bash
uv run python -m src.validate
```

Configuration lives in [config/validate.yaml](config/validate.yaml)
(input CSV / images dir, output root, optional `random_seed` for a
reproducible sample, `max_items` to cap the session, and the displayed
image size). No CLI flags.

### Outputs

All written under `validations/YYYY-MM-DD-val/` (the dated folder is created
on first launch of the day and re-used by later runs on the same date):

- `validations.csv` — appended one row per submitted item with columns:
  `image_id`, `garment_id`, `model_category`, `model_subcategory`,
  `user_category`, `user_subcategory`, `category_correct`,
  `subcategory_correct`, `timestamp`. `subcategory_correct` is left blank
  when the model said `uncertain` — the refiner has explicitly abstained
  and is not scored against any user answer. Streaming the rows means a
  crash or early exit still preserves every validation completed up to
  that point.
- `accuracy_summary.csv` — one row: `items_validated`,
  `category_accuracy`, `subcategory_items_evaluated` (non-uncertain rows
  only), `subcategory_accuracy_excl_uncertain`,
  `both_correct_accuracy_excl_uncertain`, `uncertain_items`,
  `uncertain_rate`.
- `per_category_accuracy.csv` — `model_category`, `n`, `accuracy`
  (over every row; uncertain refinement does not affect the parent
  decision).
- `per_subcategory_accuracy.csv` — `model_subcategory`, `n`, `accuracy`
  (uncertain rows are excluded entirely).
- `uncertain_distribution.csv` — per parent: `n_total`, `n_uncertain`,
  `uncertain_rate`. Shows which YOLO classes the CLIP refiner most often
  abstains on.
- `uncertain_user_labels.csv` — per `(model_category, user_category,
  user_subcategory)`: `count` of uncertain detections that the operator
  resolved to that label. Useful for diagnosing whether the refiner's
  abstentions cluster around real garment types it should be able to
  recognise (i.e. missing taxonomy entries).

Pressing **Exit** (or closing the window) triggers the summary rewrite
immediately; the same happens when the last sampled item is submitted.

The app is macOS-tested — Tkinter ships with the uv-managed Python and
Pillow is already a project dependency, so no extra install is required.

## Pipeline 2 — Consumer behavior classification (with ablation)

[src/popularity.py](src/popularity.py) classifies each post in the Kim et al.
*Fashion Conversation Data on Instagram* dataset into one of four
`BrandCategory` values — `Designer`, `Small couture`, `High street`,
`Mega couture` — from post metadata. It loads the raw `.xlsx`, computes a
swappable popularity score, builds three feature groups, and trains a
`RandomForestClassifier` five times in an ablation: each group alone, the
H2 engagement-plus-visual combination (no hashtags), and the all-groups
combination (with hashtags) — all on the same stratified train/test split.

### Usage

```bash
uv run python -m src.popularity
```

Configuration lives in [config/popularity.yaml](config/popularity.yaml)
(input path, popularity formula weight, hashtag `top_n`, split ratio,
Random Forest hyperparameters, plot settings). No CLI flags.

### Feature groups

- **Group A — engagement / reach** (5 features): `Likes`, `comments`,
  `Followers`, `MediaCount`, `popularity_score_norm`. `popularity_score`
  is `(Likes + comments) * weight` (default `weight=0.5` = mean), and its
  min-max normalization is fit on the train partition only.
- **Group B — hashtags** (top-N multi-hot + 1): the top-N most frequent
  hashtags computed on TRAIN ONLY are encoded as 0/1 columns, plus a
  `hashtag_count` column. `top_n` defaults to 100. Hashtags seen only in
  the test partition are ignored. Null `Hashtags` are treated as an empty
  list — not dropped.
- **Group C — behavioral / visual** (20 features): `Selfie`, `BodySnap`,
  `Marketing`, `ProductOnly`, `NonFashion`, `Face`, `Logo`, `BrandLogo`,
  `Smile`, `Outdoor`, `NumberOfPeople`, `NumberOfFashionProduct`,
  `Anger`, `Contempt`, `Disgust`, `Fear`, `Happiness`, `Neutral`,
  `Sadness`, `Surprise` — used as-is.

`BrandName`, `UserId`, `Link`, `ImgURL`, `Caption`, `CreationTime` are
hard-excluded from every feature matrix and the exclusion is asserted
before training. The trailing-space `Comments ` column in the input is
stripped to `Comments` and explicitly renamed to `comments` at load time.

### Outputs

All written under `data/processed/popularity/`:

- `popularity_score.csv` — one row per post with the raw and normalized
  popularity score, all engagement / behavioral feature columns, the
  `hashtag_count`, the `BrandCategory` target, and a `split` column
  (`train` / `test`).
- `hashtag_features.csv` — the train+test hashtag multi-hot matrix
  consumed by the hashtag-based ablation runs: one `hashtag_{tag}` column
  per top-N hashtag plus `hashtag_count`, augmented with a `split` column
  marking each row as `train` or `test`. Intentionally excludes the
  target / brand / identifier columns so the file is safe to inspect
  alongside the leakage discussion. Row count equals train + test rows.
- `ablation_results.csv` — one row per ablation run (`group_a_engagement`,
  `group_b_hashtags`, `group_c_behavioral`, `combined_engagement_visual`,
  `combined_all_with_hashtags`) with columns `run_name`,
  `features_used_count`, `accuracy`, `macro_f1`, `weighted_f1`, and
  `f1_{class}` per class. The `combined_engagement_visual` row is the
  H2 result; the `combined_all_with_hashtags` row is retained alongside
  it so the leakage gap (the dataset is brand-tag-collected, so hashtags
  encode brand identity) is visible directly in the table.
- `confusion_matrix_{run}.png`, `feature_importance_{run}.png` (top 20
  features), and `classification_report_{run}.txt` for each of the five
  runs.
- `model_combined.joblib` — the trained `combined_all_with_hashtags`
  Random Forest bundled with its feature names and class labels.

The console summary reports dataset size, the full and test-split class
distributions, rows imputed/dropped with reasons, and the ablation
comparison table — with macro-F1 (not bare accuracy) as the headline
metric, given the imbalanced target distribution.

Re-running is deterministic (`random_state=42` for both the split and
the Random Forest).

## Tests

```bash
uv run pytest
```
