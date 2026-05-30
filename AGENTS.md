# AGENTS.md — Fashion Trend Prediction

Guidelines for AI agents and contributors working in this repository.

---

## Package Management

Always use `uv` for all Python tooling. Never use `pip`, `pip install`, `python -m pip`, or `conda`.

| Task | Command |
|---|---|
| Run a script | `uv run python src/train.py` |
| Add a dependency | `uv add <package>` |
| Add a dev dependency | `uv add --dev <package>` |
| Remove a dependency | `uv remove <package>` |
| Run tests | `uv run pytest` |
| Run a one-off tool | `uv run <tool>` |

`uv` manages the `.venv` and `uv.lock` automatically. Never manually activate or modify the venv.

---

## Project Structure

```
FashionTrendPrediction/
├── src/
│   ├── data.py       # data loading and preprocessing
│   ├── train.py      # model training logic
│   └── predict.py    # inference / prediction logic
├── tests/            # pytest test files (mirror src/ structure)
├── data/             # raw and processed datasets (not committed)
├── analysis/         # exploratory notebooks and outputs (not committed)
├── pyproject.toml
└── uv.lock
```

---

## Code Standards

### Modularity and Reusability
- Write small, single-purpose functions. Avoid monolithic scripts.
- Expose functionality through clearly named functions, not top-level script logic.
- Entry points (CLI calls, main guards) belong at the bottom of a file inside `if __name__ == "__main__":`.
- Prefer function arguments over global state or hardcoded paths.

### Compatibility
- All new code must be compatible with Python 3.13+.
- New modules in `src/` must integrate cleanly with the existing pipeline (`data.py` → `train.py` → `predict.py`).
- Shared utilities (path helpers, config loaders, etc.) go in `src/utils.py` to avoid duplication.

### Style
- Follow PEP 8. Keep lines under 100 characters.
- Use type hints on all function signatures.
- Do not add comments that restate what the code does — only comment on non-obvious intent or constraints.

---

## Testing

- Every new feature or function must have a corresponding test.
- Tests live in `tests/` and mirror the `src/` structure (e.g. `src/data.py` → `tests/test_data.py`).
- Run the full suite before considering any task complete: `uv run pytest`
- Tests must pass without network access or access to the `data/` directory — use fixtures or small synthetic data.
- Use `pytest` fixtures for shared setup. Do not rely on test execution order.

---

## Adding New Features

When adding a new feature or capability:

1. Implement it as one or more functions in the appropriate `src/` module (or a new module if the scope warrants it).
2. Write tests in `tests/` covering the happy path and key edge cases.
3. Update this file under the **Feature Registry** section below with a one-line description.
4. If a new dependency is introduced, add it with `uv add <package>` and commit the updated `pyproject.toml` and `uv.lock`.

---

## Feature Registry

Document every significant feature or capability added to the project here. Keep entries concise.

| Feature | Module | Description |
|---|---|---|
| Data loading skeleton | `src/data.py` | Base imports and structure for loading and preprocessing fashion datasets via pandas |
| Training logic skeleton | `src/train.py` | Placeholder for model training pipeline |
| Prediction logic skeleton | `src/predict.py` | Placeholder for inference pipeline |
| Config & path helpers | `src/utils.py` | YAML config loader and project-root path resolution shared across pipelines |
| Pipeline 1: garment detection | `src/detect.py` | DeepFashion2 YOLOv8 fashion filter — splits raw images into accepted/rejected and emits per-detection CSV (see README) |
| Pipeline 1 Step 2B: pattern complexity | `src/pattern.py` | Laplacian-variance scoring of each garment bbox; quantile-bucketed into plain/subtle/patterned and written to `pattern_attributes.csv` |
| Pipeline 1 Step 2A: dominant color | `src/color.py` | LAB K-means (K=3, seeded) over each garment crop; assigns dominant RGB + nearest curated-palette name and writes `color_attributes.csv` |
| Shared crop helpers | `src/crop_utils.py` | bbox clipping and inner-center-crop primitives reused by both Step 2A and Step 2B so the two attribute extractors see the same garment region |
| Pipeline 1 Step 2C: CLIP zero-shot refinement | `src/clip_refine.py` | Parent-conditioned fine-grained sub-category labelling via `openai/clip-vit-large-patch14`; writes `clip_refinement.csv` with top label, confidence, and the full probability distribution per garment |
| Pipeline 1 join: combined attributes | `src/join.py` | Left-joins `detections.csv`, `color_attributes.csv`, `pattern_attributes.csv`, and `clip_refinement.csv` on `(image_id, garment_id)` and writes `yolo_fashion_attributes.csv` (one row per garment, every column preserved) |
| Pipeline 2: consumer behavior classification | `src/popularity.py` | Loads the Kim et al. Instagram fashion .xlsx, computes a (Likes+comments) popularity score (train-only min-max), builds engagement / hashtag-multihot / behavioral feature groups, and runs an ablation of a RandomForestClassifier across each group and the combined set — writing `popularity_score.csv`, `ablation_results.csv`, per-run confusion / importance plots, classification reports, and `model_combined.joblib` |
| Pipeline 1 visualisations | `src/pipeline1_viz.py` | Produces six descriptive PNGs from `yolo_fashion_attributes.csv` (acceptance/rejection, garment-category, dominant-colour using the curated palette, pattern complexity, refined-sub-label small multiples per parent, CLIP uncertainty per parent) into `data/processed/figures/pipeline1/` for the Methodology and Results thesis chapters. Reuses the colour palette from `config/color.yaml`; configured by `config/pipeline1_viz.yaml` (per-plot suppress flag + parent threshold). Deterministic, overwrites cleanly on re-run |
| Pipeline 1 manual validation app | `src/validate.py` | Tkinter + Pillow app that randomly samples garments from `yolo_fashion_attributes.csv`, draws each bbox on the source image, and asks the operator to validate three layers in order: bbox quality (completely correct / somewhat correct / incorrect — picking incorrect disables the label drop-downs), parent category (drop-down constrained to YOLO classes the pipeline emitted, plus a `no_clothes` option that auto-fills the subcategory), and refined subcategory (filtered to sub-labels seen under the chosen parent). Streams selections to `validations/YYYY-MM-DD-val/validations.csv` row by row; on Exit (or after the last item) writes `accuracy_summary.csv`, `bbox_quality_per_category.csv`, `per_category_accuracy.csv`, `per_subcategory_accuracy.csv`, `uncertain_distribution.csv` and `uncertain_user_labels.csv` to the same dated folder. Incorrect-bbox rows are excluded from category/subcategory accuracy; CLIP-uncertain rows are additionally excluded from subcategory accuracy and routed to the uncertain CSVs instead. Configured by `config/validate.yaml`. macOS-tested |

---

## Dependencies

Current production dependencies (see `pyproject.toml`):

| Package | Purpose |
|---|---|
| `pandas` | Tabular data loading and preprocessing |
| `ultralytics` | YOLO-based object detection for fashion item recognition |
| `pillow` | Image decoding / corrupt-image verification |
| `tqdm` | Progress bars for batch image processing |
| `pyyaml` | Loading pipeline config from `config/*.yaml` |
| `huggingface_hub` | Cached download of DeepFashion2 YOLOv8 weights |
| `opencv-python` | Image cropping, grayscale conversion, Laplacian operator for pattern complexity scoring |
| `numpy` | Array math and quantile thresholding for pattern bucketing |
| `scikit-learn` | K-means clustering of LAB pixels for dominant color extraction |
| `transformers` | CLIP model + processor for zero-shot sub-category refinement |
| `torch` | Tensor backend for CLIP inference (eval-mode, no grad) |
| `openpyxl` | Reading the Kim et al. fashion `.xlsx` dataset for Pipeline 2 |
| `matplotlib` | Confusion-matrix and feature-importance plotting for Pipeline 2 |
| `seaborn` | Heatmap rendering for Pipeline 2 confusion matrices |
| `joblib` | Persisting the trained Pipeline 2 RandomForest model artifact |
