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
