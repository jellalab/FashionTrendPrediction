"""Manual validation tool for the joined Pipeline 1 attribute CSV.

Iterates over each garment detection row in ``yolo_fashion_attributes.csv``,
displays the source image from ``data/raw/sample_images/`` with the garment's
bbox highlighted, and asks the operator to pick the correct
``(category, subcategory)`` from drop-downs that are constrained to the
labels the pipeline can actually emit (parent values seen in ``category``;
refined values seen in ``category_refined`` for the chosen parent).

Validations stream to ``validations/YYYY-MM-DD-val/validations.csv`` one row
at a time so partial work survives a crash. On ``Exit`` (or when every item
has been processed) per-class and overall accuracy CSVs are written to the
same dated folder.

GUI is Tkinter + Pillow — both already in the project's runtime dependencies
and shipped with the uv-managed Python on macOS.

Run from the project root::

    uv run python -m src.validate
"""

from __future__ import annotations

import logging
import random
import tkinter as tk
from datetime import date, datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageTk

from src.utils import ValidateConfig, load_validate_config



logger = logging.getLogger(__name__)


__all__ = [
    "ValidationApp",
    "build_category_choices",
    "build_subcategory_choices",
    "build_validation_items",
    "compute_accuracy",
    "make_validation_dir",
    "run_validation",
    "write_accuracy_summary",
]


VALIDATIONS_FILENAME = "validations.csv"
SUMMARY_FILENAME = "accuracy_summary.csv"
PER_CATEGORY_FILENAME = "per_category_accuracy.csv"
PER_SUBCATEGORY_FILENAME = "per_subcategory_accuracy.csv"
UNCERTAIN_DISTRIBUTION_FILENAME = "uncertain_distribution.csv"
UNCERTAIN_USER_LABELS_FILENAME = "uncertain_user_labels.csv"
BBOX_QUALITY_PER_CATEGORY_FILENAME = "bbox_quality_per_category.csv"

VALIDATION_COLUMNS: tuple[str, ...] = (
    "image_id",
    "garment_id",
    "model_category",
    "model_subcategory",
    "bbox_quality",
    "user_category",
    "user_subcategory",
    "category_correct",
    "subcategory_correct",
    "timestamp",
)

# Pipeline-side literal written by src/clip_refine.py when the top softmax
# score is below threshold. Rows with this model_subcategory are excluded
# from the subcategory accuracy denominator and routed to the uncertain
# distribution / user-label CSVs instead.
UNCERTAIN_LABEL = "uncertain"

# Operator-side option offered when the highlighted bbox contains no
# garment (i.e. the YOLO detector fired on a non-garment region). Recorded
# verbatim in both user_category and user_subcategory; counts as a model
# miss for category accuracy.
NO_CLOTHES_LABEL = "no_clothes"

# Bounding-box quality is validated as a separate dimension before the
# operator labels the garment itself. When the bbox is judged incorrect
# the category/subcategory drop-downs are disabled and that row is
# excluded from both category and subcategory accuracy — there is no
# meaningful garment region to label.
BBOX_QUALITY_COMPLETELY_CORRECT = "completely_correct"
BBOX_QUALITY_SOMEWHAT_CORRECT = "somewhat_correct"
BBOX_QUALITY_INCORRECT = "incorrect"
BBOX_QUALITY_VALUES: tuple[str, ...] = (
    BBOX_QUALITY_COMPLETELY_CORRECT,
    BBOX_QUALITY_SOMEWHAT_CORRECT,
    BBOX_QUALITY_INCORRECT,
)
_BBOX_QUALITY_DISPLAY: tuple[tuple[str, str], ...] = (
    ("Completely correct", BBOX_QUALITY_COMPLETELY_CORRECT),
    ("Somewhat correct", BBOX_QUALITY_SOMEWHAT_CORRECT),
    ("Incorrect", BBOX_QUALITY_INCORRECT),
)

_BBOX_OUTLINE = "#FF3B30"
_BBOX_WIDTH = 6


# --- pure helpers ---------------------------------------------------------


def build_category_choices(df: pd.DataFrame) -> list[str]:
    """Return the category drop-down options.

    ``NO_CLOTHES_LABEL`` is prepended so the operator can flag bboxes that
    contain no garment at all (a model false positive). The remaining
    entries are the sorted unique parents observed in the CSV.
    """
    parents = sorted(df["category"].dropna().astype(str).unique().tolist())
    return [NO_CLOTHES_LABEL, *parents]


def build_subcategory_choices(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each parent category to the sorted unique refined sub-labels seen.

    The CLIP refinement step uses a parent-conditioned taxonomy, so the
    sub-labels offered for each parent are strictly the values observed
    under that parent (plus ``uncertain`` when present). The synthetic
    ``no_clothes`` parent gets a single matching sub-label so the GUI can
    auto-fill the second drop-down and skip a redundant click.
    """
    sub = df.dropna(subset=["category"])
    out: dict[str, list[str]] = {NO_CLOTHES_LABEL: [NO_CLOTHES_LABEL]}
    for parent, group in sub.groupby("category", sort=False):
        labels = sorted(
            {str(x) for x in group["category_refined"].dropna()}
        )
        out[str(parent)] = labels
    return out


def make_validation_dir(root: Path, today: date | None = None) -> Path:
    """Create and return ``<root>/YYYY-MM-DD-val/``."""
    day = today or date.today()
    folder = root / f"{day.isoformat()}-val"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def build_validation_items(
    df: pd.DataFrame,
    random_seed: int | None,
    max_items: int | None,
) -> list[dict[str, Any]]:
    """Project the joined CSV into the per-row dicts the GUI iterates over.

    Rows without a parent ``category`` are dropped — there is nothing to
    validate. Items are always randomised: when ``random_seed`` is ``None``
    a fresh OS-seeded RNG is used (a different sample every session); when
    set, the sample is reproducible across runs. ``max_items`` then caps
    the session, giving a uniform random sample without replacement.
    """
    items: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        if pd.isna(row.get("category")):
            continue
        refined = row.get("category_refined")
        items.append(
            {
                "image_id": str(row["image_id"]),
                "garment_id": int(row["garment_id"]),
                "bbox_x": float(row["bbox_x"]),
                "bbox_y": float(row["bbox_y"]),
                "bbox_w": float(row["bbox_w"]),
                "bbox_h": float(row["bbox_h"]),
                "model_category": str(row["category"]),
                "model_subcategory": "" if pd.isna(refined) else str(refined),
            }
        )

    random.Random(random_seed).shuffle(items)
    if max_items is not None:
        items = items[:max_items]
    return items


def compute_accuracy(validations_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return accuracy/distribution frames covering all three validation layers.

    Three independent layers are evaluated:

    1. **Bounding-box quality** — operator-rated as completely correct,
       somewhat correct, or incorrect. Reported as raw counts/rates in the
       summary and broken down per parent in ``bbox_quality_per_category``.
    2. **Category** — model parent vs. user parent. Rows with an
       ``incorrect`` bbox are excluded; there is no meaningful garment
       region to label, so penalising the parent decision would conflate
       "wrong class" with "wrong region".
    3. **Subcategory** — model refined vs. user refined. Excludes
       ``incorrect`` bbox rows *and* rows where the model itself said
       ``uncertain`` (the CLIP refiner explicitly abstained). Those
       uncertain rows are surfaced separately in ``uncertain_distribution``
       and ``uncertain_user_labels`` instead.
    """
    keys = (
        "summary",
        "bbox_quality_per_category",
        "per_category",
        "per_subcategory",
        "uncertain_distribution",
        "uncertain_user_labels",
    )
    if validations_df.empty:
        return {k: pd.DataFrame() for k in keys}

    df = validations_df.copy()
    df["model_category"] = df["model_category"].astype(str)
    df["model_subcategory"] = df["model_subcategory"].astype(str)
    df["user_category"] = df["user_category"].fillna("").astype(str)
    df["user_subcategory"] = df["user_subcategory"].fillna("").astype(str)
    df["bbox_quality"] = df["bbox_quality"].fillna("").astype(str)

    n = int(len(df))
    is_completely = df["bbox_quality"] == BBOX_QUALITY_COMPLETELY_CORRECT
    is_somewhat = df["bbox_quality"] == BBOX_QUALITY_SOMEWHAT_CORRECT
    is_bbox_incorrect = df["bbox_quality"] == BBOX_QUALITY_INCORRECT
    n_completely = int(is_completely.sum())
    n_somewhat = int(is_somewhat.sum())
    n_bbox_incorrect = int(is_bbox_incorrect.sum())
    n_valid_bbox = n - n_bbox_incorrect

    valid = df.loc[~is_bbox_incorrect].copy()
    if n_valid_bbox:
        valid["category_correct"] = (
            valid["model_category"] == valid["user_category"]
        ).astype(int)
        is_uncertain_valid = valid["model_subcategory"] == UNCERTAIN_LABEL
        sub_match = (
            valid["model_subcategory"] == valid["user_subcategory"]
        ).astype("Int64")
        valid["subcategory_correct"] = sub_match.where(~is_uncertain_valid, pd.NA)
        sub_eval = valid.loc[~is_uncertain_valid]
        n_sub_eval = int(len(sub_eval))
        category_accuracy = float(valid["category_correct"].mean())
        if n_sub_eval:
            subcategory_accuracy = float(
                sub_eval["subcategory_correct"].astype(float).mean()
            )
            both_correct = float(
                (
                    (sub_eval["category_correct"] == 1)
                    & (sub_eval["subcategory_correct"].astype("Int64") == 1)
                ).mean()
            )
        else:
            subcategory_accuracy = float("nan")
            both_correct = float("nan")
    else:
        category_accuracy = float("nan")
        subcategory_accuracy = float("nan")
        both_correct = float("nan")
        n_sub_eval = 0
        sub_eval = valid

    n_uncertain_total = int((df["model_subcategory"] == UNCERTAIN_LABEL).sum())

    summary = pd.DataFrame(
        [
            {
                "items_validated": n,
                "bbox_completely_correct": n_completely,
                "bbox_somewhat_correct": n_somewhat,
                "bbox_incorrect": n_bbox_incorrect,
                "bbox_completely_correct_rate": float(n_completely / n) if n else 0.0,
                "bbox_somewhat_correct_rate": float(n_somewhat / n) if n else 0.0,
                "bbox_incorrect_rate": float(n_bbox_incorrect / n) if n else 0.0,
                "items_with_valid_bbox": n_valid_bbox,
                "category_accuracy_valid_bbox": category_accuracy,
                "subcategory_items_evaluated": n_sub_eval,
                "subcategory_accuracy_excl_uncertain": subcategory_accuracy,
                "both_correct_accuracy_excl_uncertain": both_correct,
                "uncertain_items": n_uncertain_total,
                "uncertain_rate": float(n_uncertain_total / n) if n else 0.0,
            }
        ]
    )

    bbox_quality_per_category = (
        df.assign(
            _completely=is_completely.astype(int),
            _somewhat=is_somewhat.astype(int),
            _incorrect=is_bbox_incorrect.astype(int),
        )
        .groupby("model_category", sort=True)
        .agg(
            n_total=("_completely", "size"),
            n_completely_correct=("_completely", "sum"),
            n_somewhat_correct=("_somewhat", "sum"),
            n_incorrect=("_incorrect", "sum"),
        )
        .reset_index()
    )
    bbox_quality_per_category["completely_correct_rate"] = (
        bbox_quality_per_category["n_completely_correct"]
        / bbox_quality_per_category["n_total"]
    )
    bbox_quality_per_category["incorrect_rate"] = (
        bbox_quality_per_category["n_incorrect"]
        / bbox_quality_per_category["n_total"]
    )

    if n_valid_bbox:
        per_category = (
            valid.groupby("model_category", sort=True)
            .agg(
                n=("category_correct", "size"),
                accuracy=("category_correct", "mean"),
            )
            .reset_index()
        )
    else:
        per_category = pd.DataFrame(columns=["model_category", "n", "accuracy"])

    if n_sub_eval:
        per_subcategory = (
            sub_eval.groupby("model_subcategory", sort=True)
            .agg(
                n=("subcategory_correct", "size"),
                accuracy=(
                    "subcategory_correct",
                    lambda s: float(pd.Series(s).astype(float).mean()),
                ),
            )
            .reset_index()
        )
    else:
        per_subcategory = pd.DataFrame(columns=["model_subcategory", "n", "accuracy"])

    is_uncertain_all = df["model_subcategory"] == UNCERTAIN_LABEL
    uncertain_distribution = (
        df.assign(_is_uncertain=is_uncertain_all.astype(int))
        .groupby("model_category", sort=True)
        .agg(n_total=("_is_uncertain", "size"), n_uncertain=("_is_uncertain", "sum"))
        .reset_index()
    )
    uncertain_distribution["uncertain_rate"] = (
        uncertain_distribution["n_uncertain"] / uncertain_distribution["n_total"]
    )

    # Only count user labels from valid-bbox rows — when the bbox is
    # incorrect the operator left both selections blank.
    uncertain_user_rows = valid.loc[valid["model_subcategory"] == UNCERTAIN_LABEL]
    if not uncertain_user_rows.empty:
        uncertain_user_labels = (
            uncertain_user_rows.groupby(
                ["model_category", "user_category", "user_subcategory"], sort=True
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                ["model_category", "count", "user_subcategory"],
                ascending=[True, False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )
    else:
        uncertain_user_labels = pd.DataFrame(
            columns=["model_category", "user_category", "user_subcategory", "count"]
        )

    return {
        "summary": summary,
        "bbox_quality_per_category": bbox_quality_per_category,
        "per_category": per_category,
        "per_subcategory": per_subcategory,
        "uncertain_distribution": uncertain_distribution,
        "uncertain_user_labels": uncertain_user_labels,
    }


def write_accuracy_summary(
    validations_csv: Path,
    output_dir: Path,
) -> dict[str, Path] | None:
    """Read ``validations_csv`` and write the three accuracy CSVs.

    Returns the paths written, or ``None`` if there is nothing to summarise
    (no file, or empty file). Re-running rewrites cleanly.
    """
    if not validations_csv.exists():
        logger.info("No validations recorded — skipping accuracy summary.")
        return None
    df = pd.read_csv(validations_csv)
    if df.empty:
        logger.info("Validations file is empty — skipping accuracy summary.")
        return None

    results = compute_accuracy(df)
    paths = {
        "summary": output_dir / SUMMARY_FILENAME,
        "bbox_quality_per_category": output_dir / BBOX_QUALITY_PER_CATEGORY_FILENAME,
        "per_category": output_dir / PER_CATEGORY_FILENAME,
        "per_subcategory": output_dir / PER_SUBCATEGORY_FILENAME,
        "uncertain_distribution": output_dir / UNCERTAIN_DISTRIBUTION_FILENAME,
        "uncertain_user_labels": output_dir / UNCERTAIN_USER_LABELS_FILENAME,
    }
    for key, path in paths.items():
        results[key].to_csv(path, index=False)
    return paths


def _append_validation(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single validation to ``csv_path``, writing the header once."""
    write_header = not csv_path.exists()
    frame = pd.DataFrame([{col: row.get(col, "") for col in VALIDATION_COLUMNS}])
    frame.to_csv(csv_path, mode="a", header=write_header, index=False)


# --- GUI ------------------------------------------------------------------


class ValidationApp:
    """Tkinter UI driving one validation session.

    Constructed against a pre-built list of items so the iteration order is
    fully owned by the caller (e.g. shuffled by :func:`build_validation_items`).
    """

    def __init__(
        self,
        root: tk.Tk,
        config: ValidateConfig,
        items: list[dict[str, Any]],
        category_choices: list[str],
        subcategory_map: dict[str, list[str]],
        output_dir: Path,
    ) -> None:
        self.root = root
        self.config = config
        self.items = items
        self.category_choices = category_choices
        self.subcategory_map = subcategory_map
        self.output_dir = output_dir
        self.validations_csv = output_dir / VALIDATIONS_FILENAME
        self.index = 0
        self._finished = False
        self._tk_image: ImageTk.PhotoImage | None = None

        root.title("Fashion Attribute Validation")
        root.geometry(
            f"{config.display_max_dim + 120}x{config.display_max_dim + 320}"
        )
        root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._build_ui()
        self._show_current()

    # --- layout ----------------------------------------------------------

    def _build_ui(self) -> None:
        self.progress_var = tk.StringVar()
        ttk.Label(
            self.root,
            textvariable=self.progress_var,
            font=("Helvetica", 12, "bold"),
        ).pack(pady=(12, 4))

        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=6)

        bbox_frame = ttk.LabelFrame(self.root, text="Bounding box quality")
        bbox_frame.pack(pady=(8, 4), padx=10, fill="x")
        self.bbox_quality_var = tk.StringVar(value="")
        for display, value in _BBOX_QUALITY_DISPLAY:
            ttk.Radiobutton(
                bbox_frame,
                text=display,
                value=value,
                variable=self.bbox_quality_var,
                command=self._on_bbox_quality_change,
            ).pack(side="left", padx=10, pady=6)

        selection = ttk.Frame(self.root)
        selection.pack(pady=10)

        ttk.Label(selection, text="Category:").grid(
            row=0, column=0, padx=6, pady=4, sticky="e"
        )
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(
            selection,
            textvariable=self.category_var,
            values=self.category_choices,
            state="readonly",
            width=30,
        )
        self.category_combo.grid(row=0, column=1, padx=6, pady=4)
        self.category_combo.bind("<<ComboboxSelected>>", self._on_category_change)

        ttk.Label(selection, text="Subcategory:").grid(
            row=1, column=0, padx=6, pady=4, sticky="e"
        )
        self.subcategory_var = tk.StringVar()
        self.subcategory_combo = ttk.Combobox(
            selection,
            textvariable=self.subcategory_var,
            values=[],
            state="readonly",
            width=30,
        )
        self.subcategory_combo.grid(row=1, column=1, padx=6, pady=4)

        buttons = ttk.Frame(self.root)
        buttons.pack(pady=12)
        ttk.Button(buttons, text="Skip", command=self._on_skip).pack(
            side="left", padx=6
        )
        ttk.Button(
            buttons, text="Submit & Next", command=self._on_submit
        ).pack(side="left", padx=6)
        ttk.Button(buttons, text="Exit", command=self._on_exit).pack(
            side="left", padx=6
        )

        # Drop-downs are gated on the bbox-quality choice — start disabled
        # so the operator can't pre-pick labels for a bbox they've not yet
        # rated.
        self._set_label_combos_enabled(False)

    # --- event handlers --------------------------------------------------

    def _set_label_combos_enabled(self, enabled: bool) -> None:
        state = ["!disabled", "readonly"] if enabled else ["disabled"]
        self.category_combo.state(state)
        self.subcategory_combo.state(state)

    def _on_bbox_quality_change(self) -> None:
        if self.bbox_quality_var.get() == BBOX_QUALITY_INCORRECT:
            # Bbox is incorrect — there's no real garment region to label,
            # so wipe and disable both drop-downs.
            self.category_var.set("")
            self.subcategory_var.set("")
            self.subcategory_combo["values"] = []
            self._set_label_combos_enabled(False)
        else:
            self._set_label_combos_enabled(True)

    def _on_category_change(self, _event: object = None) -> None:
        parent = self.category_var.get()
        subs = self.subcategory_map.get(parent, [])
        self.subcategory_combo["values"] = subs
        if parent == NO_CLOTHES_LABEL:
            # "no garment in bbox" has a single matching sub-label — auto-
            # fill it so the operator can submit with one click.
            self.subcategory_var.set(NO_CLOTHES_LABEL)
            self.subcategory_combo.state(["disabled"])
        else:
            self.subcategory_var.set("")
            self.subcategory_combo.state(["!disabled", "readonly"])

    def _on_skip(self) -> None:
        self.index += 1
        self._show_current()

    def _on_submit(self) -> None:
        bbox_quality = self.bbox_quality_var.get()
        if bbox_quality not in BBOX_QUALITY_VALUES:
            messagebox.showwarning(
                "Missing selection",
                "Please rate the bounding box quality before submitting.",
            )
            return

        item = self.items[self.index]
        is_uncertain = item["model_subcategory"] == UNCERTAIN_LABEL

        if bbox_quality == BBOX_QUALITY_INCORRECT:
            # Incorrect bbox — no garment region to label. Persist a row
            # with blank user/correctness fields so the row still counts
            # toward bbox-quality stats but is excluded from cat/sub
            # accuracy by compute_accuracy().
            cat = ""
            sub = ""
            category_correct: int | str = ""
            subcategory_correct: int | str = ""
        else:
            cat = self.category_var.get().strip()
            sub = self.subcategory_var.get().strip()
            if not cat or not sub:
                messagebox.showwarning(
                    "Missing selection",
                    "Please pick both a category and a subcategory before submitting.",
                )
                return
            category_correct = int(item["model_category"] == cat)
            # Blank subcategory_correct when the model itself abstained —
            # the accuracy summary excludes these rows from the
            # subcategory denominator and routes them to the uncertain
            # analysis instead.
            subcategory_correct = (
                "" if is_uncertain else int(item["model_subcategory"] == sub)
            )

        row = {
            "image_id": item["image_id"],
            "garment_id": item["garment_id"],
            "model_category": item["model_category"],
            "model_subcategory": item["model_subcategory"],
            "bbox_quality": bbox_quality,
            "user_category": cat,
            "user_subcategory": sub,
            "category_correct": category_correct,
            "subcategory_correct": subcategory_correct,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _append_validation(self.validations_csv, row)
        self.index += 1
        self._show_current()

    def _on_exit(self) -> None:
        self._finish()

    # --- frame rendering -------------------------------------------------

    def _show_current(self) -> None:
        if self.index >= len(self.items):
            self._finish()
            return

        item = self.items[self.index]
        self.progress_var.set(
            f"Garment {self.index + 1} of {len(self.items)}"
            f"  —  {item['image_id']} (id {item['garment_id']})"
        )

        img_path = self.config.images_dir / item["image_id"]
        try:
            pil = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Cannot open %s: %s — skipping", img_path, exc)
            self.index += 1
            self.root.after(10, self._show_current)
            return

        draw = ImageDraw.Draw(pil)
        x0 = item["bbox_x"]
        y0 = item["bbox_y"]
        x1 = x0 + item["bbox_w"]
        y1 = y0 + item["bbox_h"]
        draw.rectangle([x0, y0, x1, y1], outline=_BBOX_OUTLINE, width=_BBOX_WIDTH)

        pil.thumbnail(
            (self.config.display_max_dim, self.config.display_max_dim),
            Image.Resampling.LANCZOS,
        )
        self._tk_image = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=self._tk_image)

        self.bbox_quality_var.set("")
        self.category_var.set("")
        self.subcategory_var.set("")
        self.subcategory_combo["values"] = []
        # Drop-downs stay disabled until the operator rates the bbox.
        self._set_label_combos_enabled(False)

    # --- finalisation ----------------------------------------------------

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        write_accuracy_summary(self.validations_csv, self.output_dir)
        self.root.destroy()


# --- orchestrator ---------------------------------------------------------


def run_validation(config: ValidateConfig) -> Path:
    """Launch the validation GUI for *config*. Returns the dated output dir."""
    df = pd.read_csv(config.attributes_csv)
    category_choices = build_category_choices(df)
    subcategory_map = build_subcategory_choices(df)
    items = build_validation_items(
        df,
        random_seed=config.random_seed,
        max_items=config.max_items,
    )
    if not items:
        raise RuntimeError(
            f"No validatable rows in {config.attributes_csv} "
            "(every row was missing a parent category)."
        )

    output_dir = make_validation_dir(config.output_root)
    logger.info(
        "Starting validation: %d items, writing to %s", len(items), output_dir
    )

    root = tk.Tk()
    ValidationApp(
        root, config, items, category_choices, subcategory_map, output_dir
    )
    root.mainloop()
    return output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_validate_config()
    run_validation(config)


if __name__ == "__main__":
    main()
