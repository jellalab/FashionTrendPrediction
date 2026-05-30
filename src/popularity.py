"""Pipeline 2 — consumer behavior classification with an ablation study.

Classifies each fashion Instagram post into one of four ``BrandCategory``
values (Designer, Small couture, High street, Mega couture) from post
metadata. The pipeline:

1. Loads the Kim et al. Instagram fashion .xlsx (one sheet).
2. Computes a swappable popularity score and adds a train-only min-max
   normalized copy as a Group A feature.
3. Builds three feature groups:
   - **Group A — engagement / reach**: Likes, comments, Followers,
     MediaCount, popularity_score_norm.
   - **Group B — hashtags**: multi-hot encoding of the top-N hashtags
     (vocab fit on the TRAIN partition only) plus ``hashtag_count``.
   - **Group C — behavioral / visual**: pre-computed 0/1 image-content
     scores and emotion probabilities, used as-is.
4. Trains a ``RandomForestClassifier`` four times — Group A only, Group B
   only, Group C only, and all groups combined — on the same stratified
   train/test split, and writes:
   - ``popularity_score.csv`` — one row per post with the score columns.
   - ``ablation_results.csv`` — one row per run.
   - ``confusion_matrix_{run}.png`` / ``feature_importance_{run}.png``.
   - ``classification_report_{run}.txt``.
   - ``model_combined.joblib`` — the trained combined-features model.

BrandName is excluded from every feature matrix and the exclusion is
asserted before training. The trailing-space ``Comments `` column is
stripped to ``comments`` at load time.

Run as a module from the project root::

    uv run python -m src.popularity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.utils import (  # noqa: E402
    PopularityConfig,
    load_popularity_config,
)

logger = logging.getLogger(__name__)


__all__ = [
    "AblationRun",
    "RUN_NAMES",
    "build_hashtag_features",
    "build_popularity_features",
    "clean_dataframe",
    "compute_popularity",
    "fit_hashtag_vocabulary",
    "fit_minmax",
    "parse_hashtags",
    "popularity_formula_mean",
    "run_ablation",
    "run_pipeline",
    "stratified_split",
    "train_eval_run",
]


RUN_NAMES: tuple[str, str, str, str, str] = (
    "group_a_engagement",
    "group_b_hashtags",
    "group_c_behavioral",
    "combined_engagement_visual",
    "combined_all_with_hashtags",
)

HASHTAG_FEATURE_PREFIX = "hashtag_"


@dataclass(frozen=True)
class AblationRun:
    """A single ablation run's metrics and per-feature importances."""

    name: str
    feature_names: tuple[str, ...]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion: np.ndarray
    class_labels: tuple[str, ...]
    feature_importances: pd.Series
    classification_report_text: str
    model: RandomForestClassifier


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------


COMMENTS_RENAME: dict[str, str] = {"Comments": "comments"}


def clean_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Strip whitespace from column names and drop rows with a null target.

    The Kim et al. dataset ships with a trailing-space ``Comments `` column;
    stripping turns it into ``Comments``, which is then explicitly renamed
    to ``comments`` to match the canonical config key. Rows missing the
    target are dropped with a log line so the count appears in the run
    summary.
    """
    out = df.copy()
    whitespace_map = {c: c.strip() for c in out.columns if c != c.strip()}
    if whitespace_map:
        logger.info(
            "Stripped whitespace from %d column names", len(whitespace_map)
        )
        out = out.rename(columns=whitespace_map)

    final_map = {k: v for k, v in COMMENTS_RENAME.items() if k in out.columns}
    if final_map:
        out = out.rename(columns=final_map)

    if target_column not in out.columns:
        raise KeyError(
            f"target column {target_column!r} not present after cleaning; "
            f"have: {list(out.columns)}"
        )

    null_target = int(out[target_column].isna().sum())
    if null_target:
        logger.info(
            "Dropping %d rows with null target %r", null_target, target_column
        )
        out = out.loc[out[target_column].notna()].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Popularity score
# ---------------------------------------------------------------------------


def popularity_formula_mean(
    likes: pd.Series, comments: pd.Series, weight: float = 0.5
) -> pd.Series:
    """Default popularity formula: ``(Likes + comments) * weight`` (mean if 0.5).

    Swappable — pass a different callable to :func:`compute_popularity`.
    """
    return (likes.astype(float) + comments.astype(float)) * float(weight)


def compute_popularity(
    df: pd.DataFrame,
    likes_col: str,
    comments_col: str,
    weight: float,
    formula: Callable[[pd.Series, pd.Series, float], pd.Series] = popularity_formula_mean,
) -> pd.Series:
    """Apply ``formula`` to the engagement columns and return the raw score."""
    if likes_col not in df.columns:
        raise KeyError(f"missing likes column: {likes_col!r}")
    if comments_col not in df.columns:
        raise KeyError(f"missing comments column: {comments_col!r}")
    return formula(df[likes_col], df[comments_col], weight)


def fit_minmax(values: pd.Series) -> tuple[float, float]:
    """Return ``(min, max)`` from a training series for later normalization."""
    vmin = float(values.min())
    vmax = float(values.max())
    return vmin, vmax


def apply_minmax(
    values: pd.Series, vmin: float, vmax: float, *, source: str = "values"
) -> pd.Series:
    """Apply ``(x - vmin) / (vmax - vmin)`` without clipping.

    Test-set values can legitimately exceed [0, 1]; we log when that happens
    so the divergence is visible but never silently clipped.
    """
    if vmax == vmin:
        logger.warning(
            "%s has zero range on train (min == max == %s); returning zeros",
            source,
            vmin,
        )
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    out = (values.astype(float) - vmin) / (vmax - vmin)
    out_of_range = int(((out < 0.0) | (out > 1.0)).sum())
    if out_of_range:
        logger.info(
            "%s has %d values outside [0, 1] after train-fit normalization "
            "(not clipped)",
            source,
            out_of_range,
        )
    return out


# ---------------------------------------------------------------------------
# Feature group A — engagement / reach
# ---------------------------------------------------------------------------


def _impute_zero_with_log(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"missing engagement column: {col!r}")
        n_null = int(out[col].isna().sum())
        if n_null:
            logger.info(
                "Imputing %d null %s values with 0 (reason: missing engagement)",
                n_null,
                col,
            )
            out[col] = out[col].fillna(0)
    return out


def build_popularity_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PopularityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Compute raw + normalized popularity scores for train and test.

    Normalization (min-max) is **fit on train only** and applied to both. The
    returned frames are copies of the input frames with two extra columns,
    ``popularity_score`` and ``popularity_score_norm``; the two ``pd.Series``
    are the raw scores indexed identically to the inputs.
    """
    pop_cfg = config.popularity
    train_raw = compute_popularity(
        train_df, pop_cfg.likes_column, pop_cfg.comments_column, pop_cfg.weight
    )
    test_raw = compute_popularity(
        test_df, pop_cfg.likes_column, pop_cfg.comments_column, pop_cfg.weight
    )

    vmin, vmax = fit_minmax(train_raw)
    train_norm = apply_minmax(train_raw, vmin, vmax, source="popularity_score(train)")
    test_norm = apply_minmax(test_raw, vmin, vmax, source="popularity_score(test)")

    train_out = train_df.copy()
    train_out["popularity_score"] = train_raw.values
    train_out["popularity_score_norm"] = train_norm.values

    test_out = test_df.copy()
    test_out["popularity_score"] = test_raw.values
    test_out["popularity_score_norm"] = test_norm.values

    return train_out, test_out, train_raw, test_raw


# ---------------------------------------------------------------------------
# Feature group B — hashtags
# ---------------------------------------------------------------------------


def parse_hashtags(value: object) -> list[str]:
    """Parse a single ``Hashtags`` cell into a cleaned, lowercased tag list.

    ``"beautiful, Summer , fashion"`` → ``["beautiful", "summer", "fashion"]``.
    Null / empty values produce ``[]`` — never a crash, never a dropped row.
    """
    if value is None:
        return []
    try:
        if isinstance(value, float) and np.isnan(value):
            return []
    except TypeError:
        pass
    if not isinstance(value, str):
        return []
    parts = [p.strip().lower() for p in value.split(",")]
    return [p for p in parts if p]


def fit_hashtag_vocabulary(
    train_tags: pd.Series, top_n: int
) -> tuple[str, ...]:
    """Return the top-N most frequent hashtags across the TRAIN partition.

    Frequency is counted by post (a tag appearing twice in one post still
    counts once), so the vocabulary reflects coverage, not raw mentions.
    Ties are broken alphabetically for determinism.
    """
    counts: dict[str, int] = {}
    for tags in train_tags:
        for t in set(tags):
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return tuple()
    sorted_tags = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(t for t, _ in sorted_tags[:top_n])


def build_hashtag_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hashtags_column: str,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Parse hashtags and multi-hot encode the top-N seen on train.

    Adds one ``hashtag_{tag}`` column per top-N tag (0/1) plus a
    ``hashtag_count`` column counting **all** parsed tags (not just top-N).
    Hashtags appearing only in test are ignored — they're not in the train
    vocabulary, so they cannot influence the model.
    """
    train_tags = train_df[hashtags_column].map(parse_hashtags)
    test_tags = test_df[hashtags_column].map(parse_hashtags)

    n_train_null = int((train_tags.map(len) == 0).sum())
    n_test_null = int((test_tags.map(len) == 0).sum())
    if n_train_null or n_test_null:
        logger.info(
            "Hashtags treated as empty list: train=%d, test=%d "
            "(reason: null or empty cell)",
            n_train_null,
            n_test_null,
        )

    vocab = fit_hashtag_vocabulary(train_tags, top_n)
    vocab_index = {t: i for i, t in enumerate(vocab)}

    def encode(tags_series: pd.Series) -> pd.DataFrame:
        feature_names = [f"hashtag_{t}" for t in vocab]
        matrix = np.zeros((len(tags_series), len(vocab)), dtype=np.int8)
        for row_i, tags in enumerate(tags_series):
            for t in tags:
                col = vocab_index.get(t)
                if col is not None:
                    matrix[row_i, col] = 1
        out = pd.DataFrame(matrix, columns=feature_names, index=tags_series.index)
        out["hashtag_count"] = tags_series.map(len).astype(int).values
        return out

    return encode(train_tags), encode(test_tags), vocab


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def stratified_split(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int, stratify: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that preserves target proportions."""
    stratify_arg = df[target_column] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ablation training
# ---------------------------------------------------------------------------


def _assert_no_excluded(
    X: pd.DataFrame, excluded: tuple[str, ...], run_name: str
) -> None:
    """Hard-fail if any excluded column (e.g. BrandName) leaked into features."""
    bad = [c for c in X.columns if c in excluded]
    if bad:
        raise AssertionError(
            f"run {run_name!r}: forbidden column(s) in feature matrix: {bad}"
        )


def _assert_no_hashtag_columns(X: pd.DataFrame, run_name: str) -> None:
    """Hard-fail if any hashtag-derived column leaked into a non-hashtag run.

    Used to guarantee that the H2 ``combined_engagement_visual`` run sees
    engagement + behavioral features only. ``hashtag_count`` is excluded too
    because it is hashtag-derived and would re-introduce the same brand-tag
    leakage signal in summary form.
    """
    bad = [
        c for c in X.columns
        if c.startswith(HASHTAG_FEATURE_PREFIX) or c == "hashtag_count"
    ]
    if bad:
        raise AssertionError(
            f"run {run_name!r}: hashtag-derived column(s) leaked into "
            f"feature matrix: {bad[:5]}{'...' if len(bad) > 5 else ''}"
        )


def train_eval_run(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    excluded_columns: tuple[str, ...],
    model_cfg,
) -> AblationRun:
    """Fit a Random Forest on ``X_train`` and score it on ``X_test``."""
    _assert_no_excluded(X_train, excluded_columns, name)
    _assert_no_excluded(X_test, excluded_columns, name)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            f"run {name!r}: train and test feature columns differ"
        )

    model = RandomForestClassifier(
        n_estimators=model_cfg.n_estimators,
        random_state=model_cfg.random_state,
        class_weight=model_cfg.class_weight,
        n_jobs=model_cfg.n_jobs,
    )
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    labels = tuple(sorted(set(y_train.unique()).union(set(y_test.unique()))))
    per_class_f1_arr = f1_score(
        y_test, y_pred, labels=list(labels), average=None, zero_division=0
    )
    per_class_f1 = {
        label: float(score) for label, score in zip(labels, per_class_f1_arr)
    }

    test_counts = y_test.value_counts()
    sparse_classes = [c for c in labels if int(test_counts.get(c, 0)) < 10]
    if sparse_classes:
        logger.warning(
            "run %r: classes with <10 test samples: %s — F1 estimates noisy",
            name,
            sparse_classes,
        )

    return AblationRun(
        name=name,
        feature_names=tuple(X_train.columns),
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        per_class_f1=per_class_f1,
        confusion=confusion_matrix(y_test, y_pred, labels=list(labels)),
        class_labels=labels,
        feature_importances=pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False),
        classification_report_text=classification_report(
            y_test, y_pred, labels=list(labels), zero_division=0
        ),
        model=model,
    )


def run_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_hash: pd.DataFrame,
    test_hash: pd.DataFrame,
    config: PopularityConfig,
) -> list[AblationRun]:
    """Train + evaluate the five-run ablation on a shared stratified split.

    Single-group baselines:
    - **group_a_engagement** — engagement / reach (Group A only).
    - **group_b_hashtags** — multi-hot hashtag matrix (Group B only).
    - **group_c_behavioral** — image-content + emotion scores (Group C only).

    Combined runs:
    - **combined_engagement_visual** — Group A + Group C only. This is the
      H2 test: hashtags are explicitly excluded because the dataset was
      collected via brand-name hashtags, so the hashtag features leak
      brand identity. A runtime assertion guarantees no hashtag-derived
      column reaches this feature matrix.
    - **combined_all_with_hashtags** — Group A + Group B + Group C
      concatenated. Kept alongside the H2 model so the leakage gap is
      visible in the ablation table.
    """
    y_train = train_df[config.target_column]
    y_test = test_df[config.target_column]
    excluded = config.excluded_columns

    group_a_cols = list(config.group_a_engagement)
    X_train_a = train_df[group_a_cols].astype(float)
    X_test_a = test_df[group_a_cols].astype(float)

    X_train_b = train_hash.astype(float)
    X_test_b = test_hash.astype(float)

    group_c_cols = list(config.group_c_behavioral)
    X_train_c = train_df[group_c_cols].astype(float)
    X_test_c = test_df[group_c_cols].astype(float)

    X_train_ac = pd.concat([X_train_a, X_train_c], axis=1)
    X_test_ac = pd.concat([X_test_a, X_test_c], axis=1)
    _assert_no_hashtag_columns(X_train_ac, "combined_engagement_visual")
    _assert_no_hashtag_columns(X_test_ac, "combined_engagement_visual")

    X_train_all = pd.concat([X_train_a, X_train_b, X_train_c], axis=1)
    X_test_all = pd.concat([X_test_a, X_test_b, X_test_c], axis=1)

    runs = [
        train_eval_run("group_a_engagement", X_train_a, y_train, X_test_a, y_test, excluded, config.model),
        train_eval_run("group_b_hashtags", X_train_b, y_train, X_test_b, y_test, excluded, config.model),
        train_eval_run("group_c_behavioral", X_train_c, y_train, X_test_c, y_test, excluded, config.model),
        train_eval_run("combined_engagement_visual", X_train_ac, y_train, X_test_ac, y_test, excluded, config.model),
        train_eval_run("combined_all_with_hashtags", X_train_all, y_train, X_test_all, y_test, excluded, config.model),
    ]
    return runs


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_popularity_csv(
    path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_hash: pd.DataFrame,
    test_hash: pd.DataFrame,
    config: PopularityConfig,
) -> None:
    """Write the per-post popularity + engineered feature dump.

    Includes the raw + normalized popularity score, the target column, all
    Group A and Group C feature columns, and ``hashtag_count``. A ``split``
    column marks each row as ``train`` or ``test``.
    """
    feature_cols = (
        ["popularity_score", "popularity_score_norm"]
        + list(config.group_a_engagement)
        + list(config.group_c_behavioral)
    )
    feature_cols = list(dict.fromkeys(feature_cols))

    train_out = train_df[[config.target_column, *feature_cols]].copy()
    train_out["hashtag_count"] = train_hash["hashtag_count"].values
    train_out["split"] = "train"

    test_out = test_df[[config.target_column, *feature_cols]].copy()
    test_out["hashtag_count"] = test_hash["hashtag_count"].values
    test_out["split"] = "test"

    combined = pd.concat([train_out, test_out], axis=0, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def _write_hashtag_features_csv(
    path: Path,
    train_hash: pd.DataFrame,
    test_hash: pd.DataFrame,
) -> None:
    """Persist the train+test hashtag multi-hot matrix for documentation.

    The output is the exact feature matrix consumed by the hashtag-based
    ablation runs — one ``hashtag_{tag}`` column per top-N hashtag plus
    ``hashtag_count`` — augmented with a ``split`` column marking each row
    as ``train`` or ``test``. Target / brand / identifier columns are
    intentionally absent so the file is safe to inspect alongside the
    leakage discussion.
    """
    train_out = train_hash.copy()
    train_out["split"] = "train"
    test_out = test_hash.copy()
    test_out["split"] = "test"
    combined = pd.concat([train_out, test_out], axis=0, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def _write_ablation_results(path: Path, runs: list[AblationRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_classes = sorted({c for r in runs for c in r.class_labels})
    for r in runs:
        row: dict[str, object] = {
            "run_name": r.name,
            "features_used_count": len(r.feature_names),
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "weighted_f1": r.weighted_f1,
        }
        for cls in all_classes:
            row[f"f1_{cls}"] = r.per_class_f1.get(cls, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _write_classification_report(path: Path, run: AblationRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.classification_report_text, encoding="utf-8")


def _plot_confusion_matrix(path: Path, run: AblationRun, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    sns.heatmap(
        run.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=run.class_labels,
        yticklabels=run.class_labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {run.name}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_feature_importance(path: Path, run: AblationRun, top_k: int, dpi: int) -> None:
    top = run.feature_importances.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(top))), dpi=dpi)
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top)} features — {run.name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(
    df_clean: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    runs: list[AblationRun],
    results: pd.DataFrame,
    target_column: str,
) -> None:
    print()
    print("=== Pipeline 2: consumer behavior classification ===")
    print(f"Total rows after cleaning: {len(df_clean)}")
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")
    print()
    print("Target class distribution (full dataset):")
    counts = df_clean[target_column].value_counts()
    total = int(counts.sum())
    for cls, n in counts.items():
        print(f"  {cls:<16} {n:>6}  ({100.0 * n / total:5.2f}%)")
    print()
    print("Test split class distribution:")
    test_counts = test_df[target_column].value_counts()
    for cls, n in test_counts.items():
        print(f"  {cls:<16} {n:>6}  ({100.0 * n / len(test_df):5.2f}%)")
    print()
    print("Ablation comparison (primary metric: macro-F1):")
    cols = ["run_name", "features_used_count", "accuracy", "macro_f1", "weighted_f1"]
    print(results[cols].to_string(index=False))
    best = max(runs, key=lambda r: r.macro_f1)
    print()
    print(f"Best run by macro-F1: {best.name} (macro_f1={best.macro_f1:.4f})")
    print()


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: PopularityConfig) -> pd.DataFrame:
    """Execute the full Pipeline 2 end-to-end and return the ablation table."""
    logger.info("Loading %s", config.input_xlsx)
    raw_df = pd.read_excel(config.input_xlsx)
    logger.info("Loaded %d rows, %d columns", len(raw_df), raw_df.shape[1])

    df_clean = clean_dataframe(raw_df, config.target_column)

    engagement_cols = (
        config.popularity.likes_column,
        config.popularity.comments_column,
        "Followers",
        "MediaCount",
    )
    df_clean = _impute_zero_with_log(df_clean, engagement_cols)

    if config.hashtags.column not in df_clean.columns:
        raise KeyError(
            f"missing hashtags column: {config.hashtags.column!r}"
        )

    train_df, test_df = stratified_split(
        df_clean,
        target_column=config.target_column,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=config.split.stratify,
    )

    train_df, test_df, _train_raw, _test_raw = build_popularity_features(
        train_df, test_df, config
    )

    train_hash, test_hash, vocab = build_hashtag_features(
        train_df,
        test_df,
        hashtags_column=config.hashtags.column,
        top_n=config.hashtags.top_n,
    )
    logger.info("Hashtag vocabulary size (train, top-N): %d", len(vocab))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_popularity_csv(
        config.popularity_csv, train_df, test_df, train_hash, test_hash, config
    )
    _write_hashtag_features_csv(
        config.hashtag_features_csv, train_hash, test_hash
    )

    runs = run_ablation(train_df, test_df, train_hash, test_hash, config)
    results = _write_ablation_results(config.ablation_results_csv, runs)

    for run in runs:
        _write_classification_report(
            config.output_dir / f"classification_report_{run.name}.txt", run
        )
        _plot_confusion_matrix(
            config.output_dir / f"confusion_matrix_{run.name}.png",
            run,
            config.plots.figure_dpi,
        )
        _plot_feature_importance(
            config.output_dir / f"feature_importance_{run.name}.png",
            run,
            config.plots.top_k_importances,
            config.plots.figure_dpi,
        )

    combined_run = next(
        r for r in runs if r.name == "combined_all_with_hashtags"
    )
    config.combined_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": combined_run.model,
            "feature_names": combined_run.feature_names,
            "class_labels": combined_run.class_labels,
        },
        config.combined_model_path,
    )

    _print_summary(df_clean, train_df, test_df, runs, results, config.target_column)
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_popularity_config()
    run_pipeline(config)


if __name__ == "__main__":
    main()
