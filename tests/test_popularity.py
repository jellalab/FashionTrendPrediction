"""Tests for Pipeline 2 — consumer behavior classification.

All tests use synthetic in-memory DataFrames; no access to the real .xlsx
or any data/ artifact is required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.popularity import (
    RUN_NAMES,
    build_hashtag_features,
    build_popularity_features,
    clean_dataframe,
    compute_popularity,
    fit_hashtag_vocabulary,
    fit_minmax,
    parse_hashtags,
    popularity_formula_mean,
    run_ablation,
    run_pipeline,
    stratified_split,
)
from src.utils import (
    HashtagsConfig,
    PlotsConfig,
    PopularityConfig,
    PopularityFormulaConfig,
    RandomForestConfig,
    SplitConfig,
)


BEHAVIORAL_COLUMNS = (
    "Selfie",
    "BodySnap",
    "Marketing",
    "ProductOnly",
    "NonFashion",
    "Face",
    "Logo",
    "BrandLogo",
    "Smile",
    "Outdoor",
    "NumberOfPeople",
    "NumberOfFashionProduct",
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
)


def _make_config(
    tmp_path: Path,
    *,
    top_n: int = 5,
    test_size: float = 0.25,
    n_estimators: int = 25,
) -> PopularityConfig:
    return PopularityConfig(
        input_xlsx=tmp_path / "ignored.xlsx",
        output_dir=tmp_path,
        popularity_csv=tmp_path / "popularity_score.csv",
        ablation_results_csv=tmp_path / "ablation_results.csv",
        hashtag_features_csv=tmp_path / "hashtag_features.csv",
        combined_model_path=tmp_path / "model_combined.joblib",
        target_column="BrandCategory",
        excluded_columns=(
            "UserId",
            "BrandName",
            "Link",
            "ImgURL",
            "Caption",
            "CreationTime",
        ),
        popularity=PopularityFormulaConfig(
            likes_column="Likes",
            comments_column="comments",
            weight=0.5,
        ),
        group_a_engagement=(
            "Likes",
            "comments",
            "Followers",
            "MediaCount",
            "popularity_score_norm",
        ),
        group_c_behavioral=BEHAVIORAL_COLUMNS,
        hashtags=HashtagsConfig(column="Hashtags", top_n=top_n),
        split=SplitConfig(test_size=test_size, random_state=42, stratify=True),
        model=RandomForestConfig(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
        plots=PlotsConfig(top_k_importances=10, figure_dpi=80),
    )


def _make_synth_df(n_per_class: int = 25, random_state: int = 0) -> pd.DataFrame:
    """Build a synthetic 4-class dataset with the same columns as the real xlsx."""
    rng = np.random.default_rng(random_state)
    classes = ["Designer", "Small couture", "High street", "Mega couture"]
    rows: list[dict[str, object]] = []
    train_only_tags = ["alpha", "beta", "gamma", "delta", "epsilon"]
    rare_tags = ["zeta", "eta"]

    for cls_idx, cls in enumerate(classes):
        for i in range(n_per_class):
            likes = int(rng.integers(0, 500) + cls_idx * 100)
            comments = int(rng.integers(0, 20) + cls_idx * 5)
            tag_pick = rng.choice(train_only_tags + rare_tags, size=3, replace=False)
            row: dict[str, object] = {
                "UserId": f"user_{cls_idx}_{i}",
                "BrandName": f"brand_{cls_idx}",
                "BrandCategory": cls,
                "Followers": int(rng.integers(100, 100_000)),
                "MediaCount": int(rng.integers(10, 5000)),
                "Likes": likes,
                "Comments ": comments,  # trailing-space column name
                "Hashtags": ", ".join(tag_pick),
                "Caption": "",
                "ImgURL": "",
                "Link": "",
                "CreationTime": "2020-01-01",
            }
            for col in BEHAVIORAL_COLUMNS:
                row[col] = float(rng.random())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# clean_dataframe — trailing-space stripping and null-target drop
# ---------------------------------------------------------------------------


def test_clean_dataframe_strips_trailing_space_on_comments() -> None:
    df = pd.DataFrame(
        {"BrandCategory": ["Designer"], "Comments ": [5], "Likes": [10]}
    )
    cleaned = clean_dataframe(df, target_column="BrandCategory")
    assert "comments" in cleaned.columns
    assert "Comments " not in cleaned.columns
    assert cleaned["comments"].iloc[0] == 5


def test_clean_dataframe_drops_null_target_rows() -> None:
    df = pd.DataFrame(
        {"BrandCategory": ["Designer", None, "High street"], "Likes": [1, 2, 3]}
    )
    cleaned = clean_dataframe(df, target_column="BrandCategory")
    assert len(cleaned) == 2
    assert cleaned["BrandCategory"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Popularity formula + normalization
# ---------------------------------------------------------------------------


def test_popularity_formula_mean_specified_example() -> None:
    likes = pd.Series([200])
    comments = pd.Series([10])
    assert popularity_formula_mean(likes, comments, weight=0.5).iloc[0] == 105.0


def test_compute_popularity_uses_named_columns() -> None:
    df = pd.DataFrame({"Likes": [100, 50], "comments": [10, 5]})
    out = compute_popularity(df, "Likes", "comments", weight=0.5)
    assert out.tolist() == [55.0, 27.5]


def test_minmax_fit_on_train_maps_train_extremes_to_0_and_1() -> None:
    train_df = pd.DataFrame(
        {
            "Likes": [0, 100, 200, 300],
            "comments": [0, 0, 0, 0],
            "BrandCategory": ["A", "B", "C", "D"],
        }
    )
    test_df = pd.DataFrame(
        {
            "Likes": [50, 400],
            "comments": [0, 0],
            "BrandCategory": ["A", "B"],
        }
    )
    cfg = _make_config(Path("/tmp"))
    train_out, test_out, _, _ = build_popularity_features(train_df, test_df, cfg)
    assert train_out["popularity_score_norm"].min() == pytest.approx(0.0)
    assert train_out["popularity_score_norm"].max() == pytest.approx(1.0)
    # Test value of Likes=400 maps to (200 - 0) / (150 - 0) = 1.333 ... > 1.
    assert (test_out["popularity_score_norm"] > 1.0).any()
    # Not silently clipped — the > 1.0 row remains > 1.0.
    assert test_out["popularity_score_norm"].max() > 1.0


def test_apply_minmax_logs_when_test_values_exceed_train_range(
    caplog: pytest.LogCaptureFixture,
) -> None:
    train_df = pd.DataFrame(
        {"Likes": [0, 100], "comments": [0, 0], "BrandCategory": ["A", "B"]}
    )
    test_df = pd.DataFrame(
        {"Likes": [500], "comments": [0], "BrandCategory": ["A"]}
    )
    cfg = _make_config(Path("/tmp"))
    with caplog.at_level(logging.INFO, logger="src.popularity"):
        build_popularity_features(train_df, test_df, cfg)
    assert any("outside [0, 1]" in rec.message for rec in caplog.records)


def test_fit_minmax_returns_train_extremes() -> None:
    vmin, vmax = fit_minmax(pd.Series([3, 5, 7, 9]))
    assert vmin == 3.0
    assert vmax == 9.0


# ---------------------------------------------------------------------------
# Hashtag parsing + multi-hot encoding
# ---------------------------------------------------------------------------


def test_parse_hashtags_strips_and_lowercases() -> None:
    assert parse_hashtags("Beautiful, Summer , FASHION") == [
        "beautiful",
        "summer",
        "fashion",
    ]


def test_parse_hashtags_handles_null_as_empty_list() -> None:
    assert parse_hashtags(None) == []
    assert parse_hashtags(np.nan) == []
    assert parse_hashtags("") == []


def test_fit_hashtag_vocabulary_takes_top_n_by_post_count() -> None:
    train_tags = pd.Series(
        [
            ["alpha", "beta"],
            ["alpha", "gamma"],
            ["alpha", "delta"],
            ["beta", "gamma"],
        ]
    )
    vocab = fit_hashtag_vocabulary(train_tags, top_n=2)
    assert vocab[0] == "alpha"
    assert "beta" in vocab or "gamma" in vocab
    assert len(vocab) == 2


def test_build_hashtag_features_multihot_shape_and_values() -> None:
    train_df = pd.DataFrame(
        {
            "Hashtags": [
                "alpha, beta",
                "alpha, gamma",
                "alpha, delta",
                "beta, gamma",
            ],
            "BrandCategory": ["A", "B", "C", "D"],
        }
    )
    test_df = pd.DataFrame(
        {"Hashtags": ["alpha, omega"], "BrandCategory": ["A"]}
    )
    train_h, test_h, vocab = build_hashtag_features(
        train_df, test_df, hashtags_column="Hashtags", top_n=3
    )
    assert len(vocab) == 3
    for col in [f"hashtag_{t}" for t in vocab]:
        assert col in train_h.columns
        assert col in test_h.columns
    assert "hashtag_count" in train_h.columns
    assert train_h["hashtag_count"].tolist() == [2, 2, 2, 2]
    assert int(train_h[f"hashtag_alpha"].sum()) == 3
    assert train_h.shape == (4, len(vocab) + 1)
    assert test_h.shape == (1, len(vocab) + 1)
    assert int(test_h[f"hashtag_alpha"].iloc[0]) == 1


def test_build_hashtag_features_null_hashtags_become_empty_rows() -> None:
    train_df = pd.DataFrame(
        {"Hashtags": ["alpha, beta", None, "alpha"], "BrandCategory": list("ABC")}
    )
    test_df = pd.DataFrame(
        {"Hashtags": [np.nan], "BrandCategory": ["A"]}
    )
    train_h, test_h, _ = build_hashtag_features(
        train_df, test_df, hashtags_column="Hashtags", top_n=2
    )
    assert train_h.iloc[1].drop("hashtag_count").sum() == 0
    assert train_h["hashtag_count"].iloc[1] == 0
    assert test_h["hashtag_count"].iloc[0] == 0


def test_hashtag_vocabulary_ignores_test_only_tags() -> None:
    train_df = pd.DataFrame(
        {
            "Hashtags": ["alpha, beta", "alpha, beta", "alpha"],
            "BrandCategory": list("ABC"),
        }
    )
    test_df = pd.DataFrame(
        {"Hashtags": ["alpha, ONLY_IN_TEST"], "BrandCategory": ["A"]}
    )
    _, test_h, vocab = build_hashtag_features(
        train_df, test_df, hashtags_column="Hashtags", top_n=10
    )
    assert "only_in_test" not in vocab
    assert "hashtag_only_in_test" not in test_h.columns


# ---------------------------------------------------------------------------
# Stratified split preserves class proportions
# ---------------------------------------------------------------------------


def test_stratified_split_preserves_proportions() -> None:
    df = _make_synth_df(n_per_class=50)
    train_df, test_df = stratified_split(
        df, target_column="BrandCategory", test_size=0.2, random_state=42, stratify=True
    )
    full_props = df["BrandCategory"].value_counts(normalize=True).sort_index()
    test_props = test_df["BrandCategory"].value_counts(normalize=True).sort_index()
    np.testing.assert_allclose(
        full_props.values, test_props.values, atol=0.02
    )


# ---------------------------------------------------------------------------
# BrandName never leaks into a feature matrix
# ---------------------------------------------------------------------------


def test_brandname_absent_from_every_feature_matrix(tmp_path: Path) -> None:
    df = _make_synth_df(n_per_class=20)
    df_clean = clean_dataframe(df, target_column="BrandCategory")
    cfg = _make_config(tmp_path, top_n=5, test_size=0.25, n_estimators=10)
    train_df, test_df = stratified_split(
        df_clean,
        target_column=cfg.target_column,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=True,
    )
    train_df, test_df, _, _ = build_popularity_features(train_df, test_df, cfg)
    train_h, test_h, _ = build_hashtag_features(
        train_df, test_df, hashtags_column="Hashtags", top_n=cfg.hashtags.top_n
    )
    runs = run_ablation(train_df, test_df, train_h, test_h, cfg)
    for run in runs:
        assert "BrandName" not in run.feature_names, (
            f"BrandName leaked into run {run.name}"
        )
        for excluded in cfg.excluded_columns:
            assert excluded not in run.feature_names, (
                f"{excluded} leaked into run {run.name}"
            )


# ---------------------------------------------------------------------------
# Ablation completeness + determinism
# ---------------------------------------------------------------------------


def test_run_ablation_produces_all_five_runs(tmp_path: Path) -> None:
    df = _make_synth_df(n_per_class=25)
    df_clean = clean_dataframe(df, target_column="BrandCategory")
    cfg = _make_config(tmp_path, top_n=5, test_size=0.25, n_estimators=10)
    train_df, test_df = stratified_split(
        df_clean, "BrandCategory", 0.25, 42, True
    )
    train_df, test_df, _, _ = build_popularity_features(train_df, test_df, cfg)
    train_h, test_h, _ = build_hashtag_features(
        train_df, test_df, hashtags_column="Hashtags", top_n=5
    )
    runs = run_ablation(train_df, test_df, train_h, test_h, cfg)
    assert tuple(r.name for r in runs) == RUN_NAMES
    assert len(runs) == 5


def test_combined_engagement_visual_excludes_hashtag_columns(
    tmp_path: Path,
) -> None:
    df = _make_synth_df(n_per_class=25)
    df_clean = clean_dataframe(df, target_column="BrandCategory")
    cfg = _make_config(tmp_path, top_n=5, test_size=0.25, n_estimators=10)
    train_df, test_df = stratified_split(
        df_clean, "BrandCategory", 0.25, 42, True
    )
    train_df, test_df, _, _ = build_popularity_features(train_df, test_df, cfg)
    train_h, test_h, _ = build_hashtag_features(
        train_df, test_df, hashtags_column="Hashtags", top_n=5
    )
    runs = run_ablation(train_df, test_df, train_h, test_h, cfg)
    h2 = next(r for r in runs if r.name == "combined_engagement_visual")
    assert not any(c.startswith("hashtag_") for c in h2.feature_names)
    assert "hashtag_count" not in h2.feature_names
    expected = len(cfg.group_a_engagement) + len(cfg.group_c_behavioral)
    assert len(h2.feature_names) == expected

    all_run = next(r for r in runs if r.name == "combined_all_with_hashtags")
    assert any(c.startswith("hashtag_") for c in all_run.feature_names)
    assert len(all_run.feature_names) > len(h2.feature_names)


def test_pipeline_is_deterministic_across_runs(tmp_path: Path) -> None:
    df = _make_synth_df(n_per_class=40)
    # Persist twice as an .xlsx and run the full pipeline; macro_f1 must match.
    xlsx_path = tmp_path / "synth.xlsx"
    df.to_excel(xlsx_path, index=False)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    cfg1 = PopularityConfig(
        input_xlsx=xlsx_path,
        output_dir=out1,
        popularity_csv=out1 / "popularity_score.csv",
        ablation_results_csv=out1 / "ablation_results.csv",
        hashtag_features_csv=out1 / "hashtag_features.csv",
        combined_model_path=out1 / "model_combined.joblib",
        target_column="BrandCategory",
        excluded_columns=(
            "UserId",
            "BrandName",
            "Link",
            "ImgURL",
            "Caption",
            "CreationTime",
        ),
        popularity=PopularityFormulaConfig(
            likes_column="Likes", comments_column="comments", weight=0.5
        ),
        group_a_engagement=(
            "Likes",
            "comments",
            "Followers",
            "MediaCount",
            "popularity_score_norm",
        ),
        group_c_behavioral=BEHAVIORAL_COLUMNS,
        hashtags=HashtagsConfig(column="Hashtags", top_n=5),
        split=SplitConfig(test_size=0.25, random_state=42, stratify=True),
        model=RandomForestConfig(
            n_estimators=15, random_state=42, class_weight="balanced", n_jobs=1
        ),
        plots=PlotsConfig(top_k_importances=10, figure_dpi=80),
    )
    cfg2 = PopularityConfig(
        **{
            **cfg1.__dict__,
            "output_dir": out2,
            "popularity_csv": out2 / "popularity_score.csv",
            "ablation_results_csv": out2 / "ablation_results.csv",
            "hashtag_features_csv": out2 / "hashtag_features.csv",
            "combined_model_path": out2 / "model_combined.joblib",
        }
    )

    res1 = run_pipeline(cfg1)
    res2 = run_pipeline(cfg2)
    np.testing.assert_array_equal(
        res1[["macro_f1", "accuracy", "weighted_f1"]].values,
        res2[["macro_f1", "accuracy", "weighted_f1"]].values,
    )


def test_ablation_results_csv_has_one_row_per_run(tmp_path: Path) -> None:
    df = _make_synth_df(n_per_class=25)
    xlsx_path = tmp_path / "synth.xlsx"
    df.to_excel(xlsx_path, index=False)

    cfg = PopularityConfig(
        input_xlsx=xlsx_path,
        output_dir=tmp_path,
        popularity_csv=tmp_path / "popularity_score.csv",
        ablation_results_csv=tmp_path / "ablation_results.csv",
        hashtag_features_csv=tmp_path / "hashtag_features.csv",
        combined_model_path=tmp_path / "model_combined.joblib",
        target_column="BrandCategory",
        excluded_columns=(
            "UserId",
            "BrandName",
            "Link",
            "ImgURL",
            "Caption",
            "CreationTime",
        ),
        popularity=PopularityFormulaConfig(
            likes_column="Likes", comments_column="comments", weight=0.5
        ),
        group_a_engagement=(
            "Likes",
            "comments",
            "Followers",
            "MediaCount",
            "popularity_score_norm",
        ),
        group_c_behavioral=BEHAVIORAL_COLUMNS,
        hashtags=HashtagsConfig(column="Hashtags", top_n=5),
        split=SplitConfig(test_size=0.25, random_state=42, stratify=True),
        model=RandomForestConfig(
            n_estimators=10, random_state=42, class_weight="balanced", n_jobs=1
        ),
        plots=PlotsConfig(top_k_importances=10, figure_dpi=80),
    )
    results = run_pipeline(cfg)
    assert set(results["run_name"]) == set(RUN_NAMES)
    assert len(results) == 5
    assert (tmp_path / "popularity_score.csv").exists()
    for run_name in RUN_NAMES:
        assert (tmp_path / f"confusion_matrix_{run_name}.png").exists()
        assert (tmp_path / f"feature_importance_{run_name}.png").exists()
        assert (tmp_path / f"classification_report_{run_name}.txt").exists()
    assert (tmp_path / "model_combined.joblib").exists()


def test_hashtag_features_csv_written_with_split_and_shape(
    tmp_path: Path,
) -> None:
    df = _make_synth_df(n_per_class=20)
    xlsx_path = tmp_path / "synth.xlsx"
    df.to_excel(xlsx_path, index=False)

    cfg = _make_config(tmp_path, top_n=5, test_size=0.25, n_estimators=10)
    cfg = PopularityConfig(
        **{**cfg.__dict__, "input_xlsx": xlsx_path}
    )
    run_pipeline(cfg)

    out_path = cfg.hashtag_features_csv
    assert out_path.exists()
    out = pd.read_csv(out_path)

    # Row count: train + test = full cleaned dataset (synth has no nulls).
    assert len(out) == len(df)

    # Column set: hashtag_{tag} columns + hashtag_count + split, nothing else.
    expected_hashtag_cols = {c for c in out.columns if c.startswith("hashtag_")}
    assert "hashtag_count" in out.columns
    assert "split" in out.columns
    assert set(out.columns) == expected_hashtag_cols | {"split"}
    assert len(expected_hashtag_cols) - 1 == cfg.hashtags.top_n  # minus hashtag_count

    # split column contains exactly {"train", "test"}.
    assert set(out["split"].unique()) == {"train", "test"}

    # Target / brand / identifier leakage check.
    for forbidden in (cfg.target_column, *cfg.excluded_columns):
        assert forbidden not in out.columns
