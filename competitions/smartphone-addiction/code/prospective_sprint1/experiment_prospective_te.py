"""Prospective S6E8 ablation: raw vs frequency/missing vs exact-value TE.

The 20% holdout (fold=-2) is deliberately not scored. Target encodings for the
external training rows are cross-fitted with inner folds; external validation
encodings are learned only from the corresponding external training partition.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
TARGET = "addicted_label"
ID_COL = "id"
SEED = 20260813
INNER_FOLDS = 5
SMOOTHING = 20.0

NUM_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
SOURCE_COLS = NUM_COLS + CAT_COLS
VARIANTS = ("raw", "frequency_missing", "exact_te")

XGB_PARAMS = dict(
    n_estimators=1800,
    learning_rate=0.07,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.75,
    reg_alpha=0.05,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=100,
)


def key_series(series: pd.Series) -> pd.Series:
    """Stable exact-value keys, including missing as an explicit value."""
    return series.astype("object").where(series.notna(), "__MISSING__")


def align_categories(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    other = other.copy()
    for col in CAT_COLS:
        categories = pd.Index(
            pd.concat([train[col], other[col]], ignore_index=True).dropna().unique()
        )
        dtype = pd.CategoricalDtype(categories=categories)
        train[col] = train[col].astype(dtype)
        other[col] = other[col].astype(dtype)
    return train, other


def target_map(keys: pd.Series, y: np.ndarray, prior: float) -> pd.Series:
    stats = pd.DataFrame({"key": keys.to_numpy(), "y": y}).groupby(
        "key", sort=False, dropna=False
    )["y"].agg(["sum", "count"])
    return (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)


def add_frequency_missing(
    fit: pd.DataFrame, apply: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_out = fit.copy()
    apply_out = apply.copy()
    for col in SOURCE_COLS:
        fit_key = key_series(fit[col])
        apply_key = key_series(apply[col])
        frequencies = fit_key.value_counts(dropna=False) / len(fit_key)
        fit_out[f"{col}__freq"] = fit_key.map(frequencies).astype("float32")
        apply_out[f"{col}__freq"] = apply_key.map(frequencies).fillna(0).astype("float32")
        fit_out[f"{col}__missing"] = fit[col].isna().astype("int8")
        apply_out[f"{col}__missing"] = apply[col].isna().astype("int8")
    fit_out["missing_count"] = fit[SOURCE_COLS].isna().sum(axis=1).astype("int8")
    apply_out["missing_count"] = apply[SOURCE_COLS].isna().sum(axis=1).astype("int8")
    return fit_out, apply_out


def add_cross_fitted_te(
    fit: pd.DataFrame,
    y_fit: np.ndarray,
    apply: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_out = fit.copy()
    apply_out = apply.copy()
    prior = float(np.mean(y_fit))
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)

    for col in SOURCE_COLS:
        fit_key = key_series(fit[col]).reset_index(drop=True)
        apply_key = key_series(apply[col]).reset_index(drop=True)
        cross_fitted = np.full(len(fit), prior, dtype="float32")

        for inner_train, inner_valid in inner.split(np.zeros(len(y_fit)), y_fit):
            mapping = target_map(fit_key.iloc[inner_train], y_fit[inner_train], prior)
            cross_fitted[inner_valid] = (
                fit_key.iloc[inner_valid].map(mapping).fillna(prior).to_numpy(dtype="float32")
            )

        full_mapping = target_map(fit_key, y_fit, prior)
        fit_out[f"{col}__te"] = cross_fitted
        apply_out[f"{col}__te"] = (
            apply_key.map(full_mapping).fillna(prior).to_numpy(dtype="float32")
        )

    return fit_out, apply_out


def build_variant(
    variant: str,
    fit_raw: pd.DataFrame,
    y_fit: np.ndarray,
    valid_raw: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit, valid = align_categories(fit_raw[SOURCE_COLS], valid_raw[SOURCE_COLS])
    if variant in {"frequency_missing", "exact_te"}:
        fit, valid = add_frequency_missing(fit, valid)
    if variant == "exact_te":
        fit, valid = add_cross_fitted_te(fit, y_fit, valid, seed=SEED + fold)
    return fit, valid


def main() -> None:
    start = time.time()
    train = pd.read_csv(ROOT / "train.csv")
    folds = pd.read_csv(ROOT / "prospective_folds.csv")
    if not train[ID_COL].equals(folds[ID_COL]):
        raise ValueError("prospective_folds.csv is not aligned with train.csv")

    development = folds["fold"].to_numpy() >= 0
    # Holdout target is intentionally never sliced or scored below.
    y_all = train[TARGET].to_numpy()
    results: dict[str, dict] = {}
    oof_by_variant = {
        variant: np.full(len(train), np.nan, dtype="float32") for variant in VARIANTS
    }

    for fold in range(5):
        train_mask = development & (folds["fold"].to_numpy() != fold)
        valid_mask = development & (folds["fold"].to_numpy() == fold)
        fit_raw = train.loc[train_mask, SOURCE_COLS].reset_index(drop=True)
        valid_raw = train.loc[valid_mask, SOURCE_COLS].reset_index(drop=True)
        y_fit = y_all[train_mask]
        y_valid = y_all[valid_mask]
        print(f"\n=== external fold {fold}: train={len(fit_raw)} valid={len(valid_raw)} ===", flush=True)

        for variant in VARIANTS:
            variant_start = time.time()
            X_fit, X_valid = build_variant(variant, fit_raw, y_fit, valid_raw, fold)
            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], verbose=False)
            pred = model.predict_proba(X_valid)[:, 1]
            oof_by_variant[variant][valid_mask] = pred.astype("float32")
            fold_auc = roc_auc_score(y_valid, pred)
            print(
                f"{variant:20s} features={X_fit.shape[1]:3d} "
                f"auc={fold_auc:.6f} best_iter={model.best_iteration} "
                f"time={time.time()-variant_start:.0f}s",
                flush=True,
            )
            del X_fit, X_valid, model, pred
            gc.collect()

    for variant in VARIANTS:
        dev_pred = oof_by_variant[variant][development]
        if not np.isfinite(dev_pred).all():
            raise ValueError(f"non-finite development OOF for {variant}")
        fold_scores = []
        for fold in range(5):
            mask = folds["fold"].to_numpy() == fold
            fold_scores.append(float(roc_auc_score(y_all[mask], oof_by_variant[variant][mask])))
        results[variant] = {
            "oof_auc_development": float(roc_auc_score(y_all[development], dev_pred)),
            "fold_scores": fold_scores,
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "fold_worst": float(np.min(fold_scores)),
        }
        np.save(ROOT / f"oof_prospective_{variant}.npy", oof_by_variant[variant])

    results["protocol"] = {
        "holdout_scored": False,
        "development_rows": int(development.sum()),
        "holdout_rows": int((~development).sum()),
        "seed": SEED,
        "inner_folds": INNER_FOLDS,
        "smoothing": SMOOTHING,
        "xgb_params": XGB_PARAMS,
        "runtime_seconds": time.time() - start,
    }
    (ROOT / "prospective_te_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
