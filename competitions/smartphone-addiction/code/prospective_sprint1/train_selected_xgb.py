"""Retrain the selected exact-TE XGBoost and predict sealed holdout/test.

This script never scores the holdout. It verifies that regenerated development
OOF predictions reproduce the prospective ablation before writing artifacts.
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

import experiment_prospective_te as te

ROOT = Path(__file__).resolve().parent
TARGET = "addicted_label"
ID = "id"


def main() -> None:
    results = json.loads((ROOT / "prospective_te_results.json").read_text())
    variants = [name for name in te.VARIANTS if name in results]
    selected = max(variants, key=lambda name: results[name]["oof_auc_development"])
    if selected != "exact_te":
        raise RuntimeError(f"Expected exact_te to win the frozen ablation, got {selected}")

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    folds = pd.read_csv(ROOT / "prospective_folds.csv")
    if not train[ID].equals(folds[ID]):
        raise ValueError("fold IDs are not aligned")

    fold_values = folds["fold"].to_numpy()
    development = fold_values >= 0
    holdout = fold_values == -2
    y = train[TARGET].to_numpy()
    regenerated_oof = np.full(len(train), np.nan, dtype="float32")
    holdout_pred = np.zeros(holdout.sum(), dtype="float32")
    test_pred = np.zeros(len(test), dtype="float32")
    fold_records = []

    apply_all = pd.concat(
        [train.loc[holdout, te.SOURCE_COLS], test[te.SOURCE_COLS]],
        ignore_index=True,
    )
    n_holdout = int(holdout.sum())

    for fold in range(5):
        start = time.time()
        fit_mask = development & (fold_values != fold)
        valid_mask = fold_values == fold
        fit_raw = train.loc[fit_mask, te.SOURCE_COLS].reset_index(drop=True)
        valid_raw = train.loc[valid_mask, te.SOURCE_COLS].reset_index(drop=True)
        combined_apply = pd.concat([valid_raw, apply_all], ignore_index=True)
        y_fit = y[fit_mask]
        X_fit, X_apply = te.build_variant(
            selected, fit_raw, y_fit, combined_apply, fold
        )
        n_valid = int(valid_mask.sum())
        X_valid = X_apply.iloc[:n_valid]
        X_holdout = X_apply.iloc[n_valid:n_valid + n_holdout]
        X_test = X_apply.iloc[n_valid + n_holdout:]

        model = xgb.XGBClassifier(**te.XGB_PARAMS)
        model.fit(X_fit, y_fit, eval_set=[(X_valid, y[valid_mask])], verbose=False)
        valid_pred = model.predict_proba(X_valid)[:, 1]
        regenerated_oof[valid_mask] = valid_pred.astype("float32")
        holdout_pred += model.predict_proba(X_holdout)[:, 1].astype("float32") / 5
        test_pred += model.predict_proba(X_test)[:, 1].astype("float32") / 5
        fold_records.append({
            "fold": fold,
            "auc": float(roc_auc_score(y[valid_mask], valid_pred)),
            "best_iteration": int(model.best_iteration),
            "runtime_seconds": time.time() - start,
        })
        print(json.dumps(fold_records[-1]), flush=True)
        del X_fit, X_apply, X_valid, X_holdout, X_test, model
        gc.collect()

    reference = np.load(ROOT / "oof_prospective_exact_te.npy")
    max_abs_difference = float(np.nanmax(np.abs(reference[development] - regenerated_oof[development])))
    if max_abs_difference > 1e-7:
        raise RuntimeError(f"OOF regeneration mismatch: {max_abs_difference}")
    if not np.isfinite(holdout_pred).all() or not np.isfinite(test_pred).all():
        raise ValueError("non-finite predictions")

    pd.DataFrame({
        ID: train.loc[holdout, ID].to_numpy(), "holdout_pred": holdout_pred,
    }).to_csv(ROOT / "xgb_exact_te_holdout_predictions.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
        ROOT / "submission_xgb_exact_te_prospective.csv", index=False
    )
    metadata = {
        "selected_variant": selected,
        "selection_data": "development_oof_only",
        "holdout_scored": False,
        "development_oof_auc": float(roc_auc_score(y[development], regenerated_oof[development])),
        "oof_max_abs_difference_vs_ablation": max_abs_difference,
        "folds": fold_records,
    }
    (ROOT / "xgb_exact_te_prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
