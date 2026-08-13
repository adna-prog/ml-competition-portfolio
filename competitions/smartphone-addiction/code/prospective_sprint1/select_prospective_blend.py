"""Select a blend weight on development OOF only, then evaluate sealed holdout once.

Usage is intentionally split:
1. `select` reads development OOF and writes a frozen blend specification.
2. `evaluate` requires holdout prediction files and reports the one-time holdout score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
TARGET = "addicted_label"


def load_folds_and_target() -> tuple[pd.DataFrame, np.ndarray]:
    train = pd.read_csv(ROOT / "train.csv", usecols=["id", TARGET])
    folds = pd.read_csv(ROOT / "prospective_folds.csv")
    if not train["id"].equals(folds["id"]):
        raise ValueError("fold IDs are not aligned")
    return folds, train[TARGET].to_numpy()


def rank_average(pred: np.ndarray) -> np.ndarray:
    return pd.Series(pred).rank(method="average", pct=True).to_numpy()


def select() -> None:
    folds, y = load_folds_and_target()
    dev = folds["fold"].to_numpy() >= 0
    xgb = np.load(ROOT / "oof_prospective_exact_te.npy")[dev]
    real = pd.read_csv(ROOT / "realmlp_prospective_output" / "realmlp_prospective_oof.csv")
    if not real["id"].equals(folds["id"]):
        raise ValueError("RealMLP OOF IDs are not aligned")
    real_pred = real.loc[dev, "oof_pred"].to_numpy()
    if not np.isfinite(xgb).all() or not np.isfinite(real_pred).all():
        raise ValueError("non-finite development OOF")

    xgb_rank = rank_average(xgb)
    real_rank = rank_average(real_pred)
    candidates = []
    for weight_real in np.linspace(0, 1, 41):
        blend = (1 - weight_real) * xgb_rank + weight_real * real_rank
        fold_scores = []
        for fold in range(5):
            mask = folds.loc[dev, "fold"].to_numpy() == fold
            fold_scores.append(float(roc_auc_score(y[dev][mask], blend[mask])))
        candidates.append({
            "weight_realmlp": float(weight_real),
            "weight_xgb": float(1 - weight_real),
            "oof_auc": float(roc_auc_score(y[dev], blend)),
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "fold_worst": float(np.min(fold_scores)),
        })
    # Primary criterion OOF AUC; tie-break by worst fold and lower NN weight.
    best = max(candidates, key=lambda row: (row["oof_auc"], row["fold_worst"], -row["weight_realmlp"]))
    specification = {
        "selection_data": "development_oof_only",
        "holdout_opened": False,
        "blend_type": "rank_average",
        "best": best,
        "prediction_correlation_pearson": float(np.corrcoef(xgb, real_pred)[0, 1]),
        "prediction_correlation_rank": float(np.corrcoef(xgb_rank, real_rank)[0, 1]),
        "grid": candidates,
    }
    (ROOT / "frozen_blend_spec.json").write_text(json.dumps(specification, indent=2))
    print(json.dumps({k: v for k, v in specification.items() if k != "grid"}, indent=2))


def evaluate() -> None:
    folds, y = load_folds_and_target()
    holdout = folds["fold"].to_numpy() == -2
    spec = json.loads((ROOT / "frozen_blend_spec.json").read_text())
    if spec.get("holdout_opened"):
        raise RuntimeError("holdout already marked as opened")
    xgb = pd.read_csv(ROOT / "xgb_exact_te_holdout_predictions.csv").rename(
        columns={"holdout_pred": "xgb_pred"}
    )
    real = pd.read_csv(
        ROOT / "realmlp_prospective_output" / "realmlp_sealed_holdout_predictions.csv"
    ).rename(columns={"holdout_pred": "realmlp_pred"})
    expected = pd.DataFrame({"id": folds.loc[holdout, "id"].to_numpy(), "target": y[holdout]})
    aligned = expected.merge(xgb, on="id", how="left", validate="one_to_one").merge(
        real, on="id", how="left", validate="one_to_one"
    )
    if len(aligned) != holdout.sum() or aligned[["xgb_pred", "realmlp_pred"]].isna().any().any():
        raise ValueError("holdout IDs are incomplete or duplicated")
    wx = spec["best"]["weight_xgb"]
    wr = spec["best"]["weight_realmlp"]
    blend = wx * rank_average(aligned["xgb_pred"].to_numpy()) + wr * rank_average(
        aligned["realmlp_pred"].to_numpy()
    )
    results = {
        "holdout_opened": True,
        "xgb_auc": float(roc_auc_score(aligned["target"], aligned["xgb_pred"])),
        "realmlp_auc": float(roc_auc_score(aligned["target"], aligned["realmlp_pred"])),
        "frozen_blend_auc": float(roc_auc_score(y[holdout], blend)),
        "weight_xgb": wx,
        "weight_realmlp": wr,
    }
    (ROOT / "sealed_holdout_results.json").write_text(json.dumps(results, indent=2))
    spec["holdout_opened"] = True
    (ROOT / "frozen_blend_spec.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "evaluate"])
    args = parser.parse_args()
    select() if args.mode == "select" else evaluate()
