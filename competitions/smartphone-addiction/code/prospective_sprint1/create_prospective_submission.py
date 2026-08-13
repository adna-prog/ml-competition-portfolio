"""Create the frozen prospective rank-blend submission after holdout approval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET = "addicted_label"


def rank_average(pred: pd.Series) -> np.ndarray:
    return pred.rank(method="average", pct=True).to_numpy()


def main() -> None:
    spec = json.loads((ROOT / "frozen_blend_spec.json").read_text())
    holdout = json.loads((ROOT / "sealed_holdout_results.json").read_text())
    if not spec.get("holdout_opened") or not holdout.get("holdout_opened"):
        raise RuntimeError("sealed holdout has not approved the frozen blend")
    # Require that the frozen blend is no worse than both components on holdout.
    if holdout["frozen_blend_auc"] < max(holdout["xgb_auc"], holdout["realmlp_auc"]):
        raise RuntimeError("frozen blend did not beat the best component on holdout")

    xgb = pd.read_csv(ROOT / "submission_xgb_exact_te_prospective.csv").rename(
        columns={TARGET: "xgb_pred"}
    )
    real = pd.read_csv(
        ROOT / "realmlp_prospective_output" / "submission_realmlp_prospective.csv"
    ).rename(columns={TARGET: "realmlp_pred"})
    merged = xgb.merge(real, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(xgb) or not np.isfinite(merged[["xgb_pred", "realmlp_pred"]]).all().all():
        raise ValueError("test prediction IDs or values are invalid")

    wx = spec["best"]["weight_xgb"]
    wr = spec["best"]["weight_realmlp"]
    pred = wx * rank_average(merged["xgb_pred"]) + wr * rank_average(merged["realmlp_pred"])
    submission = pd.DataFrame({"id": merged["id"], TARGET: pred})
    if not submission[TARGET].between(0, 1).all():
        raise ValueError("submission probabilities outside [0,1]")
    submission.to_csv(ROOT / "submission_prospective_rank_blend.csv", index=False)
    print(submission.describe(include="all"))


if __name__ == "__main__":
    main()
