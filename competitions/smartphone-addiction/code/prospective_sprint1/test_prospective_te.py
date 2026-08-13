"""Unit tests for prospective exact-value target encoding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parent / "experiment_prospective_te.py"
spec = importlib.util.spec_from_file_location("prospective_te", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_unique_keys_do_not_copy_own_target() -> None:
    rows = 20
    fit = pd.DataFrame({col: np.arange(rows) for col in module.NUM_COLS})
    for col in module.CAT_COLS:
        fit[col] = [f"{col}_{i}" for i in range(rows)]
    apply = fit.iloc[:4].copy()
    y = np.array([0, 1] * (rows // 2), dtype=int)

    encoded_fit, encoded_apply = module.add_cross_fitted_te(fit, y, apply, seed=17)
    prior = y.mean()
    for col in module.SOURCE_COLS:
        # Every key is unseen in its inner-training partition, so cross-fitted TE = prior.
        assert np.allclose(encoded_fit[f"{col}__te"], prior)
        # Apply rows are encoded from the external full-fit mapping and remain finite.
        assert np.isfinite(encoded_apply[f"{col}__te"]).all()


def test_frequency_and_missing_are_fit_only() -> None:
    fit = pd.DataFrame({col: [1, 1, np.nan, 2] for col in module.NUM_COLS})
    apply = pd.DataFrame({col: [1, 999, np.nan] for col in module.NUM_COLS})
    for col in module.CAT_COLS:
        fit[col] = ["a", "a", None, "b"]
        apply[col] = ["a", "unseen", None]

    fit_out, apply_out = module.add_frequency_missing(fit, apply)
    for col in module.SOURCE_COLS:
        assert apply_out.loc[1, f"{col}__freq"] == 0
        assert apply_out.loc[2, f"{col}__missing"] == 1
    assert apply_out.loc[2, "missing_count"] == len(module.SOURCE_COLS)


if __name__ == "__main__":
    test_unique_keys_do_not_copy_own_target()
    test_frequency_and_missing_are_fit_only()
    print("prospective TE tests: OK")
