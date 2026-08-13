"""Freeze prospective development/holdout and external CV folds for S6E8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent
SEED = 20260813
HOLDOUT_SIZE = 0.20
N_FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    train_path = ROOT / "train.csv"
    train = pd.read_csv(train_path, usecols=["id", "addicted_label"])
    y = train["addicted_label"].to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLDOUT_SIZE, random_state=SEED
    )
    dev_idx, holdout_idx = next(splitter.split(np.zeros(len(y)), y))

    fold_id = np.full(len(train), -2, dtype=np.int8)  # -2 = untouched holdout
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_local) in enumerate(cv.split(dev_idx, y[dev_idx])):
        fold_id[dev_idx[valid_local]] = fold

    assert np.all(fold_id[dev_idx] >= 0)
    assert np.all(fold_id[holdout_idx] == -2)
    assert not np.intersect1d(dev_idx, holdout_idx).size

    output = pd.DataFrame(
        {"id": train["id"], "split": np.where(fold_id == -2, "holdout", "development"),
         "fold": fold_id}
    )
    output.to_csv(ROOT / "prospective_folds.csv", index=False)

    metadata = {
        "seed": SEED,
        "holdout_size": HOLDOUT_SIZE,
        "n_folds": N_FOLDS,
        "train_sha256": sha256(train_path),
        "rows": len(train),
        "development_rows": int(len(dev_idx)),
        "holdout_rows": int(len(holdout_idx)),
        "target_rate_all": float(y.mean()),
        "target_rate_development": float(y[dev_idx].mean()),
        "target_rate_holdout": float(y[holdout_idx].mean()),
        "fold_counts": output.loc[output.fold >= 0, "fold"].value_counts().sort_index().to_dict(),
    }
    (ROOT / "prospective_folds_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
