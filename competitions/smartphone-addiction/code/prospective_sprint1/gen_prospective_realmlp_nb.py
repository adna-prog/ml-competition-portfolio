"""Build the prospective S6E8 RealMLP Kaggle notebook.

The compact PyTorch RealMLP implementation is adapted from the public notebook
zhenruiweng/realmlp-for-predicting-smartphone-addiction. No model, OOF, test
prediction, or trained artifact is imported. All training runs from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/tmp/realmlp_fast_source/realmlp-for-predicting-smartphone-addiction.ipynb")
OUTPUT_DIR = ROOT / "realmlp_prospective_push"
OUTPUT_DIR.mkdir(exist_ok=True)

source_nb = json.loads(SOURCE.read_text(encoding="utf-8"))
code = {i: "".join(source_nb["cells"][i].get("source", [])) for i in [6, 8, 15, 20]}

# Make the run deterministic and bind it to our prospective protocol.
code[6] = code[6].replace('"random_state":     42,', '"random_state":     20260813,')
code[6] = code[6].replace("SEED = 42", "SEED = 20260813")

intro = nbf.v4.new_markdown_cell("""# S6E8 — Prospective RealMLP from scratch

**Objective:** add a genuinely diverse neural model to the autonomous S6E8 stack.

Protocol fixed before training:
- deterministic 80% development / 20% sealed holdout split (`seed=20260813`);
- five fixed development folds;
- fold-local preprocessing and inner cross-fitted target encoding;
- holdout predictions are generated but **holdout AUC is not computed in this run**;
- no public OOF/test predictions, fitted models, or prediction artifacts are used.

Code provenance: the compact PyTorch RealMLP architecture is adapted from
`zhenruiweng/realmlp-for-predicting-smartphone-addiction`. All models and
predictions in this notebook are trained from scratch by this run.
""")

pipeline = r'''# Prospective fold-local data pipeline and training
from sklearn.model_selection import StratifiedShuffleSplit

DATA = "/kaggle/input/competitions/playground-series-s6e8"
TARGET = "addicted_label"
ID = "id"
N_FOLDS = 5
HOLDOUT_SIZE = 0.20

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y_all = train[TARGET].to_numpy()
raw_X = train.drop(columns=[ID, TARGET])
raw_test = test.drop(columns=[ID])
base_cat_cols = raw_X.select_dtypes(include=["object"]).columns.tolist()
base_num_cols = raw_X.select_dtypes(exclude=["object"]).columns.tolist()

# Reproduce the frozen split exactly.
splitter = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_SIZE, random_state=SEED)
dev_idx, holdout_idx = next(splitter.split(np.zeros(len(y_all)), y_all))
fold_id = np.full(len(train), -2, dtype=np.int8)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (_, valid_local) in enumerate(cv.split(dev_idx, y_all[dev_idx])):
    fold_id[dev_idx[valid_local]] = fold
assert (fold_id[dev_idx] >= 0).all() and (fold_id[holdout_idx] == -2).all()
print("development", len(dev_idx), "sealed holdout", len(holdout_idx), flush=True)


def preprocess_fold(X_fit_raw, others_raw):
    """Fit medians/categories/bins on external training only, transform others."""
    medians = {col: X_fit_raw[col].median() for col in base_num_cols}
    category_map = {}
    X_fit, new_cat = feature_engineering(
        X_fit_raw.copy(), list(base_cat_cols), list(base_num_cols), category_map,
        fit=True, median_values=medians,
    )
    transformed = []
    for frame in others_raw:
        out, _ = feature_engineering(
            frame.copy(), list(base_cat_cols), list(base_num_cols), category_map,
            fit=False, median_values=medians,
        )
        transformed.append(out)
    cat_cols = sorted(base_cat_cols + new_cat)
    columns = sorted(X_fit.columns)
    X_fit = X_fit.reindex(columns=columns)
    transformed = [frame.reindex(columns=columns) for frame in transformed]
    return X_fit, transformed, cat_cols


def add_fold_target_encoding(X_fit, y_fit, others, cat_cols, fold):
    """Inner-cross-fit training TE; transform validation/holdout/test from fit only."""
    te_cols = [col for col in cat_cols if not col.endswith("_bin_")]
    encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED + fold)
    fit_te = encoder.fit_transform(X_fit[te_cols], y_fit)
    te_names = [f"_{col}TE" for col in te_cols]
    X_fit = X_fit.copy()
    X_fit[te_names] = fit_te.astype(np.float32)
    output = []
    for frame in others:
        frame = frame.copy()
        frame[te_names] = encoder.transform(frame[te_cols]).astype(np.float32)
        output.append(frame)
    return X_fit, output


oof = np.full(len(train), np.nan, dtype=np.float32)
holdout_pred = np.zeros(len(holdout_idx), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)
fold_scores = []
fold_times = []

for fold in range(N_FOLDS):
    fit_idx = np.flatnonzero((fold_id >= 0) & (fold_id != fold))
    valid_idx = np.flatnonzero(fold_id == fold)
    fold_start = time.time()
    X_fit, others, cat_cols = preprocess_fold(
        raw_X.iloc[fit_idx],
        [raw_X.iloc[valid_idx], raw_X.iloc[holdout_idx], raw_test],
    )
    X_valid, X_holdout, X_test = others
    X_fit, others = add_fold_target_encoding(
        X_fit, y_all[fit_idx], [X_valid, X_holdout, X_test], cat_cols, fold
    )
    X_valid, X_holdout, X_test = others

    print(f"\nFold {fold}: fit={len(fit_idx)} valid={len(valid_idx)} features={X_fit.shape[1]}", flush=True)
    model = RealMLP_TD_Classifier(**CONFIG)
    model.fit(
        X_fit, y_all[fit_idx], X_valid, y_all[valid_idx],
        cat_col_names=cat_cols,
    )
    valid_pred = model.predict_proba(X_valid)[:, 1]
    oof[valid_idx] = valid_pred.astype(np.float32)
    holdout_pred += model.predict_proba(X_holdout)[:, 1].astype(np.float32) / N_FOLDS
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32) / N_FOLDS
    score = roc_auc_score(y_all[valid_idx], valid_pred)
    fold_scores.append(float(score))
    fold_times.append(float(time.time() - fold_start))
    print(f"Fold {fold} AUC={score:.6f} time={fold_times[-1]/60:.1f}min", flush=True)
    del model, X_fit, X_valid, X_holdout, X_test, others
    gc.collect()
    torch.cuda.empty_cache()

# Only development labels are scored. Holdout labels remain unopened.
dev_mask = fold_id >= 0
dev_auc = roc_auc_score(y_all[dev_mask], oof[dev_mask])
assert np.isfinite(oof[dev_mask]).all() and np.isnan(oof[fold_id == -2]).all()
assert np.isfinite(holdout_pred).all() and np.isfinite(test_pred).all()

pd.DataFrame({
    "id": train[ID], "fold": fold_id, "oof_pred": oof,
}).to_csv("realmlp_prospective_oof.csv", index=False)
pd.DataFrame({
    "id": train.loc[holdout_idx, ID].to_numpy(), "holdout_pred": holdout_pred,
}).to_csv("realmlp_sealed_holdout_predictions.csv", index=False)
pd.DataFrame({
    "id": test[ID], TARGET: test_pred,
}).to_csv("submission_realmlp_prospective.csv", index=False)

results = {
    "model": "compact-pytorch-realmlp",
    "seed": SEED,
    "development_rows": int(dev_mask.sum()),
    "holdout_rows": int((~dev_mask).sum()),
    "holdout_scored": False,
    "fold_scores": fold_scores,
    "development_oof_auc": float(dev_auc),
    "fold_times_seconds": fold_times,
    "public_prediction_artifacts_used": False,
    "code_reference": "zhenruiweng/realmlp-for-predicting-smartphone-addiction",
}
with open("realmlp_prospective_results.json", "w") as handle:
    json.dump(results, handle, indent=2)
print(json.dumps(results, indent=2), flush=True)
'''

# Add gc, which the reference import block does not include.
code[6] = code[6].replace("import os\n", "import os\nimport gc\nimport json\n")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
compatibility = nbf.v4.new_code_cell(r'''# Kaggle currently may assign a Pascal P100 while shipping a PyTorch/CUDA build
# without sm_60 kernels. Install a compatible official PyTorch wheel before import.
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
    "torch==2.7.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118",
])
''')

nb.cells = [
    intro,
    compatibility,
    nbf.v4.new_code_cell(code[6]),
    nbf.v4.new_code_cell(code[8]),
    nbf.v4.new_code_cell(code[15]),
    nbf.v4.new_code_cell(code[20]),
    nbf.v4.new_code_cell(pipeline),
]

notebook_path = OUTPUT_DIR / "s6e8_prospective_realmlp.ipynb"
nbf.write(nb, notebook_path)
metadata = {
    "id": "adnaneel/s6e8-prospective-realmlp-from-scratch",
    "title": "S6E8 Prospective RealMLP From Scratch",
    "code_file": notebook_path.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "competition_sources": ["playground-series-s6e8"],
    "dataset_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(notebook_path)
