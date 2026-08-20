import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import sprint5_kaggle_kernel as s5

DATA = Path(os.environ.get("S6E8_DATA", "/workspace/s6e8/data"))
OUT = Path(os.environ.get("S6E8_OUT", "/workspace/s6e8/output"))
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sub = pd.read_csv(DATA / "sample_submission.csv")
train, test = s5.prepare_data(train, test)

cb_oof = np.load(OUT / "oof_catboost.npy")
cb_test = np.load(OUT / "test_catboost.npy")
lgb_oof = np.load(OUT / "oof_lightgbm.npy")
lgb_test = np.load(OUT / "test_lightgbm.npy")

nn_oof, nn_test, nn_metrics = s5.train_neural_net(
    train, test, n_folds=11, seed=42, epochs=32
)
np.save(OUT / "oof_nn.npy", nn_oof)
np.save(OUT / "test_nn.npy", nn_test)
(OUT / "nn_metrics.json").write_text(json.dumps(nn_metrics, indent=2))

rank = lambda x: rankdata(x) / len(x)
oof_blend = 0.49 * rank(nn_oof) + 0.31 * rank(cb_oof) + 0.21 * rank(lgb_oof)
test_blend = 0.49 * rank(nn_test) + 0.31 * rank(cb_test) + 0.21 * rank(lgb_test)
blend_auc = roc_auc_score(train[s5.TARGET], oof_blend)
print(f"NN OOF AUC: {nn_metrics['overall']:.6f}")
print(f"Blend OOF AUC: {blend_auc:.6f}")

sub[s5.TARGET] = test_blend
sub.to_csv(OUT / "submission_sprint5.csv", index=False)
np.save(OUT / "oof_blend.npy", oof_blend)
np.save(OUT / "test_blend.npy", test_blend)
(OUT / "blend_metrics.json").write_text(json.dumps({
    "blend_oof_auc": float(blend_auc),
    "weights": {"nn": 0.49, "catboost": 0.31, "lightgbm": 0.21},
}, indent=2))
print(f"SUBMISSION_SAVED {OUT / 'submission_sprint5.csv'}")
