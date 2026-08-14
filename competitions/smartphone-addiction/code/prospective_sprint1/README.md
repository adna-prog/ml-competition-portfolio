# S6E8 Prospective Sprint 1

Reproducible code for the autonomous RealMLP + exact-value-TE XGBoost benchmark.

## Evidence protocol

1. `freeze_prospective_folds.py` creates a deterministic 80% development / 20% historical-holdout split (opened once in Sprint 1).
2. `experiment_prospective_te.py` compares raw, frequency/missingness, and inner-cross-fitted exact-value target encoding **without scoring holdout**.
3. `s6e8_prospective_realmlp.ipynb` trains all RealMLP weights on Kaggle GPU and writes development OOF plus unscored holdout/test predictions.
4. `select_prospective_blend.py select` freezes rank-blend weights from development OOF only.
5. `train_selected_xgb.py` regenerates the selected XGBoost OOF exactly and writes unscored holdout/test predictions.
6. `select_prospective_blend.py evaluate` opens the (now historical) holdout once and marks it opened in `frozen_blend_spec.json`.
7. `create_prospective_submission.py` refuses to emit a blend unless it beats both components on the historical holdout (opened once, no longer used for selection).

## Verified outcome

- XGBoost exact-TE development OOF: `0.9668255`
- RealMLP development OOF: `0.9679047`
- Frozen 30% XGB / 70% RealMLP rank blend: `0.9681391`
- Sealed holdout: `0.9680330`
- Kaggle public: `0.96907` (submission `55492274`)

Result JSON files are in `../../results/prospective_sprint1/`. Large OOF arrays, raw competition data, and holdout labels are intentionally excluded.

## Dependencies

The local scripts require pandas, NumPy, scikit-learn, and XGBoost. Run them from an environment containing the official competition `train.csv` and `test.csv`; those files are gitignored.

The RealMLP notebook runs on Kaggle and contains its architecture code. Its third-party provenance and Apache 2.0 license are documented in `THIRD_PARTY_NOTICE.md` and `APACHE-2.0.txt`.

`gen_prospective_realmlp_nb.py` records how the executed notebook was assembled. To regenerate it, first download the Apache-2.0 source notebook to:

```text
/tmp/realmlp_fast_source/realmlp-for-predicting-smartphone-addiction.ipynb
```

The already generated and executed notebook is committed as `s6e8_prospective_realmlp.ipynb`.
