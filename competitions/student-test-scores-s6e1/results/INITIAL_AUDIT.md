# Initial audit — S6E1 Predicting Student Test Scores

## Competition contract

- Kaggle slug: `playground-series-s6e1`
- Status: closed / late submission mode
- Official metric: **RMSE**
- Target: `exam_score`
- Submission schema: `id,exam_score`
- Train: 630,000 rows, 13 columns
- Test: 270,000 rows, 12 columns
- Sample submission: 270,000 rows
- Train IDs: `0..629999`
- Test IDs: `630000..899999`
- ID overlap: 0
- Duplicate train/test rows: 0
- Missing values: none detected

## Target

```text
mean : 62.5067
std  : 18.9169
min  : 19.599
max  : 100.0
```

## Frozen baseline

Three-fold shuffled KFold, `random_state=20260873`, with `id` excluded:

```text
fold 0 RMSE : 8.756635
fold 1 RMSE : 8.777072
fold 2 RMSE : 8.744316
OOF RMSE    : 8.759351
```

Majority/mean predictor RMSE:

```text
18.916869
```

The first clean baseline submission completed:

- Submission ref: `55693166`
- Public RMSE: `8.71849`
- Private RMSE: **`8.74614`**
- Local 3-fold OOF RMSE: `8.759351`

The private score is slightly better than the local OOF estimate, so the baseline
transfers cleanly. It is now the protected S6E1 rollback.

Input hashes are recorded in the intake run; raw data remains outside the
portfolio repository.
