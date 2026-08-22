# Predicting Student Health Risk — Kaggle S6E7

Clean practice reconstruction for the Kaggle Playground Series Season 6 Episode 7
competition.

## Verified final result

| Pipeline | Private balanced accuracy |
|---|---:|
| Initial CatBoost, wrong local metric | `0.86792` |
| Initial LightGBM, wrong local metric | `0.87457` |
| Corrected target-encoded tree core | `0.94969` |
| **RunPod GPU MLP + tree meta-blend** | **`0.95029`** |

Best clean submission: `55690196`.

The final calibration candidate was submitted separately as `55691824`; its
result should be recorded only after Kaggle completes it. No private score is
used for training or local model selection.

## Important metric correction

The official metric is **balanced accuracy**, not ordinary accuracy. The initial
local work optimized ordinary accuracy and was not comparable to the leaderboard.
The corrected pipeline uses balanced accuracy throughout.

## Corrected core

- `id` excluded from every model.
- Five frozen stratified folds.
- Fold-fitted numeric imputation.
- Fold-safe multiclass categorical target encoding.
- Causal `sleep_bin` and `stress_level × sleep_bin` interaction.
- Balanced class weights.
- CatBoost + LightGBM OOF probabilities.
- OOF-only blend and decision calibration.

## MLP extension

The MLP extension was executed on a RunPod RTX 4090 with:

- cross-fitted target encoding;
- categorical embeddings;
- discretized numeric features;
- robust scaling;
- periodic numeric bases;
- class-weighted loss;
- EMA checkpoints;
- tree/MLP meta-blending.

The clean MLP meta-blend improved the private score by `+0.00060` over the
corrected tree core.

## Experiments rejected

- ordinary-accuracy Optuna;
- missingness feature block;
- grouped interaction block;
- adversarial reweighting;
- feature removal based on adversarial shift;
- pseudo-labeling at confidence `>0.99`;
- public-parameter CatBoost reproduction;
- three-model tree blend in the tested configuration.

## Public notebook audit

Several high-scoring public notebooks were inspected. Some were clean and useful
for methodology. Others explicitly used public submission pools, score-derived
file names, test-ID overrides and leaderboard probing. Those artifacts were not
used in the clean benchmark. See the audit files in `results/`.

## Data and submissions

Competition data, credentials, probability archives and submission CSVs are not
included in this portfolio repository. The code is provided for methodological
reproduction only and expects the Kaggle competition files in its runtime input.
