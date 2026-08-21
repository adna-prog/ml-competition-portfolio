# Practice results — World Cup 2026 Goal Prediction Challenge

## Baseline

- Model: causal historical team-history baseline
- Goals: shrinkage of recent/overall historical team goals
- Stage: last historical stage, with champion/runner-up reconstructed from the supplied tournament winner table
- Validation: walk-forward years 2010, 2014, 2018, 2022
- Local average goals RMSE: `3.462605`
- Local weighted stage F1: `0.440620`
- Local macro stage F1: `0.219044`
- Submission artifact: `output/practice_baseline_submission.csv`

## Real private result

- Heuristic baseline: `0.439492695`
- Supervised independent stage classifier: `0.445130673`
- Joint quota-constrained prediction: `0.515828600`
- Enhanced joint match/Elo model: `0.520531011`
- **Optuna causal joint model: `0.543579701`**
- Gain versus enhanced joint model: **`+0.023048690`**
- Gain versus heuristic baseline: `+0.104087006`
- Relative gain versus previous best: approximately `+4.428%`
- Top-1 reference: `0.629426195`
- Remaining gap to top-1: `0.085846494`

The Optuna v2 candidate with alpha 0.25 scored `0.516151919` privately and is
rejected. It regressed by `-0.027427782` versus the protected Optuna v1 score.
Keep `0.543579701` as the practice champion.

## Interpretation

The baseline is now anchored to a real private evaluation. Any future practice
model must beat this score under the same submission format and must be evaluated
with walk-forward validation first. Do not use the sample submission values as
labels or as a source of 2026 information.
