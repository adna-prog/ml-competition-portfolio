# Optuna and validation correction

## Important audit correction

The earlier reported walk-forward RMSE `1.2556` for the enhanced match/Elo model
was not a valid walk-forward metric. The old runner fitted the regressor on the
same target-tournament rows it then evaluated. The private score `0.520531011`
from the submitted artifact is real, but the local RMSE claim is withdrawn.

The corrected Optuna study uses causal features for every historical training row
and validates only future tournament rows.

## Optuna study

- 20 multi-objective trials.
- Objectives: minimize goals RMSE and maximize weighted stage F1.
- Six bounded CatBoost hyperparameters.
- Walk-forward validation on 2010, 2014, 2018 and 2022.
- No private score or SampleSubmission targets used.

Best Pareto candidates included:

- Trial 19: RMSE `3.329844`, F1 weighted `0.500000`.
- Trial 16: RMSE `3.368039`, F1 weighted `0.515625`.

Trial 16 was selected as the balanced candidate for a practice score check.

## Candidate artifact

`output/optuna_joint_practice_submission.csv`

The candidate keeps the tournament quotas and uses only the supplied historical
data. It has not been scored online yet.
