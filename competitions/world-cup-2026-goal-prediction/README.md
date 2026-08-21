# World Cup 2026 Goal Prediction Challenge — Practice Reconstruction

Zindi practice project for the closed **World Cup 2026 Goal Prediction Challenge**.
The competition is completed; this repository contains a reproducible historical-only
reconstruction, not an official winning submission.

## Verified outcome

| Artifact | Private score |
|---|---:|
| Heuristic baseline | `0.439492695` |
| Supervised historical model | `0.446025129` |
| Joint quota-constrained model | `0.515828600` |
| **Optuna causal joint model** | **`0.543579701`** |
| Top-1 reference from user-provided leaderboard | `0.629426195` |

The score is a private practice score reported after using Zindi's score tool.
No official rank is claimed.

## Method

- Historical Fjelstul World Cup data only.
- Causal team-history features: no future tournament row is used to build a feature.
- Match-history and Elo features from the supplied database.
- CatBoost goals regression.
- Tournament-wide ranking with exact 48-team stage quotas.
- Bounded Optuna multi-objective search over causal walk-forward folds.
- Submission schema: `ID,total_goals,Target`.

The quota-constrained joint model was substantially better than independent stage
classification. The protected practice champion is the Optuna joint model.

## Validation

Walk-forward validation uses historical tournament years (2010, 2014, 2018, 2022).
Goals RMSE and stage F1 are reported separately. The challenge's hidden score
normalisation is not reverse-engineered from a single private score.

## Included code

- `code/walk_forward_baseline.py` — causal heuristic baseline.
- `code/supervised_walk_forward.py` — causal goals/stage supervised models.
- `code/joint_prediction.py` — quota-constrained ranking.
- `code/optuna_goal_tuning.py` — bounded multi-objective Optuna study.
- `code/optuna_joint_submission.py` — Optuna candidate generation.
- `code/dixon_coles_monte_carlo.py` — corrected experimental simulation.

## Reproducibility and exclusions

Raw competition data, submissions, caches and private credentials are excluded.
The Monte Carlo experiment is retained as educational code only: official 2026
groups/bracket were not present in the supplied archive, so its assumptions are
not a validated reconstruction of the tournament.

Public repositories were audited for methodology but no third-party code,
post-competition data, current rankings, odds or live results were copied into
the historical-only benchmark. See `results/PUBLIC_SOLUTIONS_AUDIT.md`.
