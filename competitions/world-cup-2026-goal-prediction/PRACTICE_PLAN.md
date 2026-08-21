# Practice design — World Cup 2026 Goal Prediction Challenge

## Scope

This is a closed, completed Zindi challenge used only as a historical modeling
laboratory. No online submission and no post-2026 information will be used.

## Verified data

- 489 historical team/tournament rows (1930–2022)
- 48 test teams for the 2026 tournament
- 24 historical tables from the supplied Fjelstul World Cup Database
- 40 exact test-country matches; 8 naming variants/unseen historical names
- `SampleSubmission.csv` is an example/submission artifact, not ground truth

## Target definitions

### Goals

`total_goals` is numeric. The model predicts a non-negative real value; local
RMSE is evaluated on integer historical tournament rows.

### Stage

Historical labels must be normalized to the submission vocabulary. For modern
formats, use:

- `group stage` → `group`
- `round of 16` → `roundof16`
- `quarter-finals` → `qf`
- `semi-finals` → `sf`
- `third-place match` → `sf`
- `final` → reconstruct from the supplied `tournaments.csv` winner field:
  champion if the team is the tournament winner, runner-up otherwise.

Older-format labels (`second group stage`, `final round`) are not directly
comparable to the 2026 bracket and should be excluded from the first stage-model
validation or mapped only under an explicit historical-format rule.

## Validation protocol

### Main protocol: rolling tournament validation

For each validation tournament year `Y`:

1. train only on rows with `year < Y`;
2. construct team-history features using only those rows;
3. predict rows from tournament `Y`;
4. evaluate goals RMSE and stage macro/weighted F1 separately.

Use validation years 2010, 2014, 2018 and 2022 where sufficient history exists.

### Features

Fit-only historical features per team:

- number of prior World Cups;
- prior goals mean, median and recent mean;
- last observed goals;
- goals trend over prior tournaments;
- best prior stage and recent stage;
- prior matches played;
- confederation and region historical aggregates fit on past rows only;
- recency since last appearance;
- participation count and historical qualification frequency.

### Leakage controls

- Never use the target tournament row to construct its own features.
- Never use `SampleSubmission.csv` target values.
- Never use 2026 results, rankings, odds or external information.
- Never use future historical rows when simulating an earlier tournament.

## Baselines

1. Historical team mean goals + most recent stage.
2. Shrunk goals mean toward tournament/confederation prior.
3. CatBoost/LightGBM on causal team-history features.
4. Stage model with explicit class mapping and macro/weighted F1 diagnostics.

Do not collapse RMSE and F1 into the official normalized score until the
normalization is reproduced exactly from the challenge implementation.

## Deliverables

- `scripts/build_historical_features.py`
- `scripts/walk_forward_baseline.py`
- `output/walk_forward_metrics.json`
- `output/PRACTICE_RESULTS.md`
- `output/submission_practice.csv` only after local validation

No submission file will be sent to Zindi; this is practice-only.
