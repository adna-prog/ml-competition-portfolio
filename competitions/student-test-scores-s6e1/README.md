# Predicting Student Test Scores — Kaggle S6E1

Clean late-submission reconstruction for Playground Series Season 6 Episode 1.

## Final protected result

```text
Metric: RMSE
Best private RMSE: 8.73988
Best submission: 55695651
```

Progression:

| Pipeline | Private RMSE |
|---|---:|
| LightGBM baseline | 8.74614 |
| Bounded Optuna LightGBM | 8.74005 |
| Clean engineered LightGBM | **8.73988** |

## Method

- `id` excluded.
- No missing values or duplicate rows.
- Frozen 3-fold RMSE validation.
- LightGBM regression.
- Bounded Optuna search after baseline.
- Clean unsupervised features:
  - high attendance × study hours;
  - ideal sleep;
  - ideal study.
- No external predictions, ID overrides or leaderboard probing.

## Rejected experiments

- CatBoost raw model: inferior RMSE.
- Stronger regularization: inferior RMSE.
- Broader academic interaction block: inferior RMSE.
- ±3 standard-deviation capping: neutral.

The competition is frozen at the best verified private result. Raw data and
submissions are excluded from this portfolio entry.
