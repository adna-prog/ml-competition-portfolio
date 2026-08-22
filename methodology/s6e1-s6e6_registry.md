# Playground Series S6E1–S6E6 registry

The six slugs are accessible through the configured Kaggle account and are
queued for sequential intake using the competition-grade pipeline.

| Episode | Slug | Competition |
|---|---|---|
| S6E1 | `playground-series-s6e1` | Predicting Student Test Scores |
| S6E2 | `playground-series-s6e2` | Predicting Heart Disease |
| S6E3 | `playground-series-s6e3` | Predict Customer Churn |
| S6E4 | `playground-series-s6e4` | Predicting Irrigation Need |
| S6E5 | `playground-series-s6e5` | Predicting F1 Pit Stops |
| S6E6 | `playground-series-s6e6` | Predicting Stellar Class |

## Execution policy

Process one episode at a time:

1. intake and metric contract;
2. download/hash/audit;
3. baseline and one controlled submission;
4. gap diagnosis;
5. bounded model/Optuna sprint;
6. closed-competition writeup audit when appropriate;
7. freeze and publish before moving to the next episode.

No competition is selected by public leaderboard rank alone. The final/private
score, metric correctness and reproducibility determine the protected result.
