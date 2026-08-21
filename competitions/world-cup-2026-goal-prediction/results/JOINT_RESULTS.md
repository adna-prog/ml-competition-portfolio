# Joint prediction practice results

## Protocol

- Historical causal team features only.
- CatBoost goals regressor.
- Joint stage assignment by ranking predicted goals.
- Stage quotas are tournament-size aware:
  - 48 teams: 16 group, 16 roundof32, 8 roundof16, 4 qf, 2 sf, 1 runnerup, 1 champion.
- Walk-forward validation on 2010, 2014, 2018 and 2022.

## Alpha screen

The joint score was `predicted_goals + alpha * last_stage_rank`.

| Alpha | RMSE goals | F1 weighted stage | F1 macro stage |
|---:|---:|---:|---:|
| 0.00 | 1.5437 | **0.6406** | **0.4479** |
| 0.25 | 1.5437 | 0.6016 | 0.4219 |
| 0.50 | 1.5437 | 0.6016 | 0.4297 |
| 1.00 | 1.5437 | 0.5391 | 0.3359 |
| 2.00 | 1.5437 | 0.5000 | 0.2760 |

The best joint policy is therefore ranking by predicted goals alone, with hard
stage quotas. Historical stage rank adds noise.

## Practice artifact

`output/joint_practice_submission.csv`

Format checks passed: 48 rows, aligned IDs, unique IDs, non-negative goals, and
2026 stage quota counts exactly respected.

No online submission is made; the competition is closed.
