# Store Sales Time Series Forecasting — Mission d'entraînement #2

Compétition passée d'entraînement (données accessibles). Objectif : apprendre le forecasting
temporel hiérarchique retail. Pas de classement en jeu (compétition terminée).

## Résultat (validation honnête)
- **Meilleur SMAPE : ~0.47** (walk-forward multi-fenêtres, LGBM + features enrichies)
- Métrique : SMAPE, évaluée au seul niveau store×family (1782 séries)

## Progression
| Version | Approche | SMAPE |
|---------|----------|-------|
| v1 | log1p baseline | 0.5341 |
| v2 | log1p + features (45) | 0.5117 |
| v3B | **ventes brutes** + features | 0.4515 (1 fenêtre) |
| v4 | brut + XGB/LGB/blend | 0.4701 (régression) |
| v5 | brut + features enrichies (64) | **0.4663 (3 fenêtres)** |

## Les 3 découvertes clés
1. **Transformation** : prédire les ventes BRUTES (0.45) > log1p (0.51) — log1p pénalise les
   petites valeurs en SMAPE (dénominateur |y|+|p|/2).
2. **Validation honnête** : la walk-forward multi-fenêtres (0.4663) est plus fiable que 1 seule
   fenêtre (0.4515) — valider sur 1 fenêtre surestime.
3. **Réconciliation hiérarchique INUTILE ici** : la métrique SMAPE n'évalue QUE le niveau
   store×family (pas les agrégats) → MinTrace ne fait que contraindre, n'aide pas.

## Ce qui a été appris (réutilisable)
- Validation temporelle walk-forward (jamais k-fold sur données temporelles)
- Features de lag/rolling (lags 1-364, rolling 7-364) par (store, family)
- Covariables : oil, holidays, transactions, promotions
- M5 (Makridakis 2022) : LightGBM + features exogènes + cross-learning = recette gagnante retail
- Vérifier la granularité de la métrique AVANT de choisir la technique

## Fichiers
- `code/eda.py`, `improved.py`, `v3_transform.py`, `v5_walk.py`
- Voir `../../methodology/` et le skill `time-series-forecasting`
