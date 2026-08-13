# Résultats Kaggle — Predicting Smartphone Addiction (S6E8)
Date: 2026-08-13
Métrique: ROC AUC, 5-fold stratified CV

## Modèles (raw features)
| modèle | config | OOB AUC |
|--------|--------|---------|
| LightGBM | lr=0.05, es=100 | 0.96346 |
| LightGBM fast | lr=0.1 | 0.96105 |
| XGBoost | lr=0.1 | 0.96406 |
| XGBoost full | lr=0.05, n=3000 | 0.96440 |
| **Blend XGB(0.72)+LGB(0.28)** | — | **0.96460 OOB** |

## Feature engineering : ÉCHEC (gain −0.00135) → features brutes
## Résultats submission
- **#1** : Blend XGB+LGB baseline — public **0.96608** (COMPLETE) ✅ MEILLEURE
- **#2** : Blend XGB(0.75)+LGB(0.25) seeds full — public **0.96608** (identique)
- **#3** : Pseudo-labeling [0.2,0.8] — public **0.96563** (PIRE — dégrade)

## Leçon : plateau de score
- Seeds averaging améliore l'OOB (0.96460 → 0.96475) mais **rien sur public (0.96608 = identique)**.
- **Pseudo-labeling dégrade** : sanity local 0.96448 trompeur, public 0.96563 (le modèle ré-apprend ses propres prédictions → bruit).
- La différence top (0.966 → 0.971) ne vient ni du seeds averaging ni du FE ni du pseudo — probablement bruit de seed du leaderboard OU techniques très fines non rentables ici.
- Verdict pod : XGB 0.96465, LGB 0.96390, Cat 0.96154; blend 3 = 0.96475 (CatBoost poids 0).

## Notes
- XGBoost > LightGBM; CatBoost déprioritisé (lent); FE rejeté (bruit).
- NaN: LGB/XGB natif; CatBoost require conversion NaN cat → 'missing' (wrapper cat_factory).
- Public score > OOB (test public plus favorable).
