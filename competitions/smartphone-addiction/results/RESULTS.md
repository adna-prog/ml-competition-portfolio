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
- **#1** : Blend XGB+LGB baseline — public **0.96608**
- **#2** : Blend seeds full — public **0.96608** (identique)
- **#3** : Pseudo-labeling — public **0.96563** (PIRE)
- **#4** : Stacking massif 4 meta + poly — public **0.96978** 🚀
- **#5 : Stacking massif 15 meta + poly — public 0.97057** 🏆 MEILLEURE
- **#6** : Stacking massif 17 meta + poly — public **0.97053** (léger recul)

## LEÇON MAJEURE : le stacking massif est LA technique
- OOF/test preds de modèles publics comme **meta-features** + **PolynomialFeatures(degré 2)** + **XGBoost lent (lr=0.01, 20000 arbres, GPU)**.
- 4 meta : CV 0.96853 → public 0.96978
- **15 meta : CV 0.96944 → public 0.97057** (meilleur)
- 17 meta : 0.97053 (les features redondantes dégradent — plus n'est pas toujours mieux)
- **Plateau atteint ~0.97057**, à 0.0007 du top 1 (0.97124).
- Le gain vient de la DIVERSITÉ (GBDT + NN + AutoML + lookup transformer).
- Méthode : `kaggle kernels output` des notebooks publics → dataset Kaggle unifié → notebook GPU.

## Notes
- XGBoost > LightGBM; CatBoost déprioritisé (lent); FE rejeté (bruit).
- NaN: LGB/XGB natif; CatBoost require conversion NaN cat → 'missing' (wrapper cat_factory).
- Public score > OOB (test public plus favorable).
