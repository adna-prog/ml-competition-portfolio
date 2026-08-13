# Predicting Smartphone Addiction (Playground S6E8)

Première mission Kaggle complète. Classification binaire tabulaire, métrique **ROC AUC**.

## Résultat
- **Public score : 0.97057** (submission #5, stacking massif 15 meta-features)
- **Rang : quasi top-1** (top 1 à 0.97124, écart 0.0007)
- Deadline : 31 août 2026

## Approche
1. EDA : 691k lignes, 9 num + 3 cat, NaN volontaires, cible ~71%.
2. CV 5-fold stratifiée, features brutes.
3. **Stacking massif** : 15 meta-features (OOF/test preds de notebooks publics) + PolynomialFeatures(degré 2) + XGBoost lent (lr=0.01, 20000 arbres) sur GPU.
4. **Score final : 0.97057 public.**

## Progression
| Submission | Approche | Public |
|-----------|----------|--------|
| #1 | Baseline XGB+LGB | 0.96608 |
| #2 | Seeds full | 0.96608 |
| #3 | Pseudo-labeling | 0.96563 |
| #4 | Stacking 4 meta | 0.96978 |
| **#5** | **Stacking 15 meta** | **0.97057** 🏆 |
| #6-7 | 17 meta / top 14 | 0.97053 |

## Leçons clés
- **Stacking massif > tout** : les OOF des modèles publics + interactions polynomiales + GPU.
- **Diversité > perfection** : GBDT + NN + AutoML.
- Les features "faibles" en AUC individuelle apportent quand même de la diversité.
- Voir `../../methodology/stacking-massif-technique.md`.
