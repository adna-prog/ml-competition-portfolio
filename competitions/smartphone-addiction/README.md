# Predicting Smartphone Addiction (Playground S6E8)

Première mission Kaggle complète. Classification binaire tabulaire, métrique **ROC AUC**.

## Résultat
- **Score assisté par sorties publiques : 0.97057** (submission #5, stacking 15 meta-features)
- **Score autonome from-scratch : 0.96630** (submission #8, 8 GBDT propres + stacker)
- **Rang public du meilleur score au 13 août 2026 : 277 / 1 724 équipes** (top 16,1 %)
- Meilleur score public observé : 0.97124 (écart numérique : 0.00067)
- Deadline : 31 août 2026 — le rang final/private n'est pas encore connu

## Approche
1. EDA : 691k lignes, 9 num + 3 cat, NaN volontaires, cible ~71%.
2. CV 5-fold stratifiée, features brutes.
3. **Stacking assisté** : 15 meta-features (OOF/test preds de notebooks publics) + PolynomialFeatures(degré 2) + XGBoost lent (lr=0.01, 20000 arbres) sur GPU.
4. **Score assisté : 0.97057 public ; score autonome : 0.96630 public.**

## Progression
| Submission | Approche | Public |
|-----------|----------|--------|
| #1 | Baseline XGB+LGB | 0.96608 |
| #2 | Seeds full | 0.96608 |
| #3 | Pseudo-labeling | 0.96563 |
| #4 | Stacking 4 meta | 0.96978 |
| **#5** | **Stacking 15 meta** | **0.97057** 🏆 |
| #6-7 | 17 meta / top 14 | 0.97053 |
| **#8** | **8 GBDT propres + stacker (from-scratch)** | **0.96630** |

## Leçons clés
- Le stacking de sorties publiques a fourni le plus gros gain **assisté** sur cette compétition.
- **Diversité > perfection** : GBDT + NN + AutoML.
- Les features faibles en AUC individuelle peuvent apporter de la diversité.
- La priorité suivante est de recréer cette diversité avec nos propres RealMLP/TabM/TE.
- Voir `../../methodology/stacking-massif-technique.md`.
