# Predicting Smartphone Addiction (Playground S6E8)

Première mission Kaggle complète. Classification binaire tabulaire, métrique **ROC AUC**.

## Résultat
- **Score assisté par sorties publiques : 0.97057** (submission #5, stacking 15 meta-features)
- **Nouveau benchmark autonome prospectif : 0.96907 public** (submission #9, RealMLP + XGB exact-TE)
- Ancien benchmark autonome : 0.96630 (submission #8, 8 GBDT propres + stacker)
- Rang vérifié du score assisté au snapshot initial : **277 / 1 724** (top 16,1 %)
- Bande estimée du nouveau score autonome sur le snapshot de 1 728 équipes : **434–435** (top ~25,1 %)
- Deadline : 31 août 2026 — le rang final/private n'est pas encore connu

## Approche
1. EDA : 691k lignes, 9 num + 3 cat, NaN volontaires, cible ~71%.
2. CV 5-fold stratifiée, features brutes.
3. **Stacking assisté** : 15 meta-features (OOF/test preds de notebooks publics) + PolynomialFeatures(degré 2) + XGBoost lent (lr=0.01, 20000 arbres) sur GPU.
4. **Score assisté : 0.97057 public ; nouveau score autonome prospectif : 0.96907 public.**

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
| **#9** | **RealMLP + XGB exact-TE, blend figé sur OOF prospectifs** | **0.96907** |

## Leçons clés
- Le stacking de sorties publiques a fourni le plus gros gain **assisté** sur cette compétition.
- Le target encoding exact-value cross-fitté et les signaux de missingness apportent un gain stable aux GBDT.
- RealMLP fournit la diversité autonome la plus utile observée jusqu'ici.
- Le blend prospectif figé (70 % RealMLP / 30 % XGB exact-TE) a confirmé son gain sur holdout avant soumission.
- Voir `code/prospective_sprint1/` pour le pipeline reproductible et `../../methodology/stacking-massif-technique.md` pour l'étude assistée.
