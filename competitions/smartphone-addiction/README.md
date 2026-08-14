# Predicting Smartphone Addiction (Playground S6E8)

Première mission Kaggle complète. Classification binaire tabulaire, métrique **ROC AUC**.

## Résultat (au 14 août 2026)
- **Assisté / artefact public : 0.97057** (submission #5, stacking 15 meta-features) — jamais présenté comme autonome.
- **Autonome / from-scratch-code-assisted : 0.96952 public** (submission 55504165, blend 4 modèles XGB/RealMLP/TabM/CatBoost) — benchmark autonome courant.
- Benchmarks autonomes antérieurs : Sprint 2 **0.96923** (XGB/RealMLP/TabM), Sprint 1 **0.96907** (RealMLP + XGB exact-TE), initial **0.96630** (#8, 8 GBDT propres + stacker).
- Rang vérifié du score assisté au snapshot initial : **277 / 1 724** (top 16,1 %).
- Bande estimée du benchmark autonome sur le snapshot de 1 728 équipes : **434–435** (top ~25,1 %) pour Sprint 1 ; Sprint 3 non encore estimé.
- Deadline : 31 août 2026 — le rang final/private n'est pas encore connu.

## Approche
1. EDA : 691k lignes, 9 num + 3 cat, NaN volontaires, cible ~71 %.
2. CV 5-fold stratifiée sur développement gelé (80 %), **holdout historique ouvert une fois** (20 %, code `fold=-2`).
3. **Stacking assisté** : 15 meta-features (OOF/test preds de notebooks publics) + PolynomialFeatures(degré 2) + XGBoost lent (lr=0.01, 20000 arbres) sur GPU → **0.97057 assisté**.
4. **Pipeline autonome** : XGB exact-value TE cross-fitté + RealMLP (Sprint 1) → + TabM (Sprint 2) → + CatBoost exact-catégorie/compositions (Sprint 3), blend de rangs figé sur OOF développement uniquement, portes de gain robustes (5/5 folds + leave-one-fold-out).

## Progression
| Submission | Approche | Public | Catégorie |
|-----------|----------|--------|-----------|
| #1 | Baseline XGB+LGB | 0.96608 | — |
| #2 | Seeds full | 0.96608 | — |
| #3 | Pseudo-labeling | 0.96563 | — |
| #4 | Stacking 4 meta | 0.96978 | assisté |
| **#5** | **Stacking 15 meta** | **0.97057** | **assisté 🏆** |
| #6-7 | 17 meta / top 14 | 0.97053 | assisté |
| **#8** | **8 GBDT propres + stacker (from-scratch)** | **0.96630** | autonome |
| **#9** | **RealMLP + XGB exact-TE (Sprint 1)** | **0.96907** | autonome |
| **S2** | **Blend XGB/RealMLP/TabM (Sprint 2)** | **0.96923** | autonome |
| **S3** | **Blend 4 modèles + CatBoost (Sprint 3)** | **0.96952** | **autonome courant** |

## Leçons clés
- Le stacking de sorties publiques fournit le plus gros gain **assisté** ; il est conservé dans une catégorie séparée et ne compte pas comme preuve autonome.
- Le target encoding exact-value cross-fitté et les signaux de missingness apportent un gain stable aux GBDT.
- RealMLP puis TabM puis CatBoost apportent une diversité autonome réellement complémentaire, validée par des portes de gain strictes (5/5 folds + leave-one-fold-out), pas par un seul micro-gain.
- Une famille très corrélée (LightGBM) ou une feature « réputée » non reproduite fidèlement (decimal lattice) est rejetée même si elle produit un micro-gain.
- Voir `code/prospective_sprint1/` et `code/prospective_sprint2_3/` pour les pipelines reproductibles, et `../../methodology/stacking-massif-technique.md` pour l'étude assistée.
