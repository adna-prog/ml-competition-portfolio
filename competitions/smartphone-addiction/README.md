# Predicting Smartphone Addiction (Playground S6E8)

Première mission Kaggle complète. Classification binaire tabulaire, métrique **ROC AUC**.

## Résultat
- **Public score : 0.96608** (submission #1, blend XGB+LGB)
- **Rang : 670e / 1677 équipes**
- Deadline : 31 août 2026

## Approche
1. EDA : 691k lignes, 9 num + 3 cat, NaN volontaires, cible ~71%.
2. CV 5-fold stratifiée, features brutes (le FE a été testé et rejeté — gain négatif).
3. Modèles comparés : XGBoost (0.9644) > LightGBM (0.9635) > CatBoost (0.9615).
4. **Blend XGB(0.72) + LGB(0.28) = 0.96460 OOB → 0.96608 public.**

## Ce qui a été testé (et leçons)
| Levier | Résultat | Verdict |
|--------|----------|---------|
| Baseline XGB+LGB | 0.96608 | ✅ retenu |
| Seeds averaging | = | bruit leaderboard |
| Stacking L2 (XGB+LGB→LR) | +0.003 OOB | ✅ à explorer en full |
| Target Encoding | −0.001 | ❌ catégories pauvres |
| Feature engineering | −0.001 | ❌ bruit sur synthétique |
| Pseudo-labeling | 0.96563 | ❌ dégrade |

## Fichiers
- `code/` : pipeline final, tests, notebook
- `results/` : OOF, submissions, scores
- Voir `../../methodology/` pour les bonnes pratiques utilisées

## Leçons clés
- **Stacking > blending** (le levier des gagnants).
- **Valider chaque technique par CV contrôlée** avant adoption.
- **Seeds averaging = bruit** sur ce dataset (le LB est très bruité).
- Runs lourds → pod CPU (32 vCPU) avec **volume persistant**.
