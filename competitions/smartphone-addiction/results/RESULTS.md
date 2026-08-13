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
- **#6-7** : Stacking massif 17 meta / top 14 — public **0.97053** (léger recul)
- **#8** : **FROM-SCRATCH** 8 GBDT + poly + stacker XGB — public **0.96630**
- **#9** : **AUTONOME PROSPECTIF** RealMLP + XGB exact-TE — public **0.96907**

## Nouveau benchmark autonome prospectif (submission #9)
- Split gelé avant entraînement : 80 % développement (5 folds) + 20 % holdout scellé, seed 20260813.
- XGB raw : **0.963866 OOF développement**.
- + fréquence/missingness : **0.965544** (+0.001678).
- + exact-value TE cross-fitté : **0.966826** (+0.002959 vs raw).
- RealMLP propre : **0.967905**, écart-type folds 0.000185.
- Blend de rangs figé sur OOF : **70 % RealMLP / 30 % XGB**, OOF **0.968139**.
- Holdout ouvert une fois après gel des poids : XGB 0.967143, RealMLP 0.967869, blend **0.968033**.
- Score Kaggle réel : **0.96907 public**, submission 55492274.
- Gain vs ancien benchmark autonome : **+0.00277** ; écart vs score assisté : **−0.00150**.
- Bande estimée sur le snapshot de 1 728 équipes : **434–435** (top ~25,1 %), non rang officiel de submission.
- Aucun OOF/test public ou modèle entraîné public utilisé. L'architecture PyTorch RealMLP est adaptée avec attribution depuis un notebook public ; tous les poids et toutes les prédictions sont produits par notre run.

## Benchmark autonome initial (submission #8 vérifiée)
- 8 modèles propres : 3 XGB + 3 LGB + 2 CatBoost ; aucune prédiction de notebook public.
- **CV moyenne 0.96487 | OOF global 0.96473 | public 0.96630**.
- Rang estimé sur le leaderboard du 13 août : **~635–638 / 1 724** (top ~36,8 %).
- Écart vs score assisté : **−0.00427**. Notre maîtrise autonome est donc intermédiaire ; le
  principal gain actuel vient des prédictions publiques, pas encore de notre modélisation propre.

## Stacking public : technique de reproduction, pas preuve autonome
- OOF/test preds de modèles publics comme **meta-features** + **PolynomialFeatures(degré 2)** + **XGBoost lent (lr=0.01, 20000 arbres, GPU)**.
- 4 meta : CV 0.96853 → public 0.96978
- **15 meta : CV 0.96944 → public 0.97057** (meilleur)
- 17 meta : 0.97053 (les features redondantes dégradent — plus n'est pas toujours mieux)
- **Écart numérique du stack public** : 0.00067 au meilleur score observé, mais rang réel 277/1 724.
- Le gain vient de la DIVERSITÉ (GBDT + NN + AutoML + lookup transformer).
- Méthode : `kaggle kernels output` des notebooks publics → dataset Kaggle unifié → notebook GPU.
- Ce résultat mesure notre capacité à **reproduire et orchestrer** des sorties publiques. La
  performance from-scratch (0.96630) mesure séparément notre niveau autonome.

## Notes
- XGBoost > LightGBM; CatBoost déprioritisé (lent); FE rejeté (bruit).
- NaN: LGB/XGB natif; CatBoost require conversion NaN cat → 'missing' (wrapper cat_factory).
- Public > OOF sur ces submissions ; cela peut refléter un échantillon public plus favorable ou du bruit de leaderboard.
