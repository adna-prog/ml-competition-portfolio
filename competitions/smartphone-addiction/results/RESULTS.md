# Résultats Kaggle — Predicting Smartphone Addiction (S6E8)
Date: 2026-08-14 (mise à jour Sprints 2–3)
Métrique: ROC AUC, 5-fold CV stratifiée sur développement gelé

## Modèles (raw features, historique)
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
- **#4** : Stacking massif 4 meta + poly — public **0.96978** 🚀 (assisté)
- **#5 : Stacking massif 15 meta + poly — public 0.97057** 🏆 MEILLEURE (assisté)
- **#6-7** : Stacking massif 17 meta / top 14 — public **0.97053** (assisté)
- **#8** : **FROM-SCRATCH** 8 GBDT + poly + stacker XGB — public **0.96630** (autonome)
- **#9** : **AUTONOME SPRINT 1** RealMLP + XGB exact-TE — public **0.96907** (autonome)
- **S2** : **AUTONOME SPRINT 2** blend XGB/RealMLP/TabM — public **0.96923** (autonome)
- **S3** : **AUTONOME SPRINT 3** blend 4 modèles + CatBoost — public **0.96952** (autonome, submission 55504165)

## Séparation honnête des résultats
- **Assisted / public-artifact** : `0.97057` — prédictions/OOF de notebooks publics en meta-features. Reproduction/orchestration, pas preuve autonome.
- **Autonomous / from-scratch-code-assisted** : `0.96952` — aucune prédiction publique, aucun OOF public, aucun poids public importé ; architectures/packages (RealMLP, TabM, CatBoost) utilisés avec attribution, tous poids et prédictions réentraînés.

## Pipeline autonome prospectif (Sprints 1–3)
- Split gelé avant entraînement : 80 % développement (5 folds) + 20 % **holdout historique ouvert une fois** (code `fold=-2`), seed 20260813.
- Le holdout n'est plus utilisé pour sélectionner représentation, modèle ou poids de blend.
- XGB raw : **0.963866 OOF** développement.
- + fréquence/missingness : **0.965544** (+0.001678).
- + exact-value TE cross-fitté : **0.966826** (+0.002959 vs raw).

### Sprint 1 — RealMLP (public 0.96907)
- RealMLP propre : **0.967905**, écart-type folds 0.000185.
- Blend de rangs figé sur OOF : **70 % RealMLP / 30 % XGB**, OOF **0.968139**.
- Holdout ouvert une fois après gel des poids : XGB 0.967143, RealMLP 0.967869, blend **0.968033**.
- Score Kaggle réel : **0.96907 public**, submission 55492274.
- Bande estimée snapshot 1 728 équipes : **434–435** (top ~25,1 %).

### Sprint 2 — TabM (public 0.96923)
- TabM exact-value TE + fréquences fit-only + quantile fold-local : OOF **0.9671391** (folds 0.967425 / 0.967518 / 0.967086 / 0.967222 / 0.967440). Inférieur à RealMLP seul, retenu pour diversité.
- Baseline Sprint 1 (30/70) : OOF 0.9681391.
- Blend de rangs XGB **21,75 %** / RealMLP **50,75 %** / TabM **27,50 %** : OOF **0.9682957**, gain **+0.0001566**.
- Portes : gain positif à poids fixes sur 5/5 folds, leave-one-fold-out positif sur 5/5.
- **Public : 0.96923.**

### Sprint 3 — CatBoost (public 0.96952, submission 55504165)
- CatBoost exact-catégorie + 12 compositions fit-only (sans cible) : OOF **0.9678329** (folds 0.967882 / 0.968236 / 0.967610 / 0.967588 / 0.967857), 6 000 arbres max + early stopping.
- Gain de représentation fold 0 ≈ **+0.001666** vs brut ; runtime ≈ 30,6 min GPU Kaggle gratuit, coût `0 $`.
- Corrélations de rang : XGB 0.9831, RealMLP 0.9801, TabM 0.9836 → forte diversité.
- Blend de rangs XGB **13,59375 %** / RealMLP **31,71875 %** / TabM **17,1875 %** / CatBoost **37,5 %** : OOF **0.9685551**, gain **+0.0002594** vs Sprint 2.
- Portes : positif à poids fixes 5/5, leave-one-fold-out sélectionne 37,5 % sur 5/5 (gain moyen tenu à l'écart **+0.0002495**).
- **Public : 0.96952**, gain +0.00029, très proche du gain OOF prévu.
- Spec figée : `results/prospective_sprint2_3/frozen_catboost_blend_spec.json`.

### Sprint 4 — imputation prédictive augmentée (public 0.96970, submission 55509925)
- Imputation prédictive des 9 numériques par 9 `LGBMRegressor` one-shot par fold (45 au total), sans fallback ; baseline 42/12 et candidate 63/12 dans le même run.
- Candidate OOF **0.9681619** (baseline 0.9678295, gain **+0.0003324**, 5/5 folds) ; 10 CatBoost, runtime ~62 min GPU gratuit, coût `0 $`.
- Corrélations de rang candidate vs ancien CatBoost Spearman **0.9971** : l'ajout plafonne à +0.000118 (sous la porte) ; le **remplacement** fixe à poids gelé 37,5 % atteint OOF **0.9687401**, gain **+0.000185**, positif 5/5.
- Poids Sprint 4 : XGB 13,59375 % / RealMLP 31,71875 % / TabM 17,1875 % / candidate imputée 37,5 %.
- **Public : 0.96970**, gain +0.00018 vs Sprint 3, très proche du gain OOF prévu.

## NO-GO documentés (Sprints 2–4)
- **LightGBM exact-TE** (OOF 0.9664086) : corrélation de rang LGB–XGB ≈ 0.9934, gain marginal blend ≈ +0.000026 < porte utile → rejeté.
- **LightGBM fortement régularisé** (num_leaves 23, min_child_samples 864, régularisation forte) : meilleur LGBM seul (OOF 0.96672, +0.00030 vs LGBM historique, GO 5/5 folds) mais corrélation de rang XGB 0.9944 → ajout au blend à gain 0.000000 → NO-GO blend.
- **Nystroem-LR** : AUC fold 0 plafonne à 0.9519 (écart −0.0168 vs blend) → linéarisation par noyau inadaptée → NO-GO.
- **RealMLP + compositions (95 var)** : gain fold 0 +0.000066 < porte +0.00015, corrélation >0.995 → NO-GO.
- **Decimal lattice** : variante large 45 var (fold-0 −0.000070) et reproduction fidèle publiée (fold-0 −0.000267) → rejeté ; la réputation externe d'une feature ne remplace pas la validation sur nos folds.
- **xRFM** : full run cinq-folds NO-GO provisoire sur P100 (`sm_60` sans tensor cores, coût quadratique risqué) ; micro-screening non comparable à une AUC full-fold.
- **Sprint 4 — combinateur rang/prob/logit** : rang 0.9685551 / prob 0.9685530 / logit 0.9685566 ; gain max +0.00000152, sélection LOO instable 4/5 → NO-GO (conserver le blend de rangs).
- **Sprint 4 — géométrie du budget CatBoost** : candidate 0.9680293 vs baseline fraîche 0.9678936 (+0.0001357) et placebo 0.9679650 (+0.0000643), sous les portes +0.00015/+0.00010 → NO-GO.
- **Ablation de blocs CatBoost (fold 0, descriptif)** : signal concentré dans exposition écran (retrait → 0.8538) et engagement (→ 0.9478) ; récupération/contraintes non prioritaire ; contexte quasi neutre — observation, pas une promotion.
- **Interactions exposition × engagement (A1)** : candidate 0.965180 vs baseline 0.965301 (−0.000120) et placebo 0.965211 (−0.000031) → aucun signal propre, les produits/ratios croisés de numériques sont déjà captés par les splits CatBoost → NO-GO.
- **Features non supervisées (A3)** : candidate 0.965185 vs baseline 0.965334 (−0.000149) et placebo 0.965276 (−0.000091) → aucun signal propre, la structure globale des profils (cluster/anomalie) est déjà captée → NO-GO. Le gisement de features S6E8 est épuisé.
- **Stacker niveau 2 (C1)** : LR sur les 4 rangs OOF (LOO) gagne +0.000061 (5/5 folds) vs rank-average, +0.000048 avec interactions de rangs → sous la porte +0.00015, la combinaison rank-average est quasi-optimale → NO-GO.

## Benchmark autonome initial (submission #8 vérifiée)
- 8 modèles propres : 3 XGB + 3 LGB + 2 CatBoost ; aucune prédiction de notebook public.
- **CV moyenne 0.96487 | OOF global 0.96473 | public 0.96630**.
- Rang estimé leaderboard du 13 août : **~635–638 / 1 724** (top ~36,8 %).
- Écart vs score assisté : **−0.00427**.

## Stacking public : technique de reproduction, pas preuve autonome
- OOF/test preds de modèles publics comme **meta-features** + **PolynomialFeatures(degré 2)** + **XGBoost lent (lr=0.01, 20000 arbres, GPU)**.
- 4 meta : CV 0.96853 → public 0.96978
- **15 meta : CV 0.96944 → public 0.97057** (meilleur, catégorie assistée)
- 17 meta : 0.97053 (features redondantes dégradent).
- Rang réel **277/1 724**.
- Ce résultat mesure notre capacité à **reproduire et orchestrer** des sorties publiques, séparée de notre niveau autonome.

## Notes
- XGBoost > LightGBM; CatBoost déprioritisé (lent) sur les features brutes, mais fort gain via catégories/compositions exactes.
- NaN: LGB/XGB natif; CatBoost require conversion NaN cat → 'missing' (wrapper cat_factory).
- Public > OOF sur ces submissions ; peut refléter un échantillon public plus favorable ou du bruit de leaderboard.
