# Roadmap ambitieuse — de la reproduction au niveau compétitif autonome

Date de référence : 14 août 2026.

## Diagnostic de départ

| Preuve | Résultat vérifié | Interprétation |
|---|---:|---|
| S6E8 — stack de 15 sorties publiques | 0.97057 public, rang 277/1 724 (top 16,1 %) | Bonne orchestration/reproduction, pas une preuve autonome |
| S6E8 — 8 modèles propres + stacker | 0.96630 public, rang estimé ~635–638 (top ~36,8 %) | Benchmark autonome initial |
| S6E8 — RealMLP + XGB exact-TE prospectifs (S1) | 0.96907 public, bande estimée 434–435/1 728 (top ~25,1 %) | Benchmark autonome de référence au départ des Sprints 2–3 |
| S6E8 — + TabM (Sprint 2) | 0.96923 public | Blend XGB/RealMLP/TabM, gain OOF +0.0001566 |
| S6E8 — + CatBoost (Sprint 3) | 0.96952 public, submission 55504165 | Gain OOF +0.0002594 |
| S6E8 — + imputation prédictive (Sprint 4) | 0.96970 public, submission 55509925 | **Benchmark autonome courant** ; gain OOF +0.000185 vs Sprint 3 |
| Store Sales | Non validé | Mauvaise métrique (SMAPE au lieu de RMSLE) + leakage temporel |

Seuils S6E8 au 13 août : top 10 % = **0.97086**, top 5 % = **0.97097**, top 1 % = **0.97106**.

## Porte 0 — Intégrité expérimentale (faite)

- Séparer systématiquement : **autonome**, **assisté par artefacts publics**, public LB, private LB.
- Vérifier la métrique officielle avant d'écrire le pipeline.
- Auditer chaque feature cible pour causalité/leakage.
- Pour le stacking : CV nested ou holdout final intact ; une simple re-CV des OOF est diagnostique.
- Conserver config, folds, OOF, test predictions, runtime et coût de chaque expérience.

## Sprint 1 — S6E8 : viser top 10 % from-scratch avant le 31 août

### Objectif
Passer du nouveau benchmark autonome **0.96907** au seuil top 10 % (**0.97086** au snapshot de référence),
sans prédictions de notebooks publics. Le sprint prospectif RealMLP + exact-TE a réduit l'écart de
0.00456 à 0.00179, mais la porte top 10 % reste ouverte.

### Plan d'expériences, par ROI

1. **Gel prospectif d'une validation secondaire**
   - Conserver les folds actuels pour comparabilité.
   - Créer un holdout stratifié fixe (seed documentée) pour départager les gains futurs.
   - Ne pas sélectionner un modèle sur un gain inférieur au bruit inter-fold.

2. **XGBoost exact-value TE + fréquence**
   - Target encoding exact-value et interactions, fit uniquement dans le fold.
   - Frequency/count encoding sans cible.
   - Missing indicators et `missing_count`.
   - Ablation contrôlée : raw → +freq → +TE → +missing.

3. **Diversité non-GBDT propre sur Kaggle GPU**
   - RealMLP en priorité (5 folds, OOF + test, 2 seeds si gain stable).
   - TabM ensuite seulement si RealMLP apporte une erreur réellement décorrélée.
   - Preprocessing NN fit dans chaque fold : imputation, scaling, catégories.

4. **Ensemble autonome discipliné**
   - Bases : meilleur XGB-TE, LGB, CatBoost, RealMLP, éventuellement TabM.
   - Mesurer AUC individuelle **et corrélation/résidus d'erreur**.
   - Comparer rank-average, blend contraint, logit blend et stacker régularisé.
   - Validation finale sur holdout/nested ; pas de polynomial-XGB accepté sur seule re-CV des OOF.

5. **Budget de submissions**
   - Soumettre uniquement si gain local supérieur au bruit et confirmé sur tous les folds/holdout.
   - Garder deux finales : modèle autonome CV-trusted et ensemble autonome diversifié.

### Porte de sortie
- Minimum : **top 20 % autonome** avec pipeline reproductible.
- Cible : **top 10 % autonome**.
- Stretch : **top 5 % autonome**.
- En cas d'échec : publier honnêtement l'écart et les ablations ; ne pas masquer derrière le stack public.

## Sprints 2–3 — S6E8 : diversité autonome (fait, 14 août 2026)

> Note de numérotation : les Sprints 2–3 du **projet S6E8** (TabM puis CatBoost) sont terminés ;
> ils sont distincts des Sprints 2–3 de la **roadmap** ci-dessous (Store Sales, puis nouvelles Playgrounds).

### Sprint 2 — TabM entre dans le blend (public 0.96923)
- TabM exact-value TE + fréquences fit-only + quantile fold-local : OOF **0.9671391**, inférieur à RealMLP seul mais retenu pour diversité.
- Blend de rangs XGB 21,75 % / RealMLP 50,75 % / TabM 27,50 % : OOF **0.9682957**, gain **+0.0001566**.
- Portes : gain positif à poids fixes 5/5, leave-one-fold-out positif 5/5.

### Sprint 3 — CatBoost rejoint le blend (public 0.96952, submission 55504165)
- CatBoost exact-catégorie + 12 compositions fit-only : OOF **0.9678329** (6 000 arbres, ~30,6 min GPU gratuit).
- Blend de rangs XGB 13,59375 % / RealMLP 31,71875 % / TabM 17,1875 % / CatBoost 37,5 % : OOF **0.9685551**, gain **+0.0002594**.
- Portes : positif à poids fixes 5/5 ; leave-one-fold-out sélectionne 37,5 % sur 5/5 (gain tenu à l'écart +0.0002495).

### Sprint 4 — imputation prédictive augmentée (public 0.96970, submission 55509925)
- Imputation prédictive des 9 numériques par 9 `LGBMRegressor` one-shot par fold (45 au total), sans fallback, baseline 42/12 et candidate 63/12 entraînées dans le même run.
- Candidate OOF **0.9681619** (baseline 0.9678295, gain **+0.0003324**, 5/5 folds) ; 10 CatBoost, runtime ~62 min GPU gratuit.
- Très corrélée à l'ancien CatBoost (Spearman 0.9971) : l'**ajout** plafonne à +0.000118 (sous la porte) ; le **remplacement** fixe de l'ancien CatBoost à son poids gelé 37,5 % atteint OOF **0.9687401**, gain **+0.000185**, positif 5/5.
- Poids Sprint 4 : XGB 13,59375 % / RealMLP 31,71875 % / TabM 17,1875 % / candidate imputée 37,5 %.
- **Public : 0.96970**, gain +0.00018 vs Sprint 3.

### NO-GO
- **LightGBM exact-TE** : trop corrélé à XGB (0.9934), gain marginal +0.000026 < porte utile.
- **LightGBM fortement régularisé** (num_leaves 23, min_child_samples 864, régularisation forte) : meilleur LGBM seul (OOF 0.96672, gain +0.00030 vs LGBM historique, GO 5/5 folds), mais corrélation de rang XGB 0.9944 → ajout au blend à gain **0.000000** → **NO-GO blend** (le tuning GBDT n'ajoute pas de diversité).
- **Nystroem-LR** : AUC fold 0 plafonne à 0.9519 (écart −0.0168 vs blend) → linéarisation par noyau inadaptée → NO-GO.
- **RealMLP + compositions (95 var)** : gain fold 0 +0.000066 < porte +0.00015, corrélation >0.995 → NO-GO.
- **Decimal lattice** : −0.000070 (large) et −0.000267 (reproduction fidèle) sur fold 0.
- **xRFM** : full run NO-GO provisoire sur P100 (`sm_60` sans tensor cores, coût quadratique risqué).
- **Sprint 4 — combinateur rang/prob/logit** : rang 0.9685551 / prob 0.9685530 / logit 0.9685566 ; gain max +0.00000152, sélection LOO instable 4/5 → NO-GO (conserver le blend de rangs).
- **Sprint 4 — géométrie du budget CatBoost** : candidate 0.9680293 vs baseline fraîche 0.9678936 (+0.0001357) et placebo 0.9679650 (+0.0000643), sous les portes +0.00015/+0.00010 → NO-GO.
- **Ablation de blocs CatBoost (fold 0, descriptif)** : signal concentré dans exposition écran (retrait → 0.8538) et engagement (→ 0.9478) ; récupération/contraintes non prioritaire ; contexte quasi neutre — observation, pas une promotion.
- **Interactions exposition × engagement (A1)** : candidate 0.965180 vs baseline 0.965301 (−0.000120) et placebo 0.965211 (−0.000031) → aucun signal propre, les produits/ratios croisés de numériques sont déjà captés par les splits CatBoost → NO-GO.
- **Features non supervisées (A3)** : candidate 0.965185 vs baseline 0.965334 (−0.000149) et placebo 0.965276 (−0.000091) → aucun signal propre, la structure globale des profils (cluster/anomalie) est déjà captée → NO-GO. Le gisement de features S6E8 est épuisé.
- **Stacker niveau 2 (C1)** : LR sur les 4 rangs OOF (LOO) gagne +0.000061 (5/5 folds) vs rank-average, +0.000048 avec interactions de rangs → sous la porte +0.00015, la combinaison rank-average est quasi-optimale → NO-GO.

### Prochaine porte sur S6E8
- Écart au seuil top 10 % observé (0.97086) : ~0.00134. Il faut un signal autonome réellement
  complémentaire de plus, validé par les mêmes portes de gain, pas un empilement de seeds corrélées.

## Sprint 2 — Reconstruire Store Sales correctement

### Objectif
Transformer l'incident en une preuve robuste de forecasting.

1. RMSLE officielle + seasonal-naive (lag 7/14/364) comme baselines.
2. Trois fenêtres walk-forward de 16 jours ; moyenne, écart-type, pire fenêtre.
3. Features causales uniquement : `shift(1).rolling(...)`, expanding means décalées.
4. Stratégie multi-horizon explicite : directe par horizon en premier, recursive en comparaison.
5. LightGBM sur `log1p(sales)` + covariables réellement connues dans le futur.
6. Submission Kaggle réelle ; rapport CV↔LB et tests anti-leakage.

### Porte de sortie
- Battre seasonal-naive d'au moins **10 % en RMSLE moyenne** sur les mêmes fenêtres.
- Une submission complète et reproductible.
- Aucun claim de rang avant score leaderboard vérifié.

## Sprint 3 — Deux Playgrounds from-scratch

1. Prochaine Playground compatible tabulaire/time series : objectif **top 10 % autonome**.
2. Playground suivante : objectif **top 5 % autonome**.
3. Sur chaque compétition :
   - baseline en 24 h ;
   - CV verrouillée avant lecture intensive du leaderboard ;
   - 80 % du calcul sur idées/ablation, 20 % sur tuning ;
   - rapport final avec scores autonomes et assistés séparés.

## Porte Prize — uniquement après preuve autonome

Entrer sur une compétition à prix seulement si :

- type tabulaire ou time series (pas agents/RL/vision pour l'instant) ;
- au moins 21 jours avant l'entry deadline ;
- métrique et validation reproductibles localement ;
- deux résultats récents : un top 10 % et un top 5 % **from-scratch** ;
- budget calcul estimé avant lancement ; aucun RunPod payant sans accord explicite.

Les compétitions Pokémon TCG et Kaggriculture sont écartées : simulations/agents, hors plan actuel.

## Infrastructure à atteindre

- `experiments.csv` : id, commit, folds, features, modèle, seed, CV mean/std/worst, runtime, coût.
- `folds.npy` versionné ou régénérable par seed/hash.
- Tests : métrique, alignement IDs, absence de NaN, plage des prédictions, anti-leakage temporel.
- Un notebook final propre par compétition + script CLI équivalent.
- CI légère : compilation Python, validation notebooks, scan de secrets.

## Principe directeur

La prochaine étape n'est pas d'empiler davantage de travail public. Les Sprints 1→3 ont réduit l'écart
**0.97057 assisté vs 0.96952 autonome** à 0.00105 grâce à RealMLP, TabM et CatBoost (diversité réelle,
validée par portes de gain 5/5 folds + leave-one-fold-out), tout en rejetant les familles corrélées
(LightGBM) et les features non reproductibles (decimal lattice). La suite doit continuer à chercher un
signal autonome réellement complémentaire plutôt que multiplier les seeds corrélées.
