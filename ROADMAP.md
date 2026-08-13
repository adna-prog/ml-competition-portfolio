# Roadmap ambitieuse — de la reproduction au niveau compétitif autonome

Date de référence : 13 août 2026.

## Diagnostic de départ

| Preuve | Résultat vérifié | Interprétation |
|---|---:|---|
| S6E8 — stack de 15 sorties publiques | 0.97057 public, rang 277/1 724 (top 16,1 %) | Bonne orchestration/reproduction, pas une preuve autonome |
| S6E8 — 8 modèles propres + stacker | 0.96630 public, rang estimé ~635–638 (top ~36,8 %) | Benchmark autonome initial |
| S6E8 — RealMLP + XGB exact-TE prospectifs | 0.96907 public, bande estimée 434–435/1 728 (top ~25,1 %) | Nouveau benchmark autonome ; objectif top 10 % non atteint |
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

La prochaine étape n'est pas d'empiler davantage de travail public. Le sprint prospectif a réduit
l'écart **0.97057 assisté vs 0.96907 autonome** à 0.00150 grâce à RealMLP et au target encoding exact-value.
La suite doit chercher un signal autonome réellement complémentaire plutôt que multiplier les seeds corrélées.
