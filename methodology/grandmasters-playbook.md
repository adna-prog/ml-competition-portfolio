# Grandmasters Playbook — Bonnes pratiques

Synthèse des techniques des Grandmasters Kaggle (NVIDIA KGMoN, cdeotte, etc.).

## 1. Validation locale fiable (LE fondement)
> "If you can't trust your validation score, you're flying blind."
- CV 0.87 mais LB 0.72 → fold fuite OU distributions diffèrent. Corriger avant tout.
- Adapter la CV à la structure du test :
  - Time-dependent → `TimeSeriesSplit` (pas de shuffle)
  - Test aléatoire → `StratifiedKFold`
  - Distribution shift → adversarial validation
- Objectif : CV corrélée au private LB.

## 2. EDA intelligente
- Analyser le target pour patterns temporels (tendance, saisonnalité).
- Chercher les distribution shifts train/test dès le début.

## 3. Feature engineering = LE vrai edge tabulaire
- "FE contributes more to your final score than any hyperparameter tuning."
- Features ciblées + domain knowledge > changer de modèle.
- ⚠️ Valider chaque feature par test contrôlé (même modèle/config) — rejeter si négatif.

## 4. Ensemblage : diversité > perfection
- 3 modèles qui font des erreurs différentes > 1 modèle parfait.
- XGB + LGB + Cat = minimum tabulaire.
- Conserver les OOF de tous les essais pour le blend.

## 5. Blending vs Stacking
- **Blending** : moyenne pondérée (simple, robuste).
- **Stacking** : méta-modèle (L2) sur les OOF des L1 ; estimer son gain avec CV nested/holdout final.
- Une simple re-CV des OOF peut être optimiste si les folds externes ont influencé les modèles L1.

## 6. Adversarial validation
- Classifieur train-vs-test sur les features.
- AUC élevée → features discriminantes → nuiront à la généralisation.

## 7. Lire le leaderboard comme une carte
- Ne pas sur-optimiser le public LB (30% = bruit) ; le private (70%) décide.
- Sélectionner 2 submissions finales différentes : la "CV-trusted" gagne sur private.

## Checklist
1. Lire la section évaluation AVANT d'écrire du code.
2. Validation leakage-free + corrélée au LB.
3. Baseline solide tôt.
4. FE ciblé validé.
5. Ensemble diversifié (XGB+LGB+Cat + variants).
6. Blend/stack OOF, poids optimisés.
7. Adversarial validation si doute.
8. 2 submissions finales différentes.
