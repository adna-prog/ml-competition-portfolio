# ML Competition Portfolio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Mon portfolio de compétitions Kaggle / machine learning — du baseline au stacking,
avec une méthodologie reproductible pour progresser des compétitions gratuites aux payantes.

## 🎯 Objectifs
- Construire un portfolio solide via des compétitions réelles
- Maîtriser les techniques des Grandmasters (CV leakage-free, TE, stacking, ensemblage)
- Documenter chaque compétition : approche, résultats, leçons
- Viser progressivement les compétitions avec prize money

## 📁 Structure
```
ml-competition-portfolio/
├── README.md                 # ce fichier
├── methodology/              # bonnes pratiques et playbook
│   ├── grandmasters-playbook.md
│   ├── tabular-techniques.md
│   └── validation-strategy.md
├── templates/                # scripts réutilisables
│   ├── cv_pipeline.py        # CV leakage-free + OOF
│   ├── stacking.py           # stacking niveau 2
│   ├── blending.py           # blend pondéré
│   └── gs_utils.py           # utilitaires partagés
├── competitions/             # une sous-cartographie par compétition
│   ├── smartphone-addiction/ # Playground S6E8 (1ère mission)
│   │   ├── README.md         # rapport complet
│   │   ├── code/             # scripts et notebooks
│   │   └── results/          # OOF, submissions, scores
│   ├── world-cup-2026-goal-prediction/ # Zindi practice reconstruction
│   │   ├── README.md         # méthode et score privé pratique
│   │   ├── code/             # pipeline causal + Optuna
│   │   └── results/          # audits et expériences rejetées
│   └── ...
├── experiments.csv            # registre des expériences et niveaux d'assistance
├── scripts/
│   └── verify_repository.py   # syntaxe, notebooks, scan de secrets
├── .github/workflows/         # CI légère
├── ROADMAP.md                 # trajectoire autonome vers top 10 %, top 5 %, puis prize
├── LICENSE                    # MIT
└── README.md                  # résultats vérifiés et synthèse
```

## 🔍 Politique de preuve

- **Autonome / from-scratch** : modèles entraînés par nous, sans prédictions de notebooks publics.
- **Assisté** : utilise des OOF/test predictions ou autres artefacts publics ; présenté séparément.
- Un faible écart au meilleur score ne vaut pas un rang : tout rang est vérifié sur un snapshot.
- Un résultat time series n'est publié qu'avec métrique officielle et features causales auditées.

## 🏆 Compétitions
| Compétition | Type | Score | Leçons clés |
|-------------|------|-------|-------------|
| Predicting Smartphone Addiction (S6E8) | Tabulaire | **0.97057 assisté** (277/1 724) ; **0.96970 autonome** (from-scratch-code-assisted) ; **Sprint 5: 0.93799 OOF NO-GO** | Exact-value TE + RealMLP + TabM + CatBoost + imputation prédictive, blend validé par portes ; Sprint 5 checkpointé mais non soumis |
| World Cup 2026 Goal Prediction Challenge (Zindi, practice) | Tabulaire / tournoi | **0.54358 private practice** | Features historiques causales + matchs/Elo + Optuna + ranking global avec quotas ; données brutes exclues, compétition fermée |
| Predicting Student Health Risk (Kaggle S6E7) | Tabulaire multiclass | **0.95029 private** | Balanced accuracy corrigée + target encoding fold-safe + CatBoost/LightGBM + MLP GPU RunPod ; probing public exclu |
| Store Sales Forecasting (prototype) | Time series | **Non validé** | Audit: métrique officielle RMSLE; ancien SMAPE + features fuyantes à reconstruire |

## 🚀 Roadmap

Plan détaillé, portes de passage et seuils chiffrés : [ROADMAP.md](ROADMAP.md).

- [x] Compétition #1 : Smartphone Addiction (**0.97057 assisté**, **0.96970 autonome**)
- [x] Sprint prospectif : folds gelés, exact-value TE, RealMLP GPU, TabM, CatBoost, imputation prédictive (Sprint 4), holdout historique ouvert une fois
- [ ] Reconstruire Store Sales avec RMSLE et features temporelles causales
- [ ] Obtenir un premier résultat **from-scratch** top 10 % sur une Playground
- [ ] Obtenir un résultat **from-scratch** top 5 % sur une seconde compétition
- [ ] Entrer ensuite sur une compétition à prize tabulaire ou time series
