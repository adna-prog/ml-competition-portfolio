# ML Competition Portfolio

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
│   └── ...
└── LICENSE
```

## 🏆 Compétitions
| Compétition | Type | Score | Rang | Leçons clés |
|-------------|------|-------|------|-------------|
| Predicting Smartphone Addiction (S6E8) | Tabulaire | **0.97057** | ~top | **Stacking massif + poly**; diversité > perfection |

## 🚀 Roadmap
- [x] Compétition #1 : Smartphone Addiction (**0.97057**, quasi top-1)
- [x] Maîtriser le stacking massif (meta-features publiques + polynomiales + GPU)
- [ ] Compétition #2 gratuite (Playground time series ou multi-class)
- [ ] Compétition à prize (tabulaire ou time series)
