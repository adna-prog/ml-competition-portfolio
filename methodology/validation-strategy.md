# Stratégie de validation (leakage-free)

Comment valider correctement en compétition — le fondement de tout.

## Principe d'or
> Toute preprocessing (target encoding, scaling, imputation) doit être **fit DANS chaque fold**
> sur le train uniquement, jamais sur le full train avant le split.

## Choisir la bonne CV
| Type de test | CV à utiliser |
|--------------|---------------|
| Échantillon aléatoire | `StratifiedKFold(n_splits=5)` (classif) / `KFold` (régression) |
| Temporel | `TimeSeriesSplit` — jamais de shuffle |
| Groupé | `GroupKFold` (respecter les groupes) |
| Distribution shift | CV mimant le shift, ou adversarial validation |

## Pattern OOF (out-of-fold)
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in skf.split(X, y):
    # fit preprocessing sur X[tr] uniquement
    m = model.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
score = roc_auc_score(y, oof)  # OOB score = estimation fiable
```

## Signes de leakage / validation cassée
- CV très élevé mais LB mauvais → fuite dans le fold.
- AUC CV >> AUC public → quelque chose ne va pas.

## Règles
1. Toujours 1 seed fixe pour reproductibilité.
2. Garder les OOF de tous les modèles (pour blend/stack).
3. Le stacking se valide avec une **CV externe/nested** ou un holdout final intact. Re-CV des OOF
   avec les mêmes folds peut être optimiste : les labels du fold externe peuvent avoir influencé
   les modèles qui ont produit les meta-features du meta-train.
4. Reporter moyenne, écart-type et pire fold — jamais seulement le meilleur fold.
5. Distinguer clairement score autonome, score assisté par artefacts publics, public LB et private LB.
