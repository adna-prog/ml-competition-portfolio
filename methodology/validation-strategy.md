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
3. Le stacking se valide en re-CV sur les OOF (pas sur les prédictions train).
