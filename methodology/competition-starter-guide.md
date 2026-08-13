# Kit de démarrage de compétition — guide

Template générique pour attaquer n'importe quelle compétition tabulaire Kaggle rapidement.
Fichier : `templates/competition_kit.py`.

## Brancher sur une nouvelle compétition (5 min)
1. **Adapter la config** : `TARGET`, `ID_COL`, chemins des données.
2. **Charger les données** : modifier `load_data()`.
3. **Lancer le baseline** : `cv_oof(xgb_factory, X, y, X_test)`.
4. **Étendre la diversité** : ajouter des configs de modèles (depth/lr variés, features variées).
5. **Stacking massif** : `stack_massif(OOF, TEST_PRED, y)` → LA technique gagnante.

## Workflow éprouvé (S6E8 → 0.97057)
1. **CV leakage-free** : toute preprocessing fit DANS le fold (jamais sur full train).
2. **Baseline solide tôt** : XGB/LGB/Cat, on-the-board rapidement.
3. **Diversifier les modèles N1** : 3 familles GBDT × configs + features variées + NN.
4. **Stacking massif** : OOF des N1 + PolynomialFeatures(deg2) + XGBoost lent GPU.
   - ~15 meta-features optimal (plus = redondance, moins = sous-optimal).
   - Ne pas retirer les features "faibles" — elles apportent de la diversité.
5. **Confirmer sur leaderboard** avant de compter un gain.

## Pièges GBDT (rappels)
- **LightGBM 4.x** : colonnes catégorielles en dtype `category` (pas `str`) avec `categorical_feature`.
- **CatBoost** : NaN catégoriel → `astype('object').fillna('missing')` (aussi en prédiction).
- **NaN numériques** : laisser pour GBDT (gestion native), imputer pour NN.
- **NN** : scaling + embedding catégories + imputation (pipeline séparée).

## Exécution
- **Local** : itération rapide (config lr=0.1 pour explorer, lr=0.05 pour final).
- **Kaggle GPU** : stacking massif (gratuit, T4).
- **Pod payant** : seulement pour runs très lourds, avec accord explicite.

## Options par type de compétition
| Type | CV à utiliser | Stacking ? |
|------|--------------|------------|
| Classification tabulaire | StratifiedKFold | ✅ |
| Régression tabulaire | KFold | ✅ (métrique adaptée) |
| Time series | TimeSeriesSplit (pas de shuffle) | ✅ avec features de lag |
| Multi-class | StratifiedKFold + softmax | ✅ |
