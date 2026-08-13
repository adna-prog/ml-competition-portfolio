# Techniques tabulaires — notes validées

Résultats expérimentaux issus de la compétition Smartphone Addiction (S6E8).

## Modèles (raw features, ROC AUC)
| Modèle | Config | OOB AUC |
|--------|--------|---------|
| LightGBM | lr=0.05, es=100 | 0.96346 |
| XGBoost | lr=0.05, depth=6 | 0.96440 |
| CatBoost | lr=0.05, depth=6 | ~0.9615 |
| **Blend XGB(0.72)+LGB** | — | **0.96460** |

## Ce qui marche / ne marche pas
| Technique | Résultat | Leçon |
|-----------|----------|-------|
| **Stacking L2 (XGB+LGB→LR)** | +0.003 | ✅ Le levier des gagnants |
| Seeds averaging | ~0 | Bruit leaderboard |
| Target Encoding (par fold) | −0.001 | ❌ catégories pauvres |
| FE ratios/sommes | −0.001 | ❌ bruit sur synthétique |
| Rank blending | 0 | ≈ proba blend |
| Pseudo-labeling simple | −0.0005 | ❌ modèle ré-apprend ses prédictions |

## Config d'itération rapide
- `lr=0.1, n_estimators=1500` → ~6x plus rapide pour explorer.
- Config full `lr=0.05` pour la submission finale.

## Pièges modèles
- LightGBM 4.x : catégories en dtype `category` (pas `str`) avec `categorical_feature`.
- CatBoost : NaN catégoriel → `astype('object').fillna('missing')`, et nettoyer aussi en prédiction.
- NaN : laisser pour LGB/XGB (gestion native), ne pas imputer aveuglément.
