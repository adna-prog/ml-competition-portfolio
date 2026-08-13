# Stacking de sorties publiques + features polynomiales — étude S6E8

Technique validée sur **Playground S6E8** (Predicting Smartphone Addiction) :
baseline 0.96608 → **0.97057 public** (écart numérique de 0.00067 au meilleur score public observé ; rang public vérifié 277/1 724 au 13 août 2026).

## Principe
Au lieu d'entraîner nos propres modèles, on **réutilise les OOF + test preds des notebooks publics**
comme **meta-features**, on ajoute des **interactions polynomiales**, et on entraîne un **XGBoost lent sur GPU**.

## Pipeline
1. **Trouver les notebooks publics** : `kaggle kernels list --competition <comp>`
2. **Télécharger leurs OOF + test preds** : `kaggle kernels output <user>/<slug>`
3. **Empaqueter en dataset Kaggle unifié** : `kaggle datasets version -p ./meta_dataset`
4. **Notebook GPU** qui :
   - charge les meta-features (15 modèles diversifiés)
   - applique `PolynomialFeatures(degree=2, interaction_only=True)`
   - entraîne `XGBClassifier(lr=0.01, n_estimators=20000, max_depth=8, colsample=0.5, max_bin=2000, device='cuda')`

## Résultats S6E8
| Version | Meta-features | CV | Public |
|---------|--------------|-----|--------|
| baseline | 0 | 0.96460 | 0.96608 |
| stacking v1 | 4 | 0.96853 | 0.96978 |
| **stacking v2** | **15** | **0.96944** | **0.97057** 🏆 |
| stacking v3 | 17 | 0.96947 | 0.97053 |
| stacking top | 14 | 0.96947 | 0.97053 |

## Leçons clés
- **La diversité est le levier n°1** : GBDT + NN (RealMLP/TabM/ResNet/TabNet/FT-Transformer) + AutoML + lookup transformer.
- **15 meta-features est optimal** : plus de features corrélées (17) dégrade légèrement.
- **Ne pas retirer les features "faibles"** : même resnet (AUC seule 0.9661) apporte de la diversité utile via les interactions polynomiales.
- **Les notebooks top-LB ne publient souvent que leur submission (pas d'OOF)** → inutilisables en meta-features propres.

## Réutilisabilité
- Script `build_meta_dataset.py` : assemble les OOF/test preds en dataset unifié.
- Notebook `notebook_gagnant_reproduction.ipynb` : template du stacking GPU.
- Kernel metadata : référence competition + dataset de meta-features.

Fichiers de référence :
- `code/notebook_gagnant_reproduction.ipynb`
- `code/build_meta_dataset.py`
- `code/kernel-metadata.json`
