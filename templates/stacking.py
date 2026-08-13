"""Stacking niveau 2 : méta-modèle entraîné sur les OOF des modèles de base (Level 1).
La technique des top-3. Validé sur S6E8 : +0.003 vs baseline.

Usage :
    python stacking.py <oof1.npy> <oof2.npy> ...  (sauvegarde le stack et imprime l'AUC CV)
"""
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def stack_oof(oof_paths, y, n_folds=5, random_state=42):
    """Entraîne un méta-modèle (LogReg) sur les OOF des L1, évalue en CV imbriquée.
    Retourne (cv_auc, meta_model_fit, X_meta)."""
    X_meta = np.column_stack([np.load(p) for p in oof_paths])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    meta = LogisticRegression(max_iter=1000)
    auc = cross_val_score(meta, X_meta, y, cv=skf, scoring='roc_auc').mean()
    meta.fit(X_meta, y)  # fit sur full OOF pour prédire test
    return auc, meta, X_meta

if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv('train.csv')
    y = train['addicted_label'].values
    auc, meta, Xm = stack_oof(sys.argv[1:], y)
    print(f"Stack CV AUC: {auc:.5f}")
    np.save('oof_stack.npy', meta.predict_proba(Xm)[:, 1])
