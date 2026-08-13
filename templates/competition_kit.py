"""Kit de démarrage de compétition Kaggle — générique et réutilisable.
Branchez sur n'importe quelle compétition tabulaire en adaptant : TARGET, NUM_COLS, CAT_COLS,
le chargement, et les configs de modèles.

Pipeline :
1. Chargement + split (CV stratifiée, ou TimeSeriesSplit si temporel)
2. Baseline GBDT (XGB/LGB/Cat) rapide
3. Modèles diversifiés Niveau 1 (GBDT configs + features variées + option NN)
4. Stacking massif (OOF + PolynomialFeatures + XGBoost lent GPU) — LA technique gagnante
5. Submission

Usage (local ou notebook Kaggle) :
  python competition_kit.py   # ou coller les cellules dans un notebook GPU
"""
import os, warnings, gc, time, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

# ============ CONFIG (À ADAPTER PAR COMPÉTITION) ============
SEED = 42
N_FOLDS = 5
TARGET = 'addicted_label'          # à adapter
ID_COL = 'id'
# Colonnes numériques / catégorielles (à adapter)
# ===========================================================

def load_data():
    """À adapter : chemin + noms de fichiers."""
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test


def get_cv(n_splits=N_FOLDS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def cv_oof(factory, X, y, X_test, n_splits=N_FOLDS, **kwargs):
    """CV leakage-free → (oof_auc, oof, test_pred). factory(X_tr,y_tr,X_va,y_va)->model."""
    skf = get_cv(n_splits)
    oof = np.zeros(len(y)); tp = np.zeros(len(X_test))
    for tr, va in skf.split(X, y):
        m = factory(X.iloc[tr], y[tr], X.iloc[va], y[va], **kwargs)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tp += m.predict_proba(X_test)[:, 1] / n_splits
    return roc_auc_score(y, oof), oof, tp


# ============ Factories de modèles Niveau 1 (à diversifier) ============
def xgb_factory(X_tr, y_tr, X_va, y_va, params=None):
    import xgboost as xgb
    p = dict(learning_rate=0.03, n_estimators=3000, max_depth=6, subsample=0.8,
             colsample_bytree=0.5, tree_method='hist', enable_categorical=True,
             random_state=SEED, n_jobs=-1, eval_metric='auc')
    if params: p.update(params)
    m = xgb.XGBClassifier(**p)
    return m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)


def lgb_factory(X_tr, y_tr, X_va, y_va, params=None, cat_cols=None):
    import lightgbm as lgb
    p = dict(learning_rate=0.03, n_estimators=3000, num_leaves=32, min_child_samples=30,
             colsample_bytree=0.7, random_state=SEED, n_jobs=-1, verbosity=-1)
    if params: p.update(params)
    m = lgb.LGBMClassifier(**p)
    return m.fit(X_tr, y_tr, categorical_feature=cat_cols or [],
                 eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])


def cat_factory(X_tr, y_tr, X_va, y_va, params=None, cat_cols=None):
    from catboost import CatBoostClassifier
    p = dict(learning_rate=0.03, iterations=3000, depth=6, random_seed=SEED,
             verbose=0, allow_writing_files=False)
    if params: p.update(params)
    # CatBoost : NaN catégoriel -> 'missing'
    def clean(df):
        df = df.copy()
        for c in (cat_cols or []):
            df[c] = df[c].astype('object').fillna('missing')
        return df
    m = CatBoostClassifier(**p)
    return m.fit(clean(X_tr), y_tr, cat_features=cat_cols or [],
                 eval_set=(clean(X_va), y_va), early_stopping_rounds=100)


# ============ Stacking massif (LA technique gagnante) ============
def stack_massif(OOF, TEST_PRED, y, n_splits=N_FOLDS):
    """OOF/TEST des modèles N1 → PolynomialFeatures + XGBoost lent GPU → oof/test stack."""
    import xgboost as xgb
    names = list(OOF.keys())
    Xm = np.column_stack([OOF[n] for n in names])
    Xm_test = np.column_stack([TEST_PRED[n] for n in names])
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    Xp = poly.fit_transform(Xm); Xp_test = poly.transform(Xm_test)
    print(f'Meta-features: {Xm.shape[1]} | après poly: {Xp.shape[1]}')

    params = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist',
                  device='cuda' if _cuda() else 'cpu', enable_categorical=True,
                  learning_rate=0.01, max_depth=8, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0, n_estimators=20000,
                  random_state=SEED, max_bin=2000)
    skf = get_cv(n_splits)
    oof = np.zeros(len(y)); tp = np.zeros(len(Xp_test)); fa = []
    for tr, va in skf.split(Xp, y):
        m = xgb.XGBClassifier(**params, missing=np.nan, early_stopping_rounds=200)
        m.fit(Xp[tr], y[tr], eval_set=[(Xp[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(Xp[va])[:, 1]
        tp += m.predict_proba(Xp_test)[:, 1] / n_splits
        fa.append(roc_auc_score(y[va], oof[va])); gc.collect()
    print(f'Stacking CV: {np.mean(fa):.5f} | OOB: {roc_auc_score(y, oof):.5f}')
    return oof, tp


def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


def make_submission(test_ids, preds, out='submission.csv'):
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: preds})
    sub.to_csv(out, index=False)
    print(f'→ submission {out} ({len(sub)} lignes)')


# ============ Exemple d'usage (décommenter et adapter) ============
if __name__ == '__main__':
    train, test = load_data()
    y = train[TARGET].values
    NUM = [c for c in train.columns if c not in [ID_COL, TARGET]
           and train[c].dtype in ['int64','float64']]
    CAT = [c for c in train.columns if c not in [ID_COL, TARGET]
           and train[c].dtype == 'object']
    X = train[NUM+CAT].copy(); X_test = test[NUM+CAT].copy()
    for c in CAT:
        X[c] = X[c].astype('category'); X_test[c] = X_test[c].astype('category')

    # 1) Baseline rapide
    auc, oof, tp = cv_oof(xgb_factory, X, y, X_test)
    print(f'Baseline XGB: {auc:.5f}')

    # 2) Stacking massif (si on a plusieurs modèles N1)
    # OOF = {...}; TEST_PRED = {...}
    # oof_stack, tp_stack = stack_massif(OOF, TEST_PRED, y)
    # make_submission(test[ID_COL].values, tp_stack)
