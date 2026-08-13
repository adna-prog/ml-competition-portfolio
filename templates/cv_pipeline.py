"""Pipeline CV leakage-free pour modèles GBDT. Génère OOF + test preds.
Adaptez NUM_COLS/CAT_COLS/target à chaque compétition.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

NUM_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']
TARGET = 'addicted_label'
RANDOM_STATE = 42
N_FOLDS = 5


def get_Xy(train):
    y = train[TARGET].values
    X = train[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    return X, y


def cv_fit_predict(factory, X, y, Xtest, n_folds=N_FOLDS):
    """factory(X_tr, y_tr, X_va, y_va) -> model. Retourne (oof_auc, oof, test_pred)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(Xtest))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = factory(X.iloc[tr], y[tr], X.iloc[va], y[va])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_pred += m.predict_proba(Xtest)[:, 1] / n_folds
        print(f"  fold {fold+1}: AUC={roc_auc_score(y[va], oof[va]):.5f}")
    oof_auc = roc_auc_score(y, oof)
    print(f">>> OOB AUC = {oof_auc:.5f}")
    return oof_auc, oof, test_pred


# ---- Factories de modèles (adaptez les params) ----
def lgb_factory(X_tr, y_tr, X_va, y_va, params=None):
    import lightgbm as lgb
    p = dict(learning_rate=0.1, n_estimators=1500, num_leaves=31, min_child_samples=20,
             random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    if params:
        p.update(params)
    m = lgb.LGBMClassifier(**p)
    return m.fit(X_tr, y_tr, categorical_feature=CAT_COLS,
                 eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])


def xgb_factory(X_tr, y_tr, X_va, y_va, params=None):
    import xgboost as xgb
    p = dict(learning_rate=0.1, n_estimators=1500, max_depth=6,
             tree_method='hist', enable_categorical=True, random_state=RANDOM_STATE, n_jobs=-1)
    if params:
        p.update(params)
    m = xgb.XGBClassifier(**p)
    return m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
