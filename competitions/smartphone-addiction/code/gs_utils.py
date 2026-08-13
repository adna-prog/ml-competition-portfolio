"""Utilities partagées pour le projet Kaggle 'Predicting Smartphone Addiction'.
Fournit : chargement, features, métrique AUC, CV stratifié, et validation croisée pour les modèles GBDT.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "."
RANDOM_STATE = 42
N_FOLDS = 5

NUM_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, test


def get_Xy(train):
    y = train['addicted_label'].values
    X = train[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    return X, y


def get_cv(n_splits=N_FOLDS, random_state=RANDOM_STATE):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cv_score(model_factory, X, y, n_splits=N_FOLDS, verbose=True, **factory_kwargs):
    """Cross-validation stratifiée; model_factory(X_tr, y_tr, X_va, y_va, **kwargs) -> fitted model.
    factory_kwargs (ex: params={...}) sont transmis à la factory.
    Retourne (oob_auc, oof_pred, models).
    """
    skf = get_cv(n_splits)
    oof = np.zeros(len(y))
    models = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = model_factory(X_tr, y_tr, X_va, y_va, **factory_kwargs)
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        models.append(m)
        auc = roc_auc_score(y_va, oof[va_idx])
        if verbose:
            print(f"  fold {fold+1}: AUC = {auc:.5f}")
    oob_auc = roc_auc_score(y, oof)
    if verbose:
        print(f"  >>> OOB AUC = {oob_auc:.5f}  (mean of folds)")
    return oob_auc, oof, models


def make_submission(test_ids, preds, out="submission.csv"):
    sub = pd.DataFrame({'id': test_ids, 'addicted_label': preds})
    sub.to_csv(out, index=False)
    print(f"submission écrite -> {out} ({len(sub)} lignes)")


# ---- Factories de modèles ----
def lgb_factory(X_tr, y_tr, X_va=None, y_va=None, params=None):
    import lightgbm as lgb
    default = dict(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=20, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    if params:
        default.update(params)
    m = lgb.LGBMClassifier(**default)
    if X_va is not None and y_va is not None:
        return m.fit(X_tr, y_tr, eval_metric='auc',
                     categorical_feature=CAT_COLS,
                     eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.fit(X_tr, y_tr, eval_metric='auc', categorical_feature=CAT_COLS)


def xgb_factory(X_tr, y_tr, X_va=None, y_va=None, params=None):
    import xgboost as xgb
    default = dict(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method='hist', enable_categorical=True,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc',
    )
    if params:
        default.update(params)
    m = xgb.XGBClassifier(**default)
    if X_va is not None and y_va is not None:
        return m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.fit(X_tr, y_tr)


def cat_factory(X_tr, y_tr, X_va=None, y_va=None, params=None):
    from catboost import CatBoostClassifier
    default = dict(iterations=2000, learning_rate=0.05, depth=6,
                   random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False)
    if params:
        default.update(params)
    # CatBoost n'accepte pas NaN dans les catégories -> remplacer par 'missing'
    # (astype('object').fillna est robuste au dtype 'category' hérité de get_Xy)
    def _clean(df):
        df = df.copy()
        for c in CAT_COLS:
            df[c] = df[c].astype('object').fillna('missing')
        return df
    X_tr = _clean(X_tr)
    m = CatBoostClassifier(**default)
    if X_va is not None and y_va is not None:
        X_va = _clean(X_va)
        m.fit(X_tr, y_tr, cat_features=CAT_COLS,
              eval_set=(X_va, y_va), early_stopping_rounds=100)
    else:
        m.fit(X_tr, y_tr, cat_features=CAT_COLS)

    # Wrapper : nettoie aussi en prédiction
    class _CatWrap:
        def __init__(self, inner, clean):
            self._m = inner
            self._clean = clean
        def predict_proba(self, X):
            return self._m.predict_proba(self._clean(X))
    return _CatWrap(m, _clean)


# ---- Feature engineering (sera enrichi) ----
def add_features(df):
    """Features dérivées. df: DataFrame avec les colonnes d'origine. Retourne un copy augmenté."""
    df = df.copy()
    # Ratio temps d'écran consacré aux réseaux sociaux
    df['sm_ratio'] = df['social_media_hours'] / df['daily_screen_time_hours'].replace(0, np.nan)
    # Ratio week-end vs semaine
    df['weekend_ratio'] = df['weekend_screen_time'] / df['daily_screen_time_hours'].replace(0, np.nan)
    # Réseaux + gaming
    df['sm_gaming'] = df['social_media_hours'] + df['gaming_hours']
    # Temps non expliqué (screen - social - gaming - work)
    df['unexplained_screen'] = df['daily_screen_time_hours'] - df['social_media_hours'] - df['gaming_hours']
    # Notification densité
    df['notif_per_open'] = df['notifications_per_day'] / df['app_opens_per_day'].replace(0, np.nan)
    # Somme des heures
    df['total_hours'] = df[['daily_screen_time_hours', 'work_study_hours', 'sleep_hours']].sum(axis=1)
    return df
