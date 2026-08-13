"""Test des techniques des gagnants en config RAPIDE (local, ~3 min/cv):
1. Target Encoding par fold (fit DANS le fold, leakage-safe) sur les colonnes catégorielles.
2. Stacking niveau 2 (LightGBM stacker sur les OOF des modèles L1).
Compare vs baseline raw LightGBM.
"""
import time, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import gs_utils as U

train, test = U.load_data()
X_raw, y = U.get_Xy(train)
Xtest_raw = test[U.NUM_COLS + U.CAT_COLS].copy()
for c in U.CAT_COLS:
    Xtest_raw[c] = Xtest_raw[c].astype('category')

FAST = dict(learning_rate=0.1, n_estimators=1500, num_leaves=31, min_child_samples=20)
skf = U.get_cv(5)
results = {}

# --- 1) Target Encoding par fold ---
print('=== BASELINE raw (LGBM fast) ===', flush=True)
t0 = time.time()
auc0, oof0, _ = U.cv_score(U.lgb_factory, X_raw, y, n_splits=5, params=dict(FAST))
results['baseline'] = round(auc0, 5)
print(f'  baseline: {auc0:.5f} ({time.time()-t0:.0f}s)', flush=True)

# --- 2) TE par fold : construire X avec TE features, CV ---
print('\n=== LGBM + Target Encoding (par fold) ===', flush=True)
t0 = time.time()
oof_te = np.zeros(len(y)); test_te = np.zeros(len(test))
for tr, va in skf.split(X_raw, y):
    X_tr = X_raw.iloc[tr]; X_va = X_raw.iloc[va]; y_tr = y[tr]; y_va = y[va]
    # fit TE sur le fold train (mean target par catégorie)
    te_feats = {}
    for c in U.CAT_COLS:
        d = pd.DataFrame({'cat': X_tr[c].values, 'y': y_tr})
        te_map = d.groupby('cat')['y'].mean()
        te_feats[c] = te_map
    def apply_te(X):
        X = X.copy()
        for c, m in te_feats.items():
            X[f'{c}_te'] = X[c].values.astype(object)
            X[f'{c}_te'] = pd.Series(X[c].values).map(m).values
        return X
    X_tr_e = apply_te(X_tr); X_va_e = apply_te(X_va); Xtest_e = apply_te(Xtest_raw)
    m = U.lgb_factory(X_tr_e, y_tr, X_va_e, y_va, params=dict(FAST))
    oof_te[va] = m.predict_proba(X_va_e)[:, 1]
    test_te += m.predict_proba(Xtest_e)[:, 1] / 5
auc_te = roc_auc_score(y, oof_te)
results['te'] = round(auc_te, 5)
print(f'  +TE: {auc_te:.5f} (gain {auc_te-auc0:+.5f}) ({time.time()-t0:.0f}s)', flush=True)

# --- 3) Stacking niveau 2 : LGB stacker sur OOF des 2 modèles L1 (XGB+LGB) ---
print('\n=== Stacking L2 (LGB stacker sur OOF XGB+LGB) ===', flush=True)
t0 = time.time()
# OOF XGB et LGB (config rapide) en parallèle
oof_xgb = np.zeros(len(y)); test_xgb = np.zeros(len(test))
oof_lgb2 = np.zeros(len(y)); test_lgb2 = np.zeros(len(test))
for tr, va in skf.split(X_raw, y):
    mx = U.xgb_factory(X_raw.iloc[tr], y[tr], X_raw.iloc[va], y[va], params=dict(FAST))
    oof_xgb[va] = mx.predict_proba(X_raw.iloc[va])[:, 1]
    test_xgb += mx.predict_proba(Xtest_raw)[:, 1] / 5
    ml = U.lgb_factory(X_raw.iloc[tr], y[tr], X_raw.iloc[va], y[va], params=dict(FAST))
    oof_lgb2[va] = ml.predict_proba(X_raw.iloc[va])[:, 1]
    test_lgb2 += ml.predict_proba(Xtest_raw)[:, 1] / 5

# Stacker: X = [oof_xgb, oof_lgb2], target y — CV pour évaluer
Xm = np.column_stack([oof_xgb, oof_lgb2])
Xm_test = np.column_stack([test_xgb, test_lgb2])
from sklearn.linear_model import LogisticRegression
auc_stack = 0
for tr, va in skf.split(Xm, y):
    meta = LogisticRegression(max_iter=1000)
    meta.fit(Xm[tr], y[tr])
    auc_stack += roc_auc_score(y[va], meta.predict_proba(Xm[va])[:, 1]) / 5
results['stack_lr'] = round(auc_stack, 5)
print(f'  Stack LR (XGB+LGB OOF): {auc_stack:.5f} ({time.time()-t0:.0f}s)', flush=True)

json.dump(results, open('tech_results.json', 'w'), indent=2)
print('\nRESULTATS:', json.dumps(results, indent=2), flush=True)
print('DONE', flush=True)
