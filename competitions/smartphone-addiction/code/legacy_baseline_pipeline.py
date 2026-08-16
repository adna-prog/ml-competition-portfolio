"""Pipeline LÉGATAIRE (legacy) — solution XGBoost + LightGBM sur features brutes.

ATTENTION : ce script NE produit PAS le résultat autonome courant (0.96970).
Il correspond au benchmark initial (submission #8, 0.96630) : XGBoost + LightGBM
sur features brutes, blend à poids fixe XGB=0.72. Le pipeline courant (Sprints
1-4 : exact-TE + RealMLP + TabM + CatBoost + imputation prédictive) est documenté
dans results/RESULTS.md. Conservé pour référence historique uniquement.
"""
import time, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score
import gs_utils as U

RANDOM_STATE = 42
N_FOLDS = 5
BLEND_W_XGB = 0.72   # poids XGBoost dans le blend (trouvé sur OOF)

t0 = time.time()
print('Chargement...', flush=True)
train, test = U.load_data()
X, y = U.get_Xy(train)
Xtest = test[U.NUM_COLS + U.CAT_COLS].copy()
for c in U.CAT_COLS:
    Xtest[c] = Xtest[c].astype('category')

skf = U.get_cv(N_FOLDS)
oof_xgb = np.zeros(len(y)); oof_lgb = np.zeros(len(y))
test_xgb = np.zeros(len(test)); test_lgb = np.zeros(len(test))

xgb_params = dict(learning_rate=0.05, n_estimators=3000, max_depth=6, min_child_weight=5)
lgb_params = dict(learning_rate=0.05, n_estimators=3000, num_leaves=31, min_child_samples=20)

for fold, (tr, va) in enumerate(skf.split(X, y)):
    print(f'\n=== Fold {fold+1} ===', flush=True)
    # XGBoost
    mx = U.xgb_factory(X.iloc[tr], y[tr], X.iloc[va], y[va], params=xgb_params)
    oof_xgb[va] = mx.predict_proba(X.iloc[va])[:, 1]
    test_xgb += mx.predict_proba(Xtest)[:, 1] / N_FOLDS
    print(f'  XGB fold AUC={roc_auc_score(y[va], oof_xgb[va]):.5f}', flush=True)
    # LightGBM
    ml = U.lgb_factory(X.iloc[tr], y[tr], X.iloc[va], y[va], params=lgb_params)
    oof_lgb[va] = ml.predict_proba(X.iloc[va])[:, 1]
    test_lgb += ml.predict_proba(Xtest)[:, 1] / N_FOLDS
    print(f'  LGB fold AUC={roc_auc_score(y[va], oof_lgb[va]):.5f}', flush=True)

oof_blend = BLEND_W_XGB * oof_xgb + (1 - BLEND_W_XGB) * oof_lgb
auc_oof = roc_auc_score(y, oof_blend)
print(f'\nOOB AUC blend = {auc_oof:.5f}', flush=True)

test_blend = BLEND_W_XGB * test_xgb + (1 - BLEND_W_XGB) * test_lgb
U.make_submission(test['id'].values, test_blend, 'submission.csv')

np.save('oof_final_blend.npy', oof_blend)
np.save('test_pred_final_blend.npy', test_blend)
json.dump({'oof_auc': auc_oof, 'w_xgb': BLEND_W_XGB},
          open('final_result.json', 'w'), indent=2)
print(f'\nTerminé en {time.time()-t0:.0f}s. Fichiers écrits.', flush=True)
