"""Crée le dataset étendu de meta-features (15 modèles) depuis les outputs des notebooks publics.
Sortie: meta_train.csv (id + 15 meta OOF) et meta_test.csv (id + 15 meta test preds).
"""
import pandas as pd, numpy as np, os

BASE = 'meta_assets'
OUT = 'meta_dataset'

# source -> (oof_path, oof_col, test_path, test_col)
SOURCES = {
    # 4 modèles initiaux
    'cb_omid': ('omidbaghchehsaraei_catboost-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
                'omidbaghchehsaraei_catboost-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'xgb_omid': ('omidbaghchehsaraei_xgboost-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
                 'omidbaghchehsaraei_xgboost-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'cb_don': ('donmarch14_s6e8-catboost/oof_preds.csv', 'oof_pred',
               'donmarch14_s6e8-catboost/test_preds.csv', 'test_pred'),
    'lgb_don': ('donmarch14_s6e8-lgbm/lgbm_oof.csv', 'oof_pred',
                'donmarch14_s6e8-lgbm/lgbm_test_preds.csv', 'test_pred'),
    # 7 NN + automl omid
    'realmlp': ('omid_realmlp-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
                'omid_realmlp-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'tabm': ('omid_tabm-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
             'omid_tabm-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'resnet': ('omid_resnet-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
               'omid_resnet-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'tabnet': ('omid_tabnet-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
               'omid_tabnet-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'ft_transformer': ('omid_ft-transformer-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
                       'omid_ft-transformer-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'flaml_xgb': ('omid_flaml-xgboost-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
                  'omid_flaml-xgboost-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'flaml_lgb': ('omid_flaml-lgbm-for-predicting-smartphone-addiction/oof.csv', 'oof_pred',
                  'omid_flaml-lgbm-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
    'hill_climb': ('omid_hill-climbing/oof.csv', 'addicted_label',
                   'omid_hill-climbing/submission.csv', 'addicted_label'),
    # 3 modèles lookup transformer (tamerlan)
    'lt_catboost': ('tamerlan_lookup/oof_catboost.npy', None,
                    'tamerlan_lookup/test_catboost.npy', None),
    'lt_lightgbm': ('tamerlan_lookup/oof_lightgbm.npy', None,
                    'tamerlan_lookup/test_lightgbm.npy', None),
    'lt_lookup': ('tamerlan_lookup/oof_lookup_transformer.npy', None,
                  'tamerlan_lookup/test_lookup_transformer.npy', None),
    # 1 L2-stack public (ravi20076) — OOF + test preds
    'l2stack': ('ravi20076_playgrounds6e8-public-l2stack-v1/OOF_Preds_L2StackV1_1.parquet', 'L2Stack1R',
                'ravi20076_playgrounds6e8-public-l2stack-v1/Mdl_Preds_L2StackV1_1.parquet', 'L2Stack1R'),
    # RealMLP (zhenruiweng) — OOF + test preds
    'realmlp_z': ('zhenruiweng_realmlp-for-predicting-smartphone-addiction/oof_preds.csv', 'oof_pred',
                  'zhenruiweng_realmlp-for-predicting-smartphone-addiction/submission.csv', 'addicted_label'),
}

os.makedirs(OUT, exist_ok=True)

# init avec id
train_id = pd.read_csv(f'{BASE}/donmarch14_s6e8-catboost/oof_preds.csv')[['id']]
test_id = pd.read_csv(f'{BASE}/donmarch14_s6e8-catboost/test_preds.csv')[['id']]
meta_train = train_id.copy()
meta_test = test_id.copy()

for name, (op, oc, tp, tc) in SOURCES.items():
    if op.endswith('.npy'):
        o = np.load(f'{BASE}/{op}')
        t = np.load(f'{BASE}/{tp}')
        meta_train[f'pred_{name}'] = o
        meta_test[f'pred_{name}'] = t
    else:
        odf = pd.read_csv(f'{BASE}/{op}') if op.endswith('.csv') else pd.read_parquet(f'{BASE}/{op}')
        tdf = pd.read_csv(f'{BASE}/{tp}') if tp.endswith('.csv') else pd.read_parquet(f'{BASE}/{tp}')
        meta_train[f'pred_{name}'] = odf[oc].values
        meta_test[f'pred_{name}'] = tdf[tc].values
    print(f'{name}: OK')

meta_train.to_csv(f'{OUT}/meta_train.csv', index=False)
meta_test.to_csv(f'{OUT}/meta_test.csv', index=False)
print(f'\nÉcrit: meta_train {meta_train.shape}, meta_test {meta_test.shape}')
print('meta features:', [c for c in meta_train.columns if c != 'id'])
