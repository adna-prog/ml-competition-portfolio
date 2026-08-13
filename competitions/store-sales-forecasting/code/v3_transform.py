"""Store Sales v3 : tester la transformation (log1p vs brute) + features temporelles riches.
Objectif : descendre sous le 0.5117 de la v2. Compare log1p vs raw sur la même validation.
"""
import pandas as pd, numpy as np, time
import lightgbm as lgb
import xgboost as xgb

base = 'data'
t0 = time.time()
train = pd.read_csv(f'{base}/train.csv', parse_dates=['date'])
test = pd.read_csv(f'{base}/test.csv', parse_dates=['date'])
stores = pd.read_csv(f'{base}/stores.csv')
oil = pd.read_csv(f'{base}/oil.csv', parse_dates=['date'])
holidays = pd.read_csv(f'{base}/holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv(f'{base}/transactions.csv', parse_dates=['date'])

# Covariables
oil = oil.sort_values('date').ffill()
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')
trans_agg = transactions.groupby(['date','store_nbr'])['transactions'].sum().reset_index()
train = train.merge(trans_agg, on=['date','store_nbr'], how='left')
test = test.merge(trans_agg, on=['date','store_nbr'], how='left')
train = train.merge(stores[['store_nbr','cluster']], on='store_nbr', how='left')
test = test.merge(stores[['store_nbr','cluster']], on='store_nbr', how='left')
hol_agg = holidays.groupby('date').agg(
    n_holidays=('type','count'),
    is_holiday=('type', lambda x: int(any('Holiday' in str(t) for t in x))),
    is_workday=('type', lambda x: int(any('Work Day' in str(t) for t in x)))
).reset_index()
train = train.merge(hol_agg, on='date', how='left')
test = test.merge(hol_agg, on='date', how='left')

def add_calendar(df):
    df = df.copy()
    d = df['date'].dt
    df['year'] = d.year; df['month'] = d.month; df['day'] = d.day
    df['dayofweek'] = d.dayofweek; df['dayofyear'] = d.dayofyear
    df['week'] = d.isocalendar().week.astype(int)
    df['is_weekend'] = (d.dayofweek >= 5).astype(int)
    df['is_month_start'] = (d.day <= 7).astype(int)
    df['is_month_end'] = (d.day >= 24).astype(int)
    return df

train = add_calendar(train); test = add_calendar(test)

# Lag + rolling sur tout (train+test)
all_dates = pd.concat([train.assign(sales=train['sales']), test.assign(sales=np.nan)], ignore_index=True)
all_dates = all_dates.sort_values(['store_nbr','family','date']).reset_index(drop=True)
key = ['store_nbr','family']
grp = all_dates.groupby(key)['sales']

lags = [1,2,3,7,14,28,56,112,168,224,280,336,364]
for lag in lags:
    all_dates[f'lag_{lag}'] = grp.shift(lag)
for w in [7,14,28,56,84,168,364]:
    all_dates[f'rmean_{w}'] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
all_dates['rstd_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).std())
all_dates['rmax_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).max())
all_dates['store_mean'] = all_dates.groupby('store_nbr')['sales'].transform('mean')
all_dates['family_mean'] = all_dates.groupby('family')['sales'].transform('mean')
all_dates['store_family_mean'] = all_dates.groupby(key)['sales'].transform('mean')
# ratios de lag
for l1, l2 in [(7,14),(14,28),(28,56),(56,112)]:
    all_dates[f'lag_ratio_{l1}_{l2}'] = all_dates[f'lag_{l1}'] / (all_dates[f'lag_{l2}'] + 1)
all_dates['cluster'] = all_dates['cluster'].astype('category')

train = all_dates[all_dates['sales'].notna()].copy()
test = all_dates[all_dates['sales'].isna()].copy()
test['id'] = test['id'].astype(int)
print(f'features faites {time.time()-t0:.0f}s')

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1e-9, denom))

# Split temporel
last = train['date'].max()
val_dates = pd.date_range(last - pd.Timedelta(days=15), last)
val = train[train['date'].isin(val_dates)]
tr = train[~train['date'].isin(val_dates)]
exclude = ['id','date','sales','store_nbr','family']
feats = [c for c in tr.columns if c not in exclude]
print(f'features: {len(feats)}')

# TEST A : prédire log1p (comme v2) avec LGBM
trA = tr.copy(); valA = val.copy()
trA['target'] = np.log1p(trA['sales']); valA['target'] = np.log1p(valA['sales'])
m = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=800, num_leaves=63,
                      colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1, verbosity=-1)
t1=time.time()
m.fit(trA[feats], trA['target'])
p_log = np.expm1(m.predict(valA[feats]))
s_log = smape(valA['sales'].values, p_log)
print(f'[A] LGBM log1p: SMAPE {s_log:.4f} ({time.time()-t1:.0f}s)')

# TEST B : prédire les ventes brutes
trB = tr.copy(); valB = val.copy()
m2 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=800, num_leaves=63,
                       colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1, verbosity=-1)
t2=time.time()
m2.fit(trB[feats], trB['sales'])
p_raw = m2.predict(valB[feats])
p_raw = np.clip(p_raw, 0, None)
s_raw = smape(valB['sales'].values, p_raw)
print(f'[B] LGBM brut: SMAPE {s_raw:.4f} ({time.time()-t2:.0f}s)')

np.save('v3_log_pred.npy', p_log)
np.save('v3_raw_pred.npy', p_raw)
np.save('v3_val_true.npy', valA['sales'].values)
print('terminé. | référence v2: 0.5117')
