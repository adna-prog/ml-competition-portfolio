"""Baseline enrichi Store Sales : lag riches + covariables (oil/holidays/transactions) + features groupe.
Compare au baseline brut (0.53 SMAPE). Objectif : descendre vers ~0.15-0.20.
"""
import pandas as pd, numpy as np, time
import lightgbm as lgb

base = 'data'
t0 = time.time()
train = pd.read_csv(f'{base}/train.csv', parse_dates=['date'])
test = pd.read_csv(f'{base}/test.csv', parse_dates=['date'])
stores = pd.read_csv(f'{base}/stores.csv')
oil = pd.read_csv(f'{base}/oil.csv', parse_dates=['date'])
holidays = pd.read_csv(f'{base}/holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv(f'{base}/transactions.csv', parse_dates=['date'])
sub = pd.read_csv(f'{base}/sample_submission.csv')
print(f'data chargé {time.time()-t0:.0f}s')

# ============ Covariables ============
# Oil : prix du pétrole (forward fill)
oil = oil.sort_values('date').ffill()
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

# Transactions : volume par store/jour
trans_agg = transactions.groupby(['date','store_nbr'])['transactions'].sum().reset_index()
train = train.merge(trans_agg, on=['date','store_nbr'], how='left')
test = test.merge(trans_agg, on=['date','store_nbr'], how='left')

# Stores : type, cluster
train = train.merge(stores[['store_nbr','cluster','city']], on='store_nbr', how='left')
test = test.merge(stores[['store_nbr','cluster','city']], on='store_nbr', how='left')

# Holidays : flag jour férié par date
hol_agg = holidays.groupby('date').agg(
    n_holidays=('type','count'),
    is_holiday=('type', lambda x: int(any('Holiday' in str(t) for t in x))),
    is_workday=('type', lambda x: int(any('Work Day' in str(t) for t in x)))
).reset_index()
train = train.merge(hol_agg, on='date', how='left')
test = test.merge(hol_agg, on='date', how='left')
print(f'covariables fusionnées {time.time()-t0:.0f}s')

# ============ Features calendrier ============
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

# ============ Lag + rolling par (store, family) ============
print(f'avant lag {time.time()-t0:.0f}s')
train = train.sort_values(['store_nbr','family','date']).reset_index(drop=True)
test = test.sort_values(['store_nbr','family','date']).reset_index(drop=True)
key = ['store_nbr','family']
train = train.sort_values(['date']).reset_index(drop=True)

# Lag : 1,2,3,7,14,21,28,35,42,56,84,112,140,168,196,224,252,280,308,336,364
lags = [1,2,3,7,14,28,56,112,168,224,280,336,364]
# Grouper le train par (store,family) et faire les shift
grp = train.groupby(key)['sales']
for lag in lags:
    train[f'lag_{lag}'] = grp.shift(lag)
# Rolling means
for w in [7,14,28,56,84,168,364]:
    train[f'rmean_{w}'] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
# rolling max / std
train['rstd_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).std())
train['rmax_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).max())
# moyennes globales par store et par family
train['store_mean'] = train.groupby('store_nbr')['sales'].transform('mean')
train['family_mean'] = train.groupby('family')['sales'].transform('mean')
train['store_family_mean'] = train.groupby(key)['sales'].transform('mean')

# Réapplique les mêmes features au test en utilisant l'historique train complet
# (méthode simplifiée : concat, shift, re-split)
print(f'lag train fait {time.time()-t0:.0f}s')
all_dates = pd.concat([train, test.assign(sales=np.nan)], ignore_index=True)
all_dates = all_dates.sort_values(['store_nbr','family','date']).reset_index(drop=True)
grp_all = all_dates.groupby(key)['sales']
for lag in lags:
    all_dates[f'lag_{lag}'] = grp_all.shift(lag)
for w in [7,14,28,56,84,168,364]:
    all_dates[f'rmean_{w}'] = grp_all.transform(lambda s: s.rolling(w, min_periods=1).mean())
all_dates['rstd_28'] = grp_all.transform(lambda s: s.rolling(28, min_periods=1).std())
all_dates['rmax_28'] = grp_all.transform(lambda s: s.rolling(28, min_periods=1).max())
all_dates['store_mean'] = all_dates.groupby('store_nbr')['sales'].transform('mean')
all_dates['family_mean'] = all_dates.groupby('family')['sales'].transform('mean')
all_dates['store_family_mean'] = all_dates.groupby(key)['sales'].transform('mean')

train = all_dates[all_dates['sales'].notna()].copy()
test = all_dates[all_dates['sales'].isna()].copy()
test['id'] = test['id'].astype(int)
print(f'features complètes {time.time()-t0:.0f}s')

# ============ Validation temporelle ============
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1e-9, denom))

train['log_sales'] = np.log1p(train['sales'])
last_date = train['date'].max()
val_dates = pd.date_range(last_date - pd.Timedelta(days=15), last_date)
val = train[train['date'].isin(val_dates)]
tr = train[~train['date'].isin(val_dates)]

exclude = ['id','date','sales','log_sales','store_nbr','family','city']
feats = [c for c in tr.columns if c not in exclude]
print(f'features: {len(feats)}')

# type du cluster
for c in ['cluster']:
    if c in tr.columns: tr[c] = tr[c].astype('category'); val[c] = val[c].astype('category')

print(f'train: {len(tr)}, val: {len(val)}')
model = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=800, num_leaves=63,
                          colsample_bytree=0.7, subsample=0.8, random_state=42,
                          n_jobs=-1, verbosity=-1)
t1 = time.time()
model.fit(tr[feats], tr['log_sales'])
pred = np.expm1(model.predict(val[feats]))
s = smape(val['sales'].values, pred)
print(f'SMAPE validation enrichi: {s:.4f} ({time.time()-t1:.0f}s)')
print(f'vs baseline brut: 0.5341')
np.save('feat_val_pred.npy', pred)
np.save('feat_val_true.npy', val['sales'].values)
print('terminé.')
