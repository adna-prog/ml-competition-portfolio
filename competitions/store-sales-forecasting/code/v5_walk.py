"""Store Sales v5 : repartir de v3B (config exacte: brut, lr=0.05, 800 arbres, 45 features)
+ enrichir features (plus de lags/rolling) + validation walk-forward multi-fenêtres.
Change UNE chose à la fois pour isoler l'effet.
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

all_dates = pd.concat([train.assign(sales=train['sales']), test.assign(sales=np.nan)], ignore_index=True)
all_dates = all_dates.sort_values(['store_nbr','family','date']).reset_index(drop=True)
key = ['store_nbr','family']
grp = all_dates.groupby(key)['sales']
# PLUS de lags : 1..30 + saisonniers 7,14,28,56,84,112,140,168,196,224,252,280,308,336,364
lags = [1,2,3,4,5,6,7,14,21,28,42,56,84,112,140,168,196,224,252,280,308,336,364]
for lag in lags:
    all_dates[f'lag_{lag}'] = grp.shift(lag)
# PLUS de rolling
for w in [7,14,21,28,42,56,84,112,168,224,280,364]:
    all_dates[f'rmean_{w}'] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
all_dates['rstd_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).std())
all_dates['rstd_56'] = grp.transform(lambda s: s.rolling(56, min_periods=1).std())
all_dates['rmax_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).max())
all_dates['rmin_28'] = grp.transform(lambda s: s.rolling(28, min_periods=1).min())
all_dates['store_mean'] = all_dates.groupby('store_nbr')['sales'].transform('mean')
all_dates['family_mean'] = all_dates.groupby('family')['sales'].transform('mean')
all_dates['store_family_mean'] = all_dates.groupby(key)['sales'].transform('mean')
# ratios lag
for l1, l2 in [(7,14),(14,28),(28,56),(56,112),(112,224)]:
    all_dates[f'lag_ratio_{l1}_{l2}'] = all_dates[f'lag_{l1}'] / (all_dates[f'lag_{l2}'] + 1)
# tendance récente : lag1 - lag7
all_dates['trend_1_7'] = all_dates['lag_1'] - all_dates['lag_7']
all_dates['cluster'] = all_dates['cluster'].astype('category')

train = all_dates[all_dates['sales'].notna()].copy()
test = all_dates[all_dates['sales'].isna()].copy()
test['id'] = test['id'].astype(int)
print(f'features {time.time()-t0:.0f}s')

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1e-9, denom))

# ==== CV walk-forward multi-fenêtres (leçon du doc : expanding window) ====
exclude = ['id','date','sales','store_nbr','family']
feats = [c for c in train.columns if c not in exclude]
print(f'features: {len(feats)}')

def walk_forward_cv(train_df, feats, model_params, n_windows=3, horizon=16):
    """Entraîne sur fenêtres croissantes, valide sur 16 jours à chaque fois."""
    dates = sorted(train_df['date'].unique())
    last = dates[-1]
    scores = []
    # 3 fenêtres : finir à last, -16, -32
    for back in range(n_windows):
        val_end = last - pd.Timedelta(days=16*back)
        val_start = val_end - pd.Timedelta(days=15)
        val_dates_r = pd.date_range(val_start, val_end)
        val = train_df[train_df['date'].isin(val_dates_r)]
        # train = tout avant val_start
        tr = train_df[train_df['date'] < val_start]
        if len(tr) < 500000:
            break
        m = lgb.LGBMRegressor(**model_params)
        m.fit(tr[feats], tr['sales'])
        pred = np.clip(m.predict(val[feats]), 0, None)
        s = smape(val['sales'].values, pred)
        scores.append(s)
        print(f'  fenêtre back={back} (train {len(tr)}, val {len(val)}): SMAPE {s:.4f}')
    return np.mean(scores), scores

# config = v3B exacte
params = dict(learning_rate=0.05, n_estimators=800, num_leaves=63,
              colsample_bytree=0.7, subsample=0.8, random_state=42, n_jobs=-1, verbosity=-1)

print('\n=== LGB v5 (features enrichies) walk-forward 3 fenêtres ===')
t1 = time.time()
mean_s, scores = walk_forward_cv(train, feats, params, n_windows=3)
print(f'Moyenne walk-forward: {mean_s:.4f} ({time.time()-t1:.0f}s)')
print(f'Référence v3B (1 fenêtre): 0.4515')
np.save('v5_scores.npy', np.array(scores))
print('terminé.')
