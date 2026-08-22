from pathlib import Path
import pandas as pd,numpy as np,json,lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
r=Path('/home/hermes/kaggle/s6e1/data');out=Path('/home/hermes/kaggle/s6e1/output');tr=pd.read_csv(r/'train.csv');y=tr.pop('exam_score').to_numpy();X0=tr.drop(columns=['id']).copy()
def feat(d):
 d=d.copy();d['high_att_study']=((d.class_attendance>=90)&(d.study_hours>=6)).astype('int8');d['ideal_sleep']=((d.sleep_hours>=7)&(d.sleep_hours<=9)).astype('int8');d['ideal_study']=(d.study_hours>=7).astype('int8');return d
X0=feat(X0);num=X0.select_dtypes(include=np.number).columns.tolist();oof=np.zeros(len(X0));folds=[]
for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=20260873).split(X0)):
 X=X0.copy();stats=X.iloc[a][num].agg(['mean','std']);lo=stats.loc['mean']-3*stats.loc['std'];hi=stats.loc['mean']+3*stats.loc['std'];X[num]=X[num].clip(lo,hi,axis=1);cat=[]
 for c in X.columns:
  if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category');cat.append(c)
 d=lgb.Dataset(X.iloc[a],label=y[a],categorical_feature=cat);v=lgb.Dataset(X.iloc[b],label=y[b],categorical_feature=cat,reference=d);m=lgb.train({'objective':'regression','learning_rate':.03,'num_leaves':31,'max_depth':5,'min_data_in_leaf':40,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'lambda_l1':.2,'lambda_l2':.4,'verbosity':-1,'seed':20260873+fold,'num_threads':8},d,num_boost_round=5000,valid_sets=[v],callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(0)]);p=m.predict(X.iloc[b],num_iteration=m.best_iteration);oof[b]=p;folds.append({'fold':fold,'rmse':float(mean_squared_error(y[b],p)**.5),'iter':int(m.best_iteration)});print(folds[-1],flush=True)
res={'model':'public_features_capped_lgbm','rmse':float(mean_squared_error(y,oof)**.5),'folds':folds};(out/'capped_metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
