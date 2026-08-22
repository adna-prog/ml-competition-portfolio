from pathlib import Path
import pandas as pd,numpy as np,json,lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
r=Path('/home/hermes/kaggle/s6e1/data');out=Path('/home/hermes/kaggle/s6e1/output');out.mkdir(exist_ok=True)
tr=pd.read_csv(r/'train.csv');y=tr.pop('exam_score').to_numpy();X=tr.drop(columns=['id']).copy();cat=[]
for c in X.columns:
 if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category');cat.append(c)
res=[];oof=np.zeros(len(X))
for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=20260873).split(X)):
 d=lgb.Dataset(X.iloc[a],label=y[a],categorical_feature=cat);v=lgb.Dataset(X.iloc[b],label=y[b],categorical_feature=cat,reference=d);m=lgb.train({'objective':'regression','learning_rate':.03,'num_leaves':63,'min_data_in_leaf':100,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'lambda_l2':2,'verbosity':-1,'seed':20260873+fold,'num_threads':8},d,num_boost_round=1200,valid_sets=[v],callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)]);p=m.predict(X.iloc[b],num_iteration=m.best_iteration);oof[b]=p;res.append({'fold':fold,'rmse':float(mean_squared_error(y[b],p)**.5),'best_iteration':int(m.best_iteration)});(out/'baseline_progress.json').write_text(json.dumps({'folds':res},indent=2));print(res[-1],flush=True)
res2={'mean_baseline':float(mean_squared_error(y,np.full(len(y),y.mean()))**.5),'folds':res,'rmse':float(mean_squared_error(y,oof)**.5)};(out/'baseline_metrics.json').write_text(json.dumps(res2,indent=2));print(json.dumps(res2,indent=2))
