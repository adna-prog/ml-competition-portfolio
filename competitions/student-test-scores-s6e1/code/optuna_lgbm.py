from pathlib import Path
import json,numpy as np,pandas as pd,optuna,lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
SEED=20260875;ROOT=Path('/kaggle/input');files=list(ROOT.rglob('train.csv'))
if not files:raise FileNotFoundError('train.csv')
D=files[0].parent;out=Path('/kaggle/working');tr=pd.read_csv(D/'train.csv');y=tr.pop('exam_score').to_numpy();X=tr.drop(columns=['id']).copy();cat=[]
for c in X.columns:
 if not pd.api.types.is_numeric_dtype(X[c]):X[c]=X[c].astype('category');cat.append(c)
folds=list(KFold(3,shuffle=True,random_state=20260873).split(X));a,b=folds[0]
def run(params,aa,bb):
 d=lgb.Dataset(X.iloc[aa],label=y[aa],categorical_feature=cat);v=lgb.Dataset(X.iloc[bb],label=y[bb],categorical_feature=cat,reference=d);m=lgb.train({'objective':'regression','verbosity':-1,'seed':SEED,'num_threads':-1}|params,d,num_boost_round=1800,valid_sets=[v],callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)]);p=m.predict(X.iloc[bb],num_iteration=m.best_iteration);return float(mean_squared_error(y[bb],p)**.5),int(m.best_iteration)
def sug(t):return {'learning_rate':t.suggest_float('learning_rate',.01,.08,log=True),'num_leaves':t.suggest_int('num_leaves',15,127,step=16),'min_data_in_leaf':t.suggest_int('min_data_in_leaf',50,600,step=50),'feature_fraction':t.suggest_float('feature_fraction',.65,1.0),'bagging_fraction':t.suggest_float('bagging_fraction',.7,1.0),'bagging_freq':1,'lambda_l1':t.suggest_float('lambda_l1',1e-4,5,log=True),'lambda_l2':t.suggest_float('lambda_l2',.1,30,log=True)}
st=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED))
def obj(t):
 p=sug(t);v,it=run(p,a,b);t.set_user_attr('iteration',it);return v
st.optimize(obj,n_trials=8,show_progress_bar=False);top=sorted(st.trials,key=lambda t:t.value)[:3];prom=[]
for t in top:
 vals=[run(t.params,aa,bb) for aa,bb in folds];prom.append({'trial':t.number,'params':t.params,'screen_rmse':t.value,'cv_rmse':float(np.mean([v[0] for v in vals])),'iterations':int(np.median([v[1] for v in vals])),'folds':[v[0] for v in vals]})
res={'seed':SEED,'trials':len(st.trials),'promoted':prom,'best':min(prom,key=lambda z:z['cv_rmse'])};(out/'optuna_s1_metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
