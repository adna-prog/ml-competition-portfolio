from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

SEED=20260854
INPUTS=list(Path('/kaggle/input').rglob('train.csv'))
if not INPUTS: raise FileNotFoundError('train.csv not found')
DATA=INPUTS[0].parent;OUT=Path('/kaggle/working')
tr=pd.read_csv(DATA/'train.csv');te=pd.read_csv(DATA/'test.csv');sample=pd.read_csv(DATA/'sample_submission.csv')
y_raw=tr.pop('health_condition');classes=np.array(sorted(y_raw.unique()));y=pd.Categorical(y_raw,categories=classes).codes
X=tr.drop(columns=['id']).copy();Xt=te.drop(columns=['id']).copy();cat=[]
for c in X.columns:
 if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
  cats=pd.Index(pd.unique(pd.concat([X[c].dropna().astype(str),Xt[c].dropna().astype(str)])));X[c]=pd.Categorical(X[c].astype(str),categories=cats);Xt[c]=pd.Categorical(Xt[c].astype(str),categories=cats);cat.append(c)
folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(X,y));screen_fit,screen_val=folds[0]
base={'objective':'multiclass','num_class':3,'verbosity':-1,'seed':SEED,'num_threads':-1}
def run(params,a,b):
 p=base|params;d=lgb.Dataset(X.iloc[a],label=y[a],categorical_feature=cat);v=lgb.Dataset(X.iloc[b],label=y[b],categorical_feature=cat,reference=d);m=lgb.train(p,d,num_boost_round=1200,valid_sets=[v],callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(0)]);pr=np.argmax(m.predict(X.iloc[b],num_iteration=m.best_iteration),axis=1);return float(accuracy_score(y[b],pr)),float(f1_score(y[b],pr,average='macro')),int(m.best_iteration)
def suggest(t):return {'learning_rate':t.suggest_float('learning_rate',.02,.12,log=True),'num_leaves':t.suggest_int('num_leaves',15,127,step=16),'min_data_in_leaf':t.suggest_int('min_data_in_leaf',50,500,step=50),'feature_fraction':t.suggest_float('feature_fraction',.65,1.0),'bagging_fraction':t.suggest_float('bagging_fraction',.65,1.0),'bagging_freq':1,'lambda_l1':t.suggest_float('lambda_l1',1e-3,5.0,log=True),'lambda_l2':t.suggest_float('lambda_l2',.1,20.0,log=True)}
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
def objective(t):
 p=suggest(t);a,f,it=run(p,screen_fit,screen_val);t.set_user_attr('f1_macro',f);t.set_user_attr('best_iteration',it);return a
study.optimize(objective,n_trials=10,show_progress_bar=False)
ranked=sorted(study.trials,key=lambda t:t.value,reverse=True)[:3];promoted=[]
for t in ranked:
 vals=[]
 for a,b in folds:vals.append(run(t.params,a,b))
 promoted.append({'trial':t.number,'params':t.params,'screen_accuracy':t.value,'cv_accuracy':float(np.mean([x[0] for x in vals])),'cv_f1_macro':float(np.mean([x[1] for x in vals])),'iterations':int(np.median([x[2] for x in vals]))})
best=max(promoted,key=lambda z:z['cv_accuracy']);p=base|best['params'];d=lgb.Dataset(X,label=y,categorical_feature=cat);m=lgb.train(p,d,num_boost_round=best['iterations'],callbacks=[lgb.log_evaluation(0)]);pred=classes[np.argmax(m.predict(Xt),axis=1)];sub=pd.DataFrame({'id':sample['id'],'health_condition':pred});sub.to_csv(OUT/'submission_optuna_lgbm.csv',index=False)
res={'seed':SEED,'n_trials':10,'screen_best':study.best_value,'promoted':promoted,'selected':best,'id_excluded':True};(OUT/'optuna_metrics.json').write_text(json.dumps(res,indent=2,default=float));print(json.dumps(res,indent=2))
