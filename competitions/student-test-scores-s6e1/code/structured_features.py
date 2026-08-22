from pathlib import Path
import pandas as pd,numpy as np,json,lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
r=Path('/home/hermes/kaggle/s6e1/data');out=Path('/home/hermes/kaggle/s6e1/output');tr=pd.read_csv(r/'train.csv');y=tr.pop('exam_score').to_numpy();te=pd.read_csv(r/'test.csv');X=tr.drop(columns=['id']).copy();T=te.drop(columns=['id']).copy()
def feat(d):
 d=d.copy();d['study_x_attendance']=d.study_hours*d.class_attendance;d['study_per_sleep']=d.study_hours/(d.sleep_hours+1e-3);d['attendance_per_study']=d.class_attendance/(d.study_hours+1);d['study_sq']=d.study_hours**2;d['attendance_sq']=d.class_attendance**2;d['sleep_sq']=d.sleep_hours**2;d['sleep_quality_num']=d.sleep_quality.map({'poor':0,'average':1,'good':2});d['sleep_quality_x_sleep']=d.sleep_quality_num*d.sleep_hours;d['study_method__sleep_quality']=d.study_method.astype(str)+'|'+d.sleep_quality.astype(str);d['study_method__facility']=d.study_method.astype(str)+'|'+d.facility_rating.astype(str);d['course__difficulty']=d.course.astype(str)+'|'+d.exam_difficulty.astype(str);return d
X=feat(X);T=feat(T);cat=[]
for c in X.columns:
 if not pd.api.types.is_numeric_dtype(X[c]):
  cats=pd.Index(pd.unique(pd.concat([X[c].dropna().astype(str),T[c].dropna().astype(str)])));X[c]=pd.Categorical(X[c].astype(str),categories=cats);T[c]=pd.Categorical(T[c].astype(str),categories=cats);cat.append(c)
oof=np.zeros(len(X));folds=[]
for fold,(a,b) in enumerate(KFold(3,shuffle=True,random_state=20260873).split(X)):
 d=lgb.Dataset(X.iloc[a],label=y[a],categorical_feature=cat);v=lgb.Dataset(X.iloc[b],label=y[b],categorical_feature=cat,reference=d);m=lgb.train({'objective':'regression','learning_rate':.03,'num_leaves':63,'min_data_in_leaf':100,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'lambda_l2':2,'verbosity':-1,'seed':20260873+fold,'num_threads':8},d,num_boost_round=1400,valid_sets=[v],callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)]);p=m.predict(X.iloc[b],num_iteration=m.best_iteration);oof[b]=p;folds.append({'fold':fold,'rmse':float(mean_squared_error(y[b],p)**.5),'iter':int(m.best_iteration)});print(folds[-1],flush=True)
res={'model':'structured_features_lgbm','rmse':float(mean_squared_error(y,oof)**.5),'folds':folds};(out/'structured_features_metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
