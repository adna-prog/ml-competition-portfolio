from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')
SEED=20260867
files=list(Path('/kaggle/input').rglob('train.csv'))
if not files: raise FileNotFoundError('train.csv not found')
DATA=files[0].parent;OUT=Path('/kaggle/working')
tr=pd.read_csv(DATA/'train.csv');te=pd.read_csv(DATA/'test.csv');sample=pd.read_csv(DATA/'sample_submission.csv')
y_raw=tr.pop('health_condition');classes=np.array(sorted(y_raw.unique()));y=pd.Categorical(y_raw,categories=classes).codes
# Fold-safe feature construction.
def engineer(df,bin_edges=None):
 out=df.drop(columns=['id'],errors='ignore').copy()
 if bin_edges is None:
  vals=out.sleep_duration.dropna().to_numpy();bin_edges=np.unique(np.quantile(vals,np.linspace(0,1,6)))
 out['sleep_bin']=pd.cut(out.sleep_duration,bins=np.r_[-np.inf,bin_edges[1:-1],np.inf],labels=False,include_lowest=True).astype('Int64').astype(str)
 out['stress_sleep_interact']=out['stress_level'].fillna('Missing').astype(str)+'_'+out['sleep_bin'].fillna('Missing').astype(str)
 return out,bin_edges
cat_cols=['stress_level','physical_activity_level','diet_type','gender','smoking_alcohol','sleep_quality','stress_sleep_interact']
num_cols=['sleep_duration','heart_rate','bmi','calorie_expenditure','step_count','exercise_duration','water_intake']
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y));oof_cb=np.zeros((len(tr),3));oof_lgb=np.zeros((len(tr),3));test_cb=np.zeros((len(te),3));test_lgb=np.zeros((len(te),3));fold_meta=[]
for fold,(a,b) in enumerate(folds):
 raw_tr,edges=engineer(tr.iloc[a]);raw_va,_=engineer(tr.iloc[b],edges);raw_te,_=engineer(te,edges)
 for c in cat_cols:
  raw_tr[c]=raw_tr[c].fillna('Missing').astype(str);raw_va[c]=raw_va[c].fillna('Missing').astype(str);raw_te[c]=raw_te[c].fillna('Missing').astype(str)
 # sklearn TargetEncoder performs inner cross-fitting for fit_transform.
 enc=TargetEncoder(categories='auto',target_type='multiclass',cv=5,smooth='auto',random_state=SEED+fold)
 Xtr_cat=enc.fit_transform(raw_tr[cat_cols],y[a]);Xva_cat=enc.transform(raw_va[cat_cols]);Xte_cat=enc.transform(raw_te[cat_cols])
 imp=SimpleImputer(strategy='median');Xtr_num=imp.fit_transform(raw_tr[num_cols]);Xva_num=imp.transform(raw_va[num_cols]);Xte_num=imp.transform(raw_te[num_cols])
 Xtr=np.hstack([Xtr_num,Xtr_cat]);Xva=np.hstack([Xva_num,Xva_cat]);Xtest=np.hstack([Xte_num,Xte_cat]);sw=compute_sample_weight('balanced',y[a])
 cb=CatBoostClassifier(iterations=1200,learning_rate=.03,depth=6,loss_function='MultiClass',auto_class_weights='Balanced',random_seed=SEED+fold,verbose=False,allow_writing_files=False,thread_count=8)
 cb.fit(Xtr,y[a],eval_set=(Xva,y[b]),early_stopping_rounds=80,verbose=False);oof_cb[b]=cb.predict_proba(Xva);test_cb+=cb.predict_proba(Xtest)/5
 dl=lgb.Dataset(Xtr,label=y[a],weight=sw);vl=lgb.Dataset(Xva,label=y[b],reference=dl);lm=lgb.train({'objective':'multiclass','num_class':3,'learning_rate':.03,'num_leaves':63,'min_data_in_leaf':100,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':1,'lambda_l2':2,'verbosity':-1,'seed':SEED+fold,'num_threads':-1},dl,num_boost_round=1500,valid_sets=[vl],callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)]);oof_lgb[b]=lm.predict(Xva,num_iteration=lm.best_iteration);test_lgb+=lm.predict(Xtest,num_iteration=lm.best_iteration)/5
 fold_meta.append({'fold':fold,'catboost_best_iteration':int(cb.get_best_iteration()+1),'lgb_best_iteration':int(lm.best_iteration)})
 print('fold',fold,fold_meta[-1],flush=True)
def metrics(p):
 q=np.argmax(p,axis=1);return {'balanced_accuracy':float(balanced_accuracy_score(y,q)),'accuracy':float(accuracy_score(y,q)),'macro_f1':float(f1_score(y,q,average='macro')),'confusion_matrix':confusion_matrix(y,q).tolist()}
res={'metric':'balanced_accuracy','catboost':metrics(oof_cb),'lightgbm':metrics(oof_lgb),'blends':{},'fold_meta':fold_meta}
for a in [0,.25,.5,.75,1]:res['blends'][str(a)]=metrics(a*oof_cb+(1-a)*oof_lgb)
np.savez_compressed(OUT/'corrected_oof_predictions.npz',catboost=oof_cb,lightgbm=oof_lgb,target=y);(OUT/'corrected_oof_metrics.json').write_text(json.dumps(res,indent=2,default=float))
# Candidate uses fixed OOF-selected blend and class decision biases.
alpha=.75;bias=np.array([-.07030469168305757,-.04532351377094208,.09936039579317002]);p=np.log(np.clip(alpha*test_cb+(1-alpha)*test_lgb,1e-8,1))+bias;sub=pd.DataFrame({'id':sample.id,'health_condition':classes[np.argmax(p,axis=1)]});sub.to_csv(OUT/'submission_corrected_core.csv',index=False);print(json.dumps(res,indent=2,default=float));print('selected_alpha',alpha,'decision_bias',bias.tolist())
