from pathlib import Path
import json,warnings,itertools,gc
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
warnings.filterwarnings('ignore')
SEED=20260871
files=list(Path('/kaggle/input').rglob('train.csv'))
if not files:raise FileNotFoundError('train.csv not found')
DATA=files[0].parent;OUT=Path('/kaggle/working');tr=pd.read_csv(DATA/'train.csv');te=pd.read_csv(DATA/'test.csv');sample=pd.read_csv(DATA/'sample_submission.csv');yraw=tr.pop('health_condition');classes=np.array(sorted(yraw.unique()));y=pd.Categorical(yraw,categories=classes).codes
cat_cols=['stress_level','physical_activity_level','diet_type','gender','smoking_alcohol','sleep_quality','stress_sleep_interact'];num_cols=['sleep_duration','heart_rate','bmi','calorie_expenditure','step_count','exercise_duration','water_intake']
def eng(df,edges=None):
 x=df.drop(columns=['id'],errors='ignore').copy()
 if edges is None:edges=np.unique(np.quantile(x.sleep_duration.dropna(),np.linspace(0,1,6)))
 x['sleep_bin']=pd.cut(x.sleep_duration,bins=np.r_[-np.inf,edges[1:-1],np.inf],labels=False,include_lowest=True).astype('Int64').astype(str);x['stress_sleep_interact']=x.stress_level.fillna('Missing').astype(str)+'_'+x.sleep_bin.fillna('Missing').astype(str)
 # Selected unsupervised signal features from the audited notebook.
 x['missing_count']=x.isna().sum(axis=1).astype('int8');x['sleep_distance_8h']=(x.sleep_duration-8).abs();x['sleep_shortfall_7h']=(7-x.sleep_duration).clip(lower=0);x['sleep_excess_9h']=(x.sleep_duration-9).clip(lower=0);x['bmi_distance_22']=(x.bmi-22).abs();x['bmi_over_25']=(x.bmi-25).clip(lower=0);x['log_step_count']=np.log1p(x.step_count.clip(lower=0));x['activity_mix']=x.step_count/1000+x.exercise_duration/10;x['exercise_per_1k_steps']=x.exercise_duration/(1+x.step_count/1000);x['sleep_x_exercise']=x.sleep_duration*x.exercise_duration;x['sleep_x_log_steps']=x.sleep_duration*np.log1p(x.step_count.clip(lower=0));return x,edges
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y));oof={k:np.zeros((len(tr),3)) for k in ['catboost','lgb_signal','xgb']};testp={k:np.zeros((len(te),3)) for k in oof};records=[]
for fold,(a,b) in enumerate(folds):
 ra,e=eng(tr.iloc[a]);rv,_=eng(tr.iloc[b],e);rt,_=eng(te,e)
 for z in [ra,rv,rt]:
  for c in cat_cols:z[c]=z[c].fillna('Missing').astype(str)
 enc=TargetEncoder(target_type='multiclass',cv=5,smooth='auto',random_state=SEED+fold);Acat=enc.fit_transform(ra[cat_cols],y[a]);Bcat=enc.transform(rv[cat_cols]);Tcat=enc.transform(rt[cat_cols]);num_feature_cols=[c for c in ra.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(ra[c])];imp=SimpleImputer(strategy='median');An=imp.fit_transform(ra[num_feature_cols]);Bn=imp.transform(rv[num_feature_cols]);Tn=imp.transform(rt[num_feature_cols]);A=np.hstack([An,Acat]);B=np.hstack([Bn,Bcat]);T=np.hstack([Tn,Tcat]);sw=compute_sample_weight('balanced',y[a])
 # signal: numeric features plus TE blocks for stress/activity/sleep interaction.
 signal_cols=[c for c in ra.drop(columns=cat_cols).columns if c in num_cols or c.startswith(('missing_count','sleep_distance','sleep_shortfall','sleep_excess','bmi_distance','bmi_over','log_step','activity_mix','exercise_per','sleep_x_'))]
 idx_num=list(range(An.shape[1]));idx_cat=[]
 for j,c in enumerate(cat_cols):
  if c in ['stress_level','physical_activity_level','stress_sleep_interact']:idx_cat += list(range(An.shape[1]+3*j,An.shape[1]+3*(j+1)))
 idx=idx_num+idx_cat;As=A[:,idx];Bs=B[:,idx];Ts=T[:,idx]
 cb=CatBoostClassifier(iterations=1400,learning_rate=.04,depth=6,loss_function='MultiClass',auto_class_weights='Balanced',random_seed=SEED+fold,verbose=False,allow_writing_files=False,thread_count=8);cb.fit(A,y[a],eval_set=(B,y[b]),early_stopping_rounds=80,verbose=False);oof['catboost'][b]=cb.predict_proba(B);testp['catboost']+=cb.predict_proba(T)/5
 lm=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=1800,learning_rate=.03,num_leaves=63,min_child_samples=100,subsample=.9,colsample_bytree=.9,reg_alpha=.03,reg_lambda=2,random_state=SEED+fold,n_jobs=-1,verbosity=-1);lm.fit(As,y[a],sample_weight=sw,eval_set=[(Bs,y[b])],callbacks=[lgb.early_stopping(100,verbose=False)]);oof['lgb_signal'][b]=lm.predict_proba(Bs);testp['lgb_signal']+=lm.predict_proba(Ts)/5
 xgb=XGBClassifier(objective='multi:softprob',num_class=3,n_estimators=1400,learning_rate=.04,max_depth=7,min_child_weight=8,subsample=.9,colsample_bytree=.9,reg_alpha=.03,reg_lambda=2,tree_method='hist',device='cuda',random_state=SEED+fold,n_jobs=4);xgb.fit(A,y[a],sample_weight=sw,eval_set=[(B,y[b])],verbose=False);oof['xgb'][b]=xgb.predict_proba(B);testp['xgb']+=xgb.predict_proba(T)/5
 records.append({'fold':fold,'cat_iter':int(cb.get_best_iteration()+1),'lgb_iter':int(lm.best_iteration_)});print('fold',fold,records[-1],flush=True);del cb,lm,xgb;gc.collect()
# Held-out OOF weight selection.
keys=list(oof);stack=np.stack([oof[k] for k in keys]);tstack=np.stack([testp[k] for k in keys]);priors=np.bincount(y,minlength=3)/len(y);cal,dec=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=SEED+100)
def score(probs,idx,alpha):return balanced_accuracy_score(y[idx],np.argmax(np.log(np.clip(probs[idx],1e-12,1))+(-alpha*np.log(priors))[None,:],1))
records_w=[]
for w in itertools.product([0,.25,.5,.75,1],repeat=3):
 if abs(sum(w)-1)>1e-9:continue
 p=np.tensordot(np.array(w),stack,axes=(0,0));
 for alpha in [.9,1,1.1]:records_w.append({'weights':w,'alpha':alpha,'cal':score(p,cal,alpha),'dec':score(p,dec,alpha)})
# Add the equal-weight baseline explicitly; it is not representable by the quarter grid.
p=np.mean(stack,axis=0)
for alpha in [.9,1,1.1]:records_w.append({'weights':(1/3,1/3,1/3),'alpha':alpha,'cal':score(p,cal,alpha),'dec':score(p,dec,alpha)})
base=[r for r in records_w if r['weights']==(1/3,1/3,1/3) and r['alpha']==1][0];best=max(records_w,key=lambda r:r['cal']);accept=best['dec']>=base['dec'];sel=best if accept else base
w=np.array(sel['weights']);p=np.tensordot(w,stack,axes=(0,0));pt=np.tensordot(w,tstack,axes=(0,0));finalpred=np.argmax(np.log(np.clip(p,1e-12,1))+(-sel['alpha']*np.log(priors))[None,:],1);metrics={'models':{k:float(balanced_accuracy_score(y,np.argmax(oof[k],1))) for k in keys},'selected':sel,'base_decision':base['dec'],'accepted':accept,'full_oof_bacc':float(balanced_accuracy_score(y,finalpred)),'fold_records':records};(OUT/'three_model_metrics.json').write_text(json.dumps(metrics,indent=2,default=float));np.savez_compressed(OUT/'three_model_oof.npz',target=y,catboost=oof['catboost'],lgb_signal=oof['lgb_signal'],xgb=oof['xgb']);pred=classes[np.argmax(np.log(np.clip(pt,1e-12,1))+(-sel['alpha']*np.log(priors))[None,:],1)];pd.DataFrame({'id':sample.id,'health_condition':pred}).to_csv(OUT/'submission_three_model_clean.csv',index=False);print(json.dumps(metrics,indent=2,default=float))
