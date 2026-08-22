from pathlib import Path
import json,warnings
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')
SEED=20260869
files=list(Path('/kaggle/input').rglob('train.csv'))
if not files: raise FileNotFoundError('train.csv not found')
DATA=files[0].parent;OUT=Path('/kaggle/working')
tr=pd.read_csv(DATA/'train.csv');te=pd.read_csv(DATA/'test.csv');sample=pd.read_csv(DATA/'sample_submission.csv');yraw=tr.pop('health_condition');classes=np.array(sorted(yraw.unique()));y=pd.Categorical(yraw,categories=classes).codes
cat_cols=['stress_level','physical_activity_level','diet_type','gender','smoking_alcohol','sleep_quality','stress_sleep_interact'];num_cols=['sleep_duration','heart_rate','bmi','calorie_expenditure','step_count','exercise_duration','water_intake']
def eng(df,edges=None):
 x=df.drop(columns=['id'],errors='ignore').copy()
 if edges is None:edges=np.unique(np.quantile(x.sleep_duration.dropna(),np.linspace(0,1,6)))
 x['sleep_bin']=pd.cut(x.sleep_duration,bins=np.r_[-np.inf,edges[1:-1],np.inf],labels=False,include_lowest=True).astype('Int64').astype(str);x['stress_sleep_interact']=x.stress_level.fillna('Missing').astype(str)+'_'+x.sleep_bin.fillna('Missing').astype(str);return x,edges
def encode(a,b,c,ya,seed):
 for z in [a,b,c]:
  for col in cat_cols:z[col]=z[col].fillna('Missing').astype(str)
 enc=TargetEncoder(target_type='multiclass',cv=5,smooth='auto',random_state=seed);A=enc.fit_transform(a[cat_cols],ya);B=enc.transform(b[cat_cols]);C=enc.transform(c[cat_cols]);imp=SimpleImputer(strategy='median');An=imp.fit_transform(a[num_cols]);Bn=imp.transform(b[num_cols]);Cn=imp.transform(c[num_cols]);return np.hstack([An,A]),np.hstack([Bn,B]),np.hstack([Cn,C])
def model(seed):return CatBoostClassifier(iterations=1200,learning_rate=.03,depth=6,loss_function='MultiClass',auto_class_weights='Balanced',random_seed=seed,verbose=False,allow_writing_files=False,thread_count=8)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y));base_scores=[];aug_scores=[];pseudo_counts=[]
for fold,(a,b) in enumerate(folds):
 ra,edges=eng(tr.iloc[a]);rv,_=eng(tr.iloc[b],edges);rt,_=eng(te,edges);A,B,T=encode(ra,rv,rt,y[a],SEED+fold)
 m=model(SEED+fold);m.fit(A,y[a],verbose=False);pt=m.predict_proba(T);conf=pt.max(1);mask=conf>.99;pl=pt.argmax(1);pseudo_counts.append(int(mask.sum()));base_scores.append(balanced_accuracy_score(y[b],np.argmax(m.predict_proba(B),1)))
 A2=np.vstack([A,T[mask]]);y2=np.r_[y[a],pl[mask]];sw=np.r_[compute_sample_weight('balanced',y[a]),np.full(mask.sum(),.25)];m2=model(SEED+fold);m2.fit(A2,y2,sample_weight=sw,verbose=False);aug_scores.append(balanced_accuracy_score(y[b],np.argmax(m2.predict_proba(B),1)));print('fold',fold,'pseudo',int(mask.sum()),'base',base_scores[-1],'aug',aug_scores[-1],flush=True)
# Final full-data pseudo-label retrain.
ra,edges=eng(tr);rt,_=eng(te,edges);A,_,T=encode(ra,ra,rt,y,SEED);m=model(SEED);m.fit(A,y,verbose=False);pt=m.predict_proba(T);mask=pt.max(1)>.99;pl=pt.argmax(1);A2=np.vstack([A,T[mask]]);y2=np.r_[y,pl[mask]];sw=np.r_[compute_sample_weight('balanced',y),np.full(mask.sum(),.25)];m2=model(SEED);m2.fit(A2,y2,sample_weight=sw,verbose=False);pred=classes[np.argmax(m2.predict_proba(T),1)];pd.DataFrame({'id':sample.id,'health_condition':pred}).to_csv(OUT/'submission_pseudolabel_catboost.csv',index=False)
res={'base_cv':float(np.mean(base_scores)),'augmented_cv':float(np.mean(aug_scores)),'delta':float(np.mean(aug_scores)-np.mean(base_scores)),'base_folds':base_scores,'augmented_folds':aug_scores,'pseudo_counts':pseudo_counts,'final_pseudo_count':int(mask.sum()),'confidence_threshold':.99,'pseudo_weight':.25};(OUT/'pseudolabel_metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
