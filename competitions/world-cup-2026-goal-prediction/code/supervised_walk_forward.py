from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error, f1_score

ROOT=Path(__file__).parents[1]; RAW=ROOT/'data/raw'; OUT=ROOT/'output'; OUT.mkdir(parents=True,exist_ok=True)
TRAIN=pd.read_csv(RAW/'Train.csv'); TEST=pd.read_csv(RAW/'Test.csv'); TOURN=pd.read_csv(RAW/'tournaments.csv'); WINNERS=dict(zip(TOURN.tournament_id,TOURN.winner)); SEED=20260842
STAGE_MAP={'group stage':'group','round of 16':'roundof16','quarter-finals':'qf','semi-finals':'sf','third-place match':'sf'}; STAGES=['group','roundof32','roundof16','qf','sf','runnerup','champion']

def stage(row):
 if row.stage_reached=='final': return 'champion' if WINNERS.get(row.tournament_id)==row.country else 'runnerup'
 return STAGE_MAP.get(row.stage_reached)

def history_features(history, target):
 global_goals=float(history.total_goals.mean()) if len(history) else 3.0
 rows=[]
 for _,r in target.iterrows():
  h=history[history.country==r.country].sort_values('year');g=h.total_goals.to_numpy(float);m=h.matches_played.to_numpy(float)
  if len(h):
   stages=[stage(x) for _,x in h.iterrows()];stages=[s for s in stages if s in STAGES]
   rows.append({'country':r.country,'team_code':r.team_code,'confederation_name':r.confederation_name,'region_name':r.region_name,'target_year':r.year,'history_count':len(h),'goal_mean':g.mean(),'goal_median':np.median(g),'recent_goal_mean':g[-3:].mean(),'last_goals':g[-1],'goals_std':g.std(),'matches_mean':m.mean(),'last_matches':m[-1],'stage_rank_last':STAGES.index(stages[-1]) if stages else 0,'stage_rank_best':max([STAGES.index(s) for s in stages]) if stages else 0,'stage_rank_mean':np.mean([STAGES.index(s) for s in stages]) if stages else 0,'years_since_last':int(r.year-h.year.max())})
  else:
   rows.append({'country':r.country,'team_code':r.team_code,'confederation_name':r.confederation_name,'region_name':r.region_name,'target_year':r.year,'history_count':0,'goal_mean':global_goals,'goal_median':global_goals,'recent_goal_mean':global_goals,'last_goals':global_goals,'goals_std':0.0,'matches_mean':3.0,'last_matches':3.0,'stage_rank_last':0,'stage_rank_best':0,'stage_rank_mean':0.0,'years_since_last':99})
 return pd.DataFrame(rows)

def prep(df):
 x=df.drop(columns=['country'],errors='ignore').copy()
 cats=[]
 for c in x.columns:
  if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c]) or str(x[c].dtype)=='category':x[c]=x[c].fillna('missing').astype(str);cats.append(x.columns.get_loc(c))
 return x,cats

def evaluate(year):
 hist=TRAIN[TRAIN.year<year].copy();target=TRAIN[TRAIN.year==year].copy();target['true_stage']=target.apply(stage,axis=1);feat=history_features(hist,target);X,cats=prep(feat.drop(columns=['true_stage'],errors='ignore'))
 reg=CatBoostRegressor(iterations=300,depth=4,learning_rate=.04,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8);reg.fit(X,target.total_goals,cat_features=cats,verbose=False);goal=reg.predict(X)
 mask=target.true_stage.notna().to_numpy();clf=CatBoostClassifier(iterations=300,depth=4,learning_rate=.04,loss_function='MultiClass',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8);clf.fit(X.iloc[mask],target.loc[mask,'true_stage'],cat_features=cats,verbose=False);pred=clf.predict(X.iloc[mask]).reshape(-1)
 return {'year':year,'rows':len(target),'rmse_goals':float(mean_squared_error(target.total_goals,goal)**.5),'f1_weighted_stage':float(f1_score(target.loc[mask,'true_stage'],pred,average='weighted',zero_division=0)),'f1_macro_stage':float(f1_score(target.loc[mask,'true_stage'],pred,average='macro',zero_division=0))}

def causal_training_features(df):
    parts=[]
    for year in sorted(df.year.unique()):
        hist=df[df.year<year].copy(); cur=df[df.year==year].copy(); cur['true_stage']=cur.apply(stage,axis=1); parts.append(history_features(hist,cur))
    return pd.concat(parts,ignore_index=True)


def test_metadata(test):
    meta=TRAIN.sort_values('year').drop_duplicates('country',keep='last')[['country','team_code','confederation_name','region_name']]
    out=test.merge(meta,on='country',how='left'); out['year']=2026; return out


if __name__=='__main__':
 vals=[evaluate(y) for y in [2010,2014,2018,2022]];res={'validation':vals,'summary':{k:float(np.mean([v[k] for v in vals])) for k in ['rmse_goals','f1_weighted_stage','f1_macro_stage']}}
 train_feat=causal_training_features(TRAIN); X,cats=prep(train_feat); reg=CatBoostRegressor(iterations=300,depth=4,learning_rate=.04,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8); reg.fit(X,TRAIN.total_goals,cat_features=cats,verbose=False)
 tmeta=test_metadata(TEST); test_feat=history_features(TRAIN,tmeta); Xt,_=prep(test_feat); goals=np.maximum(0,reg.predict(Xt))
 stage_labels=TRAIN.apply(stage,axis=1); valid=stage_labels.notna().to_numpy(); clf=CatBoostClassifier(iterations=300,depth=4,learning_rate=.04,loss_function='MultiClass',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8); clf.fit(X.iloc[valid],stage_labels[valid],cat_features=cats,verbose=False); pred_stage=clf.predict(Xt).reshape(-1)
 sub=pd.DataFrame({'ID':TEST.ID,'total_goals':np.round(goals,3),'Target':pred_stage}); sub.to_csv(OUT/'supervised_stage_practice_submission.csv',index=False); (OUT/'supervised_stage_walk_forward_metrics.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2)); print(sub.head(10).to_string(index=False))
