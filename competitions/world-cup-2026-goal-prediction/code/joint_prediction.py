from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, f1_score

ROOT=Path(__file__).parents[1];RAW=ROOT/'data/raw';OUT=ROOT/'output';OUT.mkdir(parents=True,exist_ok=True);SEED=20260843
TRAIN=pd.read_csv(RAW/'Train.csv');TEST=pd.read_csv(RAW/'Test.csv');TOURN=pd.read_csv(RAW/'tournaments.csv');WINNERS=dict(zip(TOURN.tournament_id,TOURN.winner));STAGES=['group','roundof32','roundof16','qf','sf','runnerup','champion'];MAP={'group stage':'group','round of 16':'roundof16','quarter-finals':'qf','semi-finals':'sf','third-place match':'sf'}

def stage(row):
 if row.stage_reached=='final':return 'champion' if WINNERS.get(row.tournament_id)==row.country else 'runnerup'
 return MAP.get(row.stage_reached)

def history_features(history,target):
 gg=float(history.total_goals.mean()) if len(history) else 3.;rows=[]
 for _,r in target.iterrows():
  h=history[history.country==r.country].sort_values('year');g=h.total_goals.to_numpy(float);m=h.matches_played.to_numpy(float)
  if len(h):
   ss=[stage(x) for _,x in h.iterrows()];ss=[s for s in ss if s in STAGES];ranks=[STAGES.index(s) for s in ss]
   rows.append({'country':r.country,'team_code':r.team_code,'confederation_name':r.confederation_name,'region_name':r.region_name,'target_year':r.year,'history_count':len(h),'goal_mean':g.mean(),'goal_median':np.median(g),'recent_goal_mean':g[-3:].mean(),'last_goals':g[-1],'goals_std':g.std(),'matches_mean':m.mean(),'last_matches':m[-1],'stage_rank_last':ranks[-1] if ranks else 0,'stage_rank_best':max(ranks) if ranks else 0,'stage_rank_mean':np.mean(ranks) if ranks else 0,'years_since_last':int(r.year-h.year.max())})
  else:rows.append({'country':r.country,'team_code':r.team_code,'confederation_name':r.confederation_name,'region_name':r.region_name,'target_year':r.year,'history_count':0,'goal_mean':gg,'goal_median':gg,'recent_goal_mean':gg,'last_goals':gg,'goals_std':0.,'matches_mean':3.,'last_matches':3.,'stage_rank_last':0,'stage_rank_best':0,'stage_rank_mean':0.,'years_since_last':99})
 return pd.DataFrame(rows)

def prep(d):
 x=d.drop(columns=['country'],errors='ignore').copy();cats=[]
 for c in x.columns:
  if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c]):x[c]=x[c].fillna('missing').astype(str);cats.append(x.columns.get_loc(c))
 return x,cats

def quotas(n):
 if n>=48:return {'group':16,'roundof32':16,'roundof16':8,'qf':4,'sf':2,'runnerup':1,'champion':1}
 if n>=32:return {'group':16,'roundof16':8,'qf':4,'sf':2,'runnerup':1,'champion':1}
 # proportional fallback for historical small formats
 q={'group':max(1,n-15),'roundof16':max(0,min(8,n-7)),'qf':min(4,max(0,n-3)),'sf':min(2,max(0,n-1)),'runnerup':1,'champion':1};
 while sum(q.values())>n:q['group']-=1
 return q

def assign(scores,n):
 order=np.argsort(-np.asarray(scores));q=quotas(n);out=np.array(['group']*n,dtype=object);pos=0
 for s,count in [('champion',q.get('champion',0)),('runnerup',q.get('runnerup',0)),('sf',q.get('sf',0)),('qf',q.get('qf',0)),('roundof16',q.get('roundof16',0)),('roundof32',q.get('roundof32',0))]:
  out[order[pos:pos+count]]=s;pos+=count
 return out

def evaluate(year,alpha):
 hist=TRAIN[TRAIN.year<year].copy();target=TRAIN[TRAIN.year==year].copy();feat=history_features(hist,target);X,c=prep(feat);reg=CatBoostRegressor(iterations=300,depth=4,learning_rate=.04,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8);reg.fit(X,target.total_goals,cat_features=c,verbose=False);goals=reg.predict(X);scores=goals+alpha*feat.stage_rank_last.to_numpy();pred=assign(scores,len(target));true=target.apply(stage,axis=1);mask=true.notna();return {'rmse_goals':float(mean_squared_error(target.total_goals,goals)**.5),'f1_weighted_stage':float(f1_score(true[mask],pred[mask],average='weighted',zero_division=0)),'f1_macro_stage':float(f1_score(true[mask],pred[mask],average='macro',zero_division=0))}

def causal_features(df):
 parts=[]
 for year in sorted(df.year.unique()):
  hist=df[df.year<year].copy();cur=df[df.year==year].copy();parts.append(history_features(hist,cur))
 return pd.concat(parts,ignore_index=True)


if __name__=='__main__':
 res={'alphas':{}}
 for alpha in [0.,.25,.5,1.0,2.0]:
  vals=[evaluate(y,alpha) for y in [2010,2014,2018,2022]];res['alphas'][str(alpha)]={'validation':vals,'summary':{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}}
 # final goals and joint stage predictions for 2026
 meta=TRAIN.sort_values('year').drop_duplicates('country',keep='last')[['country','team_code','confederation_name','region_name']];tmeta=TEST.merge(meta,on='country',how='left');tmeta['year']=2026
 train_feat=causal_features(TRAIN);X,c=prep(train_feat);reg=CatBoostRegressor(iterations=300,depth=4,learning_rate=.04,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8);reg.fit(X,TRAIN.total_goals,cat_features=c,verbose=False);test_feat=history_features(TRAIN,tmeta);Xt,_=prep(test_feat);goals=np.maximum(0,reg.predict(Xt));alpha=.5;pred_stage=assign(goals+alpha*test_feat.stage_rank_last.to_numpy(),len(TEST));sub=pd.DataFrame({'ID':TEST.ID,'total_goals':np.round(goals,3),'Target':pred_stage});sub.to_csv(OUT/'joint_practice_submission.csv',index=False);(OUT/'joint_walk_forward_metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2));print(sub.Target.value_counts().to_dict())
