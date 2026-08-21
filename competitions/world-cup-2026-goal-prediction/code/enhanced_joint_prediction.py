from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, f1_score
from joint_prediction import STAGES, stage, prep, assign

ROOT=Path(__file__).parents[1]; RAW=ROOT/'data/raw'; OUT=ROOT/'output'; SEED=20260844
TRAIN=pd.read_csv(RAW/'Train.csv'); TEST=pd.read_csv(RAW/'Test.csv'); APPS=pd.read_csv(RAW/'team_appearances.csv'); MATCH=pd.read_csv(RAW/'matches.csv')
# Normalize historical match dates and use only matches strictly before each target tournament year.
APPS['match_year']=pd.to_datetime(APPS['match_date']).dt.year
MATCH['match_year']=pd.to_datetime(MATCH['match_date']).dt.year

def enhanced_history_features(history, target):
    base=[]
    for _,r in target.iterrows():
        h=history[history.country==r.country].sort_values('year'); g=h.total_goals.to_numpy(float); m=h.matches_played.to_numpy(float)
        if len(h):
            ss=[stage(x) for _,x in h.iterrows() if stage(x) in STAGES]; ranks=[STAGES.index(s) for s in ss]
            vals=dict(history_count=len(h),goal_mean=g.mean(),goal_median=np.median(g),recent_goal_mean=g[-3:].mean(),last_goals=g[-1],goals_std=g.std(),matches_mean=m.mean(),last_matches=m[-1],stage_rank_last=ranks[-1] if ranks else 0,stage_rank_best=max(ranks) if ranks else 0,stage_rank_mean=np.mean(ranks) if ranks else 0,years_since_last=int(r.year-h.year.max()))
        else:
            gg=float(history.total_goals.mean()) if len(history) else 3.; vals=dict(history_count=0,goal_mean=gg,goal_median=gg,recent_goal_mean=gg,last_goals=gg,goals_std=0.,matches_mean=3.,last_matches=3.,stage_rank_last=0,stage_rank_best=0,stage_rank_mean=0.,years_since_last=99)
        code=r.team_code; apps=APPS[(APPS.team_code==code)&(APPS.match_year<r.year)].sort_values('match_date')
        if len(apps):
            recent=apps.tail(10); vals.update(match_count=len(apps),match_gf_mean=float(apps.goals_for.mean()),match_ga_mean=float(apps.goals_against.mean()),match_gd_mean=float((apps.goals_for-apps.goals_against).mean()),win_rate=float(apps.win.mean()),draw_rate=float(apps.draw.mean()),recent_gf_mean=float(recent.goals_for.mean()),recent_gd_mean=float((recent.goals_for-recent.goals_against).mean()))
        else: vals.update(match_count=0,match_gf_mean=0.,match_ga_mean=0.,match_gd_mean=0.,win_rate=.33,draw_rate=.33,recent_gf_mean=0.,recent_gd_mean=0.)
        # Causal Elo from all prior matches, initialized at 1500.
        elo={}
        past=MATCH[MATCH.match_year<r.year].sort_values('match_date')
        for _,q in past.iterrows():
            hcode,acode=q.home_team_code,q.away_team_code; eh=elo.get(hcode,1500.); ea=elo.get(acode,1500.); hs,as_=q.home_team_score,q.away_team_score
            if pd.isna(hs) or pd.isna(as_): continue
            result=1. if hs>as_ else 0. if hs<as_ else .5; expected=1/(1+10**((ea-eh)/400)); k=20.
            elo[hcode]=eh+k*(result-expected); elo[acode]=ea+k*((1-result)-(1-expected))
        vals['elo']=float(elo.get(code,1500.)); base.append({'country':r.country,'team_code':code,'confederation_name':r.confederation_name,'region_name':r.region_name,'target_year':r.year,**vals})
    return pd.DataFrame(base)

def causal_features(df):
    parts=[]
    for year in sorted(df.year.unique()):
        hist=df[df.year<year].copy();cur=df[df.year==year].copy();parts.append(enhanced_history_features(hist,cur))
    return pd.concat(parts,ignore_index=True)

def test_metadata(test):
    meta=TRAIN.sort_values('year').drop_duplicates('country',keep='last')[['country','team_code','confederation_name','region_name']]
    out=test.merge(meta,on='country',how='left');out['year']=2026;return out

def evaluate(year):
    hist=TRAIN[TRAIN.year<year].copy();target=TRAIN[TRAIN.year==year].copy();feat=enhanced_history_features(hist,target);X,c=prep(feat);reg=CatBoostRegressor(iterations=300,depth=4,learning_rate=.04,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8);reg.fit(X,target.total_goals,cat_features=c,verbose=False);goals=reg.predict(X);pred=assign(goals,len(target));true=target.apply(stage,axis=1);mask=true.notna();return {'rmse_goals':float(mean_squared_error(target.total_goals,goals)**.5),'f1_weighted_stage':float(f1_score(true[mask],pred[mask],average='weighted',zero_division=0)),'f1_macro_stage':float(f1_score(true[mask],pred[mask],average='macro',zero_division=0))}

if __name__=='__main__':
 vals=[evaluate(y) for y in [2010,2014,2018,2022]];res={'validation':vals,'summary':{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}}
 train_feat=causal_features(TRAIN);X,c=prep(train_feat);reg=CatBoostRegressor(iterations=300,depth=4,learning_rate=.04,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4,l2_leaf_reg=8);reg.fit(X,TRAIN.total_goals,cat_features=c,verbose=False);tmeta=test_metadata(TEST);test_feat=enhanced_history_features(TRAIN,tmeta);Xt,_=prep(test_feat);goals=np.maximum(0,reg.predict(Xt));pred=assign(goals,len(TEST));sub=pd.DataFrame({'ID':TEST.ID,'total_goals':np.round(goals,3),'Target':pred});sub.to_csv(OUT/'enhanced_joint_practice_submission.csv',index=False);(OUT/'enhanced_joint_metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2));print(sub.Target.value_counts().to_dict())