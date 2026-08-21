from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from joint_prediction import TRAIN,TEST,OUT,prep,assign
from enhanced_joint_prediction import enhanced_history_features
from optuna_goal_tuning import causal_features
SEED=20260845
# Pareto-balanced Optuna trial 16: lower RMSE than baseline while retaining F1.
PARAMS={'iterations':200,'depth':5,'learning_rate':0.062484276547307475,'l2_leaf_reg':7.869196117031656,'random_strength':0.40075825649287056,'bagging_temperature':1.0798667916079816}
meta=TRAIN.sort_values('year').drop_duplicates('country',keep='last')[['country','team_code','confederation_name','region_name']];tm=TEST.merge(meta,on='country',how='left');tm['year']=2026
train_feat=causal_features(TRAIN);X,c=prep(train_feat);target=TRAIN.total_goals.to_numpy();m=CatBoostRegressor(**PARAMS,loss_function='RMSE',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4);m.fit(X,target,cat_features=c,verbose=False);tf=enhanced_history_features(TRAIN,tm);Xt,_=prep(tf);goals=np.maximum(0,m.predict(Xt));st=assign(goals,len(TEST));sub=pd.DataFrame({'ID':TEST.ID,'total_goals':np.round(goals,3),'Target':st});sub.to_csv(OUT/'optuna_joint_practice_submission.csv',index=False);print(sub.Target.value_counts().to_dict());print(sub.head(10).to_string(index=False))
