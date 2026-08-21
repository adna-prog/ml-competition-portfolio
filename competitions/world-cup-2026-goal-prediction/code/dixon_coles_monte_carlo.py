from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT=Path(__file__).parents[1];RAW=ROOT/'data/raw';OUT=ROOT/'output';SEED=20260846
MATCH=pd.read_csv(RAW/'matches.csv');TEAMS=pd.read_csv(RAW/'teams.csv');TEST=pd.read_csv(RAW/'Test.csv')
MATCH['year']=pd.to_datetime(MATCH.match_date).dt.year
MATCH=MATCH.dropna(subset=['home_team_code','away_team_code','home_team_score','away_team_score'])
MATCH=MATCH[MATCH.year<=2022].copy()
all_codes=sorted(set(MATCH.home_team_code)|set(MATCH.away_team_code));idx={t:i for i,t in enumerate(all_codes)};n=len(all_codes)
hi=np.array([idx[x] for x in MATCH.home_team_code]);ai=np.array([idx[x] for x in MATCH.away_team_code]);hg=MATCH.home_team_score.to_numpy(float);ag=MATCH.away_team_score.to_numpy(float)
# Recent matches receive modestly larger weight; all weights use historical dates only.
days=(pd.Timestamp('2022-12-31')-pd.to_datetime(MATCH.match_date)).dt.days.to_numpy(float);weights=np.exp(-0.00035*np.maximum(days,0))
def tau(x,y,l1,l2,rho):
 t=np.ones(len(x));m=(x==0)&(y==0);t[m]=1-l1[m]*l2[m]*rho;m=(x==1)&(y==0);t[m]=1+l1[m]*rho;m=(x==0)&(y==1);t[m]=1+l2[m]*rho;m=(x==1)&(y==1);t[m]=1-rho;return np.clip(t,1e-8,None)
def nll(p):
 att=p[:n];de=p[n:2*n];gamma=p[2*n];rho=p[2*n+1];l1=np.clip(np.exp(att[hi]+de[ai]+gamma),.01,8);l2=np.clip(np.exp(att[ai]+de[hi]),.01,8);ll=np.log(tau(hg.astype(int),ag.astype(int),l1,l2,rho))+poisson.logpmf(hg,l1)+poisson.logpmf(ag,l2);return -np.sum(weights*ll)+2*(np.sum(att**2)+np.sum(de**2))+1000*np.sum(att)**2
x0=np.zeros(2*n+2);x0[2*n]=.1;x0[2*n+1]=-.05;bounds=[(-3,3)]*(2*n)+[(-1,2),(-1,1)]
fit=minimize(nll,x0,method='L-BFGS-B',bounds=bounds,options={'maxiter':350,'ftol':1e-8});att=fit.x[:n];de=fit.x[n:2*n];gamma=fit.x[2*n];rho=fit.x[2*n+1];att-=att.mean()
def lam(a,b):
 ia=idx.get(a);ib=idx.get(b)
 if ia is None or ib is None:return 1.2,1.2
 return float(np.clip(np.exp(att[ia]+de[ib]),.05,6)),float(np.clip(np.exp(att[ib]+de[ia]),.05,6))
# Map test names to codes.
alias={'Czechia':'CZE','Turkiye':'TUR','Cote d\'Ivoire':'CIV','Cabo Verde':'CPV','DR Congo':'COD','Curacao':'CUW','Jordan':'JOR','Uzbekistan':'UZB'}
code_by_country=dict(zip(TEAMS.team_name,TEAMS.team_code));codes=[]
for c in TEST.country: codes.append(alias.get(c,code_by_country.get(c)))
# Seeded deterministic 12 groups of 4, because no 2026 groups are supplied.
rng=np.random.default_rng(SEED);ordered=list(codes);rng.shuffle(ordered);groups=[ordered[i:i+4] for i in range(0,48,4)]
results={c:{'goals':[],'stage':[]} for c in codes};stage_order=['group','roundof32','roundof16','qf','sf','runnerup','champion']
def scoreline_probs(a,b):
 l1,l2=lam(a,b);mg=9;xx,yy=np.meshgrid(np.arange(mg),np.arange(mg),indexing='ij');mat=poisson.pmf(xx,l1)*poisson.pmf(yy,l2);corr=np.ones_like(mat,dtype=float);corr[0,0]=1-l1*l2*rho;corr[1,0]=1+l1*rho;corr[0,1]=1+l2*rho;corr[1,1]=1-rho;mat=np.clip(mat*corr,1e-12,None);return mat/mat.sum()
def sample_score(a,b):
 mat=scoreline_probs(a,b);flat=mat.ravel();k=rng.choice(len(flat),p=flat);return divmod(k,mat.shape[1])
def knockout_pair(a,b):
 l1,l2=lam(a,b);ga,gb=sample_score(a,b)
 if ga==gb:return (a,b,ga,gb) if rng.random()<l1/(l1+l2) else (b,a,ga,gb)
 return (a,b,ga,gb) if ga>gb else (b,a,ga,gb)
def one():
 res={c:{'goals':0,'stage':'group'} for c in codes}; qual=[]
 for gr in groups:
  tab={c:[0,0,0,0] for c in gr} # pts,gf,ga,gd
  for i in range(4):
   for j in range(i+1,4):
    a,b=gr[i],gr[j];ga,gb=sample_score(a,b);res[a]['goals']+=ga;res[b]['goals']+=gb;tab[a][1]+=ga;tab[a][2]+=gb;tab[b][1]+=gb;tab[b][2]+=ga;tab[a][3]+=ga-gb;tab[b][3]+=gb-ga
    if ga>gb:tab[a][0]+=3
    elif gb>ga:tab[b][0]+=3
    else:tab[a][0]+=1;tab[b][0]+=1
  rank=sorted(gr,key=lambda c:(tab[c][0],tab[c][3],tab[c][1],rng.random()),reverse=True);qual.extend(rank[:2]);groups_third=(rank[2],tab[rank[2]][0],tab[rank[2]][3],tab[rank[2]][1])
  # store thirds externally
  if 'thirds' not in locals():thirds=[]
  thirds.append(groups_third)
 thirds=sorted(thirds,key=lambda x:(x[1],x[2],x[3]),reverse=True);qual.extend([x[0] for x in thirds[:8]])
 # R32 onward random bracket, with exact stages.
 remain=qual.copy();
 for st in ['roundof32','roundof16','qf','sf']:
  winners=[];losers=[]
  for i in range(0,len(remain),2):
   w,l,ga,gb=knockout_pair(remain[i],remain[i+1]);res[remain[i]]['goals']+=ga;res[remain[i+1]]['goals']+=gb;winners.append(w);losers.append(l);res[l]['stage']=st
  remain=winners
 winner,runner,ga,gb=knockout_pair(remain[0],remain[1]);res[remain[0]]['goals']+=ga;res[remain[1]]['goals']+=gb;res[runner]['stage']='runnerup';res[winner]['stage']='champion'
 return res
for _ in range(3000):
 z=one()
 for c in codes:results[c]['goals'].append(z[c]['goals']);results[c]['stage'].append(z[c]['stage'])
from scipy.optimize import linear_sum_assignment
# Aggregate stage probabilities, then enforce the global 48-team distribution.
goals={c:float(np.mean(v['goals'])) for c,v in results.items()};stage_cols=['group','roundof32','roundof16','qf','sf','runnerup','champion'];slots=[]
for s,k in [('champion',1),('runnerup',1),('sf',2),('qf',4),('roundof16',8),('roundof32',16),('group',16)]:slots.extend([s]*k)
teams=list(codes);probs=np.zeros((len(teams),len(slots)))
for j,s in enumerate(slots):probs[:,j]=[-np.log(max(np.mean(np.array(results[c]['stage'])==s),1e-9)) for c in teams]
ri,ci=linear_sum_assignment(probs);stage_assign={teams[r]:slots[c] for r,c in zip(ri,ci)}
sub=pd.DataFrame({'ID':TEST.ID,'total_goals':[round(goals[c],3) for c in codes],'Target':[stage_assign[c] for c in codes]});sub.to_csv(OUT/'dixon_coles_monte_carlo_submission.csv',index=False)
meta={'seed':SEED,'optimizer_success':bool(fit.success),'n_historical_matches':len(MATCH),'n_simulations':3000,'groups':groups,'stage_counts':sub.Target.value_counts().to_dict(),'dixon_coles_nll':float(fit.fun)}
(OUT/'dixon_coles_monte_carlo_metrics.json').write_text(json.dumps(meta,indent=2,default=str));print(json.dumps(meta,indent=2));print(sub.head(10).to_string(index=False))
