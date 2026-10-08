"""Fine beta (src.beta_fine, 25 models) vs exact-100B alpha and numbers seen, next to the old 4-group beta.

    python -m src.beta_fine_analysis     # -> results/beta_fine/fine_vs_data.csv
"""
import sys, pandas as pd, numpy as np
from scipy import stats
sys.path.insert(0,'.')
from src.decade_metrics import FAMILY

f=pd.read_csv('results/beta_fine/step69369-seed-default/summary.csv')
a=pd.read_csv('results/corpus_alpha_datadecide/alpha_exact_25.csv')
o=pd.read_csv('results/datadecide/summary.csv')[['recipe','beta_direct_mean','beta_log_mean']].rename(columns={'beta_direct_mean':'old_direct','beta_log_mean':'old_log'})
m=f.merge(a,on='recipe').merge(o,on='recipe'); m['family']=m.recipe.map(lambda r: FAMILY.get(r,r)); m['log_numbers']=np.log(m.integer_matches)
print(len(m),'models;', m.family.value_counts().to_dict())
pd.set_option('display.width',250)
print(m[['recipe','family','layer','old_layer','old_direct','beta_coarse','beta_cont','r2_cont','beta_fine','err_fine','beta_coarse_oldlayer','beta_cont_oldlayer','beta_cont_std']].round(3).to_string(index=False))
print('\nsame layer as old:', (m.layer==m.old_layer).sum(),'/25; layers chosen:', m.layer.value_counts().sort_index().to_dict())
print('median seed std  coarse %.3f cont %.3f fine %.3f | median err coarse %.2f fine %.2f | median r2_cont %.2f'%(m.beta_coarse_std.median(),m.beta_cont_std.median(),m.beta_fine_std.median(),m.err_coarse.median(),m.err_fine.median(),m.r2_cont.median()))
B=['old_direct','old_log','beta_coarse','beta_cont','beta_fine','beta_coarse_oldlayer','beta_cont_oldlayer']
print('\ncorrelation between beta versions:'); print(m[B].corr().round(2).to_string())
X=['alpha_ols','alpha_mle','log_numbers']
def within(x,b,d=m):
    g=d.groupby('family'); return stats.pearsonr(d[x]-g[x].transform('mean'), d[b]-g[b].transform('mean'))
def loo(x,b,d=m):  # leave-one-family-out stability: min r
    return min(stats.pearsonr(d[d.family!=fa][x], d[d.family!=fa][b])[0] for fa in d.family.unique())
rows=[]
for b in B:
  for x in X:
    r,p=stats.pearsonr(m[x],m[b]); rho,_=stats.spearmanr(m[x],m[b]); rw,pw=within(x,b)
    nd=m[m.family!='dolma']; rnd,pnd=stats.pearsonr(nd[x],nd[b])
    rows.append((b,x,r,p,rho,rw,pw,rnd,pnd,loo(x,b)))
t=pd.DataFrame(rows,columns=['beta','predictor','r','p','spearman','within_r','within_p','r_noDolma','p_noDolma','min_r_dropfamily'])
print(); print(t.round(3).to_string(index=False))
print('\nwithin each family, r(alpha_ols, beta):')
for fa,g in m.groupby('family'):
  if len(g)>=4: print(f'  {fa:8s} n={len(g)} '+'  '.join(f"{b}={stats.pearsonr(g.alpha_ols,g[b])[0]:+.2f}" for b in ['old_direct','beta_coarse','beta_cont','beta_cont_oldlayer']))
# OLS beta ~ alpha + log_numbers
import numpy.linalg as la
for b in ['old_direct','beta_coarse','beta_cont']:
  Xm=np.column_stack([np.ones(len(m)),m.alpha_ols,m.log_numbers]); co,*_=la.lstsq(Xm,m[b],rcond=None); pr=Xm@co
  r2=1-((m[b]-pr)**2).sum()/((m[b]-m[b].mean())**2).sum()
  print(f'{b}: beta ~ alpha_ols + log_numbers  R2={r2:.2f}  coefs alpha {co[1]:+.3f} lognum {co[2]:+.3f}')
m.to_csv('results/beta_fine/fine_vs_data.csv',index=False)
