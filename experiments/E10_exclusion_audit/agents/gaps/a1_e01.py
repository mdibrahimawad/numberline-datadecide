# Reconstruct E01 per-seed gaps d1,d2,d3 from (beta, beta_scale, beta_log) and verify with r2_beta
import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.spacing_fit import fit_spacing_direct
R='/home/user/numberline-datadecide/results/datadecide/'
ab=pd.read_csv(R+'alpha_beta.csv')
rows=[]
for rec in ab.recipe:
    d=json.load(open(R+rec+'.json'))
    ev=d['evaluation']; L=ev['target_layer']
    for run in ev['runs']:
        b=run['beta']; s=run['beta_scale']; r=run['beta_log']**2
        A=np.array([[1,b,b*b],[0,1,2*b],[-r,0,1]],float)
        y=np.array([s*(1+b**2+b**4), s*(b+2*b**3),0.0])
        g=np.linalg.solve(A,y)
        b2,s2,r22=fit_spacing_direct(g)
        rows.append(dict(recipe=rec,layer=L,seed=run['seed'],d1=g[0],d2=g[1],d3=g[2],beta=b,beta_log=run['beta_log'],
             r2=run['r2_beta'],chk_beta=abs(b2-b),chk_r2=abs(r22-run['r2_beta'])))
df=pd.DataFrame(rows)
df['q12']=df.d2/df.d1; df['q23']=df.d3/df.d2
df.to_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/gaps/e01_gaps_per_seed.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print('max |beta refit - beta|',df.chk_beta.max(),' max |r2 refit - r2|',df.chk_r2.max(), ' min gap', df[['d1','d2','d3']].min().min())
print(df[['recipe','layer','seed','d1','d2','d3','q12','q23','beta','r2']].round(3).to_string())
