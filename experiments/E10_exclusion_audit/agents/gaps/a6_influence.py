# Influence of each E01 gap on beta: replace E01 gap i by the E06 replicate of that gap, rescaled to E01 units
# via the other two gaps (scale k = sum_{j!=i} d_j / sum_{j!=i} e_j), refit beta_direct; delta_i = beta_new - beta_E01.
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.spacing_fit import fit_spacing_direct
P='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/gaps/'
s=pd.read_csv(P+'gap_summary.csv'); ab=pd.read_csv('/home/user/numberline-datadecide/results/datadecide/alpha_beta.csv').set_index('recipe')
# expected E01 gap ratios under an exactly logarithmic line (beta=1): differences of mean log10(n) per E01 group
grp={1:range(1,40),2:range(80,120),3:range(980,1020),4:range(9980,10020)}
ml=[np.mean(np.log10(list(grp[g]))) for g in (1,2,3,4)]
print('E01 group mean log10 n:',np.round(ml,3),' log-spans of gaps:',np.round(np.diff(ml),3))
E06g={0:range(9,12),3:range(90,111),6:range(900,1101),9:range(9000,11001)}
ml6=[np.mean(np.log10(list(E06g[g]))) for g in (0,3,6,9)]; print('E06 band mean log10:',np.round(ml6,3),np.round(np.diff(ml6),3))
b_log_line,_,_=fit_spacing_direct(np.diff(ml)); print('beta_direct of a perfectly log line with E01 groups:',round(b_log_line,3))
rows=[]
for _,r in s.iterrows():
    d=np.array([r.d1,r.d2,r.d3]); e=np.array([r.e1,r.e2,r.e3]); b0,_,_=fit_spacing_direct(d); out=dict(recipe=r.recipe,beta_seedmean_gaps=b0)
    for i in range(3):
        o=[j for j in range(3) if j!=i]; k=d[o].sum()/e[o].sum(); dd=d.copy(); dd[i]=k*e[i]
        out[f'dbeta_{i+1}']=fit_spacing_direct(dd)[0]-b0
    out['beta_std']=ab.loc[r.recipe,'beta_direct_std']; rows.append(out)
df=pd.DataFrame(rows); df['max_abs']=df[['dbeta_1','dbeta_2','dbeta_3']].abs().max(1)
df['which']=df[['dbeta_1','dbeta_2','dbeta_3']].abs().idxmax(1)
pd.set_option('display.width',200); print(df.round(3).to_string()); print('median beta_std',df.beta_std.median().round(3))
df.to_csv(P+'gap_influence.csv',index=False)
