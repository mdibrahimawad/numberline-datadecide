import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.beta_fine import band
from scipy.stats import chisquare
R='/home/user/numberline-datadecide/'
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
g=pd.read_csv(D+'e06_groups.csv')
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
g['dlogn_pct_of_band']=100*(g.mean_logn-g.prop_mean_logn)/(np.log10(g.hi)-np.log10(g.lo))
g['ratio_lo_hi']=g.acc_rate_low/g.acc_rate_high
print('KS p<0.01 count:',(g.ks_p<0.01).sum(),'of',len(g), ' p<0.001:',(g.ks_p<0.001).sum())
print(g[g.ks_p<0.01][['recipe','group','rej','ks_p','share_above','prop_above','ratio_lo_hi','dlogn_pct_of_band']].round(4).to_string())
b=g[g.group.isin([0,3,6,9])]
print(b.pivot(index='recipe',columns='group',values='ratio_lo_hi').round(2).to_string())
print('max |dlogn| pct of band width per model:'); print(g.groupby('recipe').dlogn_pct_of_band.apply(lambda x: x.abs().max()).round(2).sort_values().to_string())
# within band analysis for high-rejection groups: chi-square on 10 equal sub-bins, accepted counts vs tried*share
rows=[]
for r in g.recipe.unique():
    Z=np.load(R+f'results/beta_fine/step69369-seed-default/{r}.npz')
    for gr in range(10):
        T=np.concatenate([Z[f'seed{s}_targets'][Z[f'seed{s}_groups']==gr] for s in (45,46,47)])
        bb=band(gr); lo,hi=bb.start,bb.stop-1
        ints=np.arange(lo,hi+1); nb=min(10,len(ints))
        edges=np.linspace(lo,hi+1,nb+1); cnt=np.histogram(T,edges)[0]; pr=np.histogram(ints,edges)[0]/len(ints)
        rows.append(dict(recipe=r,group=gr,chi_p=chisquare(cnt,pr*cnt.sum()).pvalue,min_bin_ratio=(cnt/(pr*cnt.sum())).min()))
c=pd.DataFrame(rows); c.to_csv(D+'e06_subbin.csv',index=False)
print('chi p<0.01:',(c.chi_p<0.01).sum(), 'expected by chance ~', 0.01*len(c))
print(c[c.chi_p<0.01].round(4).to_string())
