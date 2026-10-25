# Digit-boundary jump inside each E06 decade band at E01's layer: mean PC1(n>=10^k) - mean PC1(n<10^k),
# minus the jump expected from a smooth log-linear trend whose slope = the mean of the adjacent coarse gaps (per decade).
import json, numpy as np, pandas as pd
R='/home/user/numberline-datadecide/results/'
P='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/gaps/'
ab=pd.read_csv(R+'datadecide/alpha_beta.csv'); e6=pd.read_csv(P+'e06_coarse_gaps_oldlayer.csv')
rows=[]
for rec,L in zip(ab.recipe,ab.selected_layer):
    z=np.load(R+f'beta_fine/step69369-seed-default/{rec}.npz')
    for s in (45,46,47):
        t=z[f'seed{s}_targets']; g=z[f'seed{s}_groups']; pc1=z[f'seed{s}_pc_scores'][L][:,0].astype(float)
        if np.corrcoef(np.log10(t),pc1)[0,1]<0: pc1=-pc1
        gg=e6[(e6.recipe==rec)&(e6.seed==s)][['a1','a2','a3']].values[0]
        adj={1:gg[0],2:(gg[0]+gg[1])/2,3:(gg[1]+gg[2])/2,4:gg[2]}
        for k,band in zip((1,2,3,4),(0,3,6,9)):
            m=g==band; lo=m&(t<10**k); hi=m&(t>=10**k)
            raw=pc1[hi].mean()-pc1[lo].mean()
            dlog=np.log10(t[hi]).mean()-np.log10(t[lo]).mean()
            se=np.sqrt(pc1[hi].var(ddof=1)/hi.sum()+pc1[lo].var(ddof=1)/lo.sum())
            rows.append(dict(recipe=rec,seed=s,boundary=10**k,raw=raw,expected=adj[k]*dlog,excess=raw-adj[k]*dlog,se=se,
                             within_sd=pc1[m].std(),gap_adj=adj[k]))
df=pd.DataFrame(rows); df.to_csv(P+'e06_digit_jumps.csv',index=False)
agg=df.groupby(['recipe','boundary']).agg(excess=('excess','mean'),se=('se','mean'),gap=('gap_adj','mean'),sd=('within_sd','mean')).reset_index()
agg['z']=agg.excess/(agg.se/np.sqrt(3)); agg['frac_of_gap']=agg.excess/agg.gap
w=agg.pivot(index='recipe',columns='boundary',values='frac_of_gap').round(2)
wz=agg.pivot(index='recipe',columns='boundary',values='z').round(1)
pd.set_option('display.width',250)
print('excess jump / adjacent decade gap'); print(w.to_string()); print('z (3-seed mean)'); print(wz.to_string())
print('median frac by boundary', w.median().round(3).to_dict())
