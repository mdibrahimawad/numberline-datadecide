import json, numpy as np, pandas as pd, collections
R='/home/user/numberline-datadecide/'
df=pd.read_pickle('data.pkl')
h=collections.defaultdict(collections.Counter)
for rec in df.recipe:
    d=json.load(open(R+f'results/datadecide/{rec}.json'))
    for r in d['evaluation']['runs']: h[r['seed']][r['target_hash']]+=1
    print(rec, d['evaluation']['runs'][0].keys()) if rec=='c4' else None
for s,c in h.items(): print('eval seed',s,'distinct target hashes',len(c),'most common count',c.most_common(1)[0][1])
# E06 targets equality across models
T={}
for rec in df.recipe:
    z=np.load(R+f'results/beta_fine/step69369-seed-default/{rec}.npz')
    T[rec]=z['seed45_targets']
ref=T['dolma1_7']
print('E06 seed45 fraction targets identical to dolma1_7:', {r: round(float(np.mean(t==ref)),3) for r,t in list(T.items())[:25]})
A=lambda c: np.vstack(df[c].values)
for name,X in [('E01 sel+eval',np.hstack([A('sel'),A('ev')])),('E01 eval',A('ev')),('E01 sel',A('sel')),('E06 coarse',A('c6')),('E06 cont',A('k6'))]:
    m,s=X.shape
    gm=X.mean(); rowm=X.mean(1,keepdims=True); colm=X.mean(0,keepdims=True)
    ss_seed=m*((colm-gm)**2).sum(); res=X-rowm-colm+gm; ss_res=(res**2).sum()
    df_seed=s-1; df_res=(m-1)*(s-1)
    within=((X-rowm)**2).sum()/(m*(s-1))
    F=(ss_seed/df_seed)/(ss_res/df_res)
    from scipy.stats import f
    print(f'{name}: seed means {np.round(colm.ravel(),3)}  within-model sd {np.sqrt(within):.4f}  interaction sd {np.sqrt(ss_res/df_res):.4f}  F_seed={F:.2f} p={1-f.cdf(F,df_seed,df_res):.2g}')
