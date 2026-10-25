import numpy as np, pandas as pd, sys, random, re
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.beta_fine import band, make_prompt, N_GROUPS
from scipy.stats import fisher_exact, mannwhitneyu
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
df=pd.read_csv(D+'e06_firsttry.csv')
# reconstruct initial prompts
P={}
for s in (45,46,47):
    rng=random.Random(s); L=[]
    for g in range(N_GROUPS):
        for _ in range(100): L.append(make_prompt(g,rng))
    P[s]=L
df['idx']=df.groupby(['recipe','seed']).cumcount()
df['prompt']=[P[s][i][0] for s,i in zip(df.seed,df.idx)]
assert all(P[s][i][1]==n for s,i,n in zip(df.seed,df.idx,df.n0)); print('prompts reconstructed OK')
def feats(p,n):
    ctx=[c.split('=')[0] for c in p.split(',')[:-1]]; ns=str(n)
    return dict(ndig=len(ns), rep_adj=bool(re.search(r'(\d)\1',ns)), ctx_contains=any(ns in c for c in ctx), ctx_prefix=any(c.startswith(ns) for c in ctx),
                last_ctx_digits=len(ctx[-1]))
F=pd.DataFrame([feats(p,n) for p,n in zip(df.prompt,df.n0)]); df=pd.concat([df,F],axis=1)
df['rej']=~df.first_ok
hi=df[df.recipe.isin(['fineweb-pro','dclm-baseline-qc-fw-3p','dclm-baseline-qc-7p-fw3','c4','fineweb-edu','dclm-baseline-qc-fw-10p','dclm-baseline-qc-10p','dclm-baseline'])]
print('first-try rejection rate by digit count (8 highest-rejection models pooled):'); print(hi.groupby('ndig').rej.agg(['mean','size']).round(3))
print('by group x ndig:'); print(hi.groupby(['group','ndig']).rej.agg(['mean','size']).round(3))
for f in ['rep_adj','ctx_contains','ctx_prefix']:
    print(f, hi.groupby(f).rej.agg(['mean','size']).round(3).to_dict())
# is rejection number-specific? same initial target across models: agreement of rejection across models per slot
pv=df.pivot_table(index=['seed','idx'],columns='recipe',values='rej',aggfunc='first')
hm=pv[[c for c in pv.columns if c in hi.recipe.unique()]]
print('slots rejected by >=k of 8 high-rej models:',{k:int((hm.sum(1)>=k).sum()) for k in range(1,9)})
# chance expectation of >=4 under independence
rates=hm.mean(0).values
sim=np.random.default_rng(0).random((20000,8))<rates
print('expected count >=4 under independence (per 3000 slots):', round(((sim.sum(1)>=4).mean())*3000,1))
# edge concentration: per model, per group: mean position u in band of rejected vs all; shift in accepted mean log n (first-try sample) as fraction of 1/3-decade spacing
rows=[]
for (r,g),x in df.groupby(['recipe','group']):
    u=np.log10(x.n0.values); acc=~x.rej.values
    shift=(u[acc].mean()-u.mean())/(1/3)
    p=mannwhitneyu(u[x.rej.values],u[acc]).pvalue if x.rej.sum()>=5 else np.nan
    rows.append(dict(recipe=r,group=g,n_rej=int(x.rej.sum()),rej=x.rej.mean(),shift_frac_spacing=shift,mw_p=p))
E=pd.DataFrame(rows); E.to_csv(D+'e06_edge.csv',index=False)
print('max |shift| (fraction of 1/3-decade group spacing) per model:')
print(E.assign(a=E.shift_frac_spacing.abs()).groupby('recipe').a.max().round(4).sort_values().to_string())
print(E[(E.mw_p<0.01)].round(4).to_string())
df.drop(columns=['prompt']).to_csv(D+'e06_firsttry_feats.csv',index=False)
# examples of numbers repeatedly rejected
print(hi[hi.rej].groupby('n0').size().sort_values(ascending=False).head(15).to_dict())
