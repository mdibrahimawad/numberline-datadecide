import sys,json,random,numpy as np,pandas as pd
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.beta_fine import make_prompt,N_GROUPS
from scipy import stats
R='/home/user/numberline-datadecide/results/beta_fine/step69369-seed-default/'
fv=pd.read_csv('/home/user/numberline-datadecide/results/beta_fine/fine_vs_data.csv')
rows=[]
for rec in fv.recipe:
    z=np.load(R+rec+'.npz')
    for seed in (45,46,47):
        rng=random.Random(seed); init=[]
        for g in range(N_GROUPS):
            for _ in range(100): init.append((g,)+make_prompt(g,rng))
        t=z[f'seed{seed}_targets']
        for i,(g,p,n) in enumerate(init):
            ctx=[int(x.split('=')[0]) for x in p.split(',')[:-1]]
            rows.append(dict(recipe=rec,seed=seed,g=g,rej=int(n!=t[i]),mlog=np.mean(np.log10(np.array(ctx)+1)),last4=int(ctx[-1]>=1000),n4=sum(c>=1000 for c in ctx)))
d=pd.DataFrame(rows)
d.to_csv('first_try.csv',index=False)
# within recipe x group, rejected vs accepted context
out=[]
for rec,s in d.groupby('recipe'):
    nr=s.rej.sum()
    if nr<10: out.append(dict(recipe=rec,n_rej_slots=nr)); continue
    s=s.copy(); s['mlog_c']=s.mlog-s.groupby(['seed','g']).mlog.transform('mean')
    a=s[s.rej==0].mlog_c; r=s[s.rej==1].mlog_c
    out.append(dict(recipe=rec,n_rej_slots=nr,dmlog_rej_minus_acc=r.mean()-a.mean(),p=stats.mannwhitneyu(r,a).pvalue,
                    worst_group=int(s.groupby('g').rej.mean().idxmax()),worst_rate=s.groupby('g').rej.mean().max()))
print(pd.DataFrame(out).round(4).to_string())
