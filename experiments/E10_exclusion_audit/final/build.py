"""Per-model table for the final exclusion analysis (E01, 3 gaps). Outcome-blind construction."""
import json, numpy as np, pandas as pd
R='/home/user/numberline-datadecide/results/datadecide/'
a=pd.read_csv(R+'alpha_beta.csv')
conf=pd.DataFrame(json.load(open(R+'confounders.json'))['recipes'])[['recipe','code_share','math_share']]
fam=pd.read_csv('/home/user/numberline-datadecide/results/beta_fine/fine_vs_data.csv')[['recipe','family','beta_coarse_oldlayer']]
rows=[]
for r in a.itertuples():
    j=json.load(open(R+f'{r.recipe}.json')); s=j['selection']; L=int(s['selected_layer'])
    ls={int(k):v for k,v in s['layer_scores'].items()}; ld={int(k):v for k,v in s['layers_direct'].items()}
    qmed={k:v['q_median'] for k,v in ls.items()}
    qps=np.array([ls[k]['q_per_seed'] for k in sorted(ls)]); keys=sorted(ls)
    seedwin={keys[i] for i in np.argmax(qps,axis=0)}
    T={L}|{k for k in qmed if qmed[L]-qmed[k]<=0.01}|seedwin
    bsel=ld[L]['beta_mean']
    bT=[ld[k]['beta_mean'] for k in T]
    rows.append(dict(recipe=r.recipe,layer=L,beta_eval=r.beta_direct_mean,beta_sel=bsel,
        beta_6=(r.beta_direct_mean+bsel)/2, tieset=sorted(T), beta_tie_range=max(bT)-min(bT),
        beta_tie_mean=np.mean(bT)))
d=pd.DataFrame(rows).merge(a[['recipe','alpha_ols']],on='recipe').merge(conf,on='recipe').merge(fam,on='recipe')
d.to_pickle('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/final/table.pkl')
print(d.drop(columns=['alpha_ols']).round(3).to_string(index=False))
