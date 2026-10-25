import json, numpy as np, pandas as pd, sys
R='/home/user/numberline-datadecide/'
ab=pd.read_csv(R+'results/datadecide/alpha_beta.csv')[['recipe','selected_layer','beta_direct_mean','beta_direct_std']]  # beta columns only
fv=pd.read_csv(R+'results/beta_fine/fine_vs_data.csv')[['recipe','layer','old_layer','beta_coarse_oldlayer','beta_cont_oldlayer','old_direct']]
rows=[]
for rec in ab.recipe:
    d=json.load(open(R+f'results/datadecide/{rec}.json'))
    L=d['selection']['selected_layer']; assert L==d['evaluation']['target_layer']
    sel=[s['layers'][str(L)]['beta_direct'] for s in d['selection']['per_seed']]
    sseeds=[s['seed'] for s in d['selection']['per_seed']]
    ev=[r['beta'] for r in d['evaluation']['runs']]; eseeds=[r['seed'] for r in d['evaluation']['runs']]
    e6=json.load(open(R+f'results/beta_fine/step69369-seed-default/{rec}.json'))
    ol=e6['report']['old_layer']; assert ol['layer']==L
    c6=[s['layers'][str(L)]['beta_coarse'] for s in e6['per_seed']]
    k6=[s['layers'][str(L)]['beta_cont'] for s in e6['per_seed']]
    e6seeds=[s['seed'] for s in e6['per_seed']]
    rows.append(dict(recipe=rec,L=L,sel=sel,ev=ev,c6=c6,k6=k6,sseeds=sseeds,eseeds=eseeds,e6seeds=e6seeds,
        c6_mean_rep=ol['beta_coarse']['mean'],c6_std_rep=ol['beta_coarse']['std'],k6_mean_rep=ol['beta_cont']['mean'],k6_std_rep=ol['beta_cont']['std'],
        ev_mean_rep=d['evaluation']['summary']['beta_mean'],ev_std_rep=d['evaluation']['summary']['beta_std']))
df=pd.DataFrame(rows).merge(ab,on='recipe').merge(fv,on='recipe')
df.to_pickle(sys.argv[1])
# consistency checks
print('seeds', set(map(tuple,df.sseeds)), set(map(tuple,df.eseeds)), set(map(tuple,df.e6seeds)))
print('eval mean vs csv', np.max(abs(df.ev.apply(np.mean)-df.beta_direct_mean)), 'std ddof?', np.max(abs(df.ev.apply(lambda x: np.std(x,ddof=0))-df.beta_direct_std)), np.max(abs(df.ev.apply(lambda x: np.std(x,ddof=1))-df.beta_direct_std)))
print('E06 coarse mean vs csv', np.max(abs(df.c6.apply(np.mean)-df.beta_coarse_oldlayer)), 'std ddof0', np.max(abs(df.c6.apply(lambda x: np.std(x))-df.c6_std_rep)), 'ddof1',np.max(abs(df.c6.apply(lambda x: np.std(x,ddof=1))-df.c6_std_rep)))
print('E06 cont mean vs csv', np.max(abs(df.k6.apply(np.mean)-df.beta_cont_oldlayer)))
print('layer match', (df.L==df.old_layer).all(), (df.L==df.selected_layer).all(), 'old_direct', np.max(abs(df.old_direct-df.beta_direct_mean)))
