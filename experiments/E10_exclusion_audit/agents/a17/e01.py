import json, numpy as np, pandas as pd
D='/home/user/numberline-datadecide/results/datadecide/'
ab=pd.read_csv(D+'alpha_beta.csv')[['recipe','selected_layer','beta_direct_mean','beta_direct_std','beta_log_mean','beta_log_std']]
rows=[];hashes={}
for rec in ab.recipe:
    j=json.load(open(D+rec+'.json')); L=j['selection']['selected_layer']
    ev=[float(r['beta']) for r in j['evaluation']['runs']]
    evl=[float(r['beta_log']) for r in j['evaluation']['runs']]
    for r in j['evaluation']['runs']: hashes.setdefault(r['seed'],[]).append(r['target_hash'])
    sel=[ps['layers'][str(L)]['beta_direct'] for ps in j['selection']['per_seed']]
    rows.append(dict(recipe=rec,L=L,e45=ev[0],e46=ev[1],e47=ev[2],s42=sel[0],s43=sel[1],s44=sel[2],
        ev_sd=np.std(ev,ddof=1),six_mean=np.mean(ev+sel),six_sd=np.std(ev+sel,ddof=1),evlog_sd=np.std(evl,ddof=1)))
d=pd.DataFrame(rows).merge(ab,on='recipe')
print('hash identical across models per seed:',{k:len(set(v)) for k,v in hashes.items()})
print('check std matches csv:',np.allclose(d.ev_sd,d.beta_direct_std) , np.allclose(np.std(d[['e45','e46','e47']].values,axis=1,ddof=0),d.beta_direct_std))
d.to_csv('e01_seeds.csv',index=False)
pd.set_option('display.width',250)
print(d.round(3).to_string())
