import json, pandas as pd, numpy as np, scipy.stats as st
R='/home/user/numberline-datadecide/'
o=pd.read_csv('per_model3.csv'); rows=[]
for rec in o.recipe:
    d=json.load(open(R+f'results/datadecide/{rec}.json')); fs=d['filter_summary']
    e=json.load(open(R+f'results/beta_fine/step69369-seed-default/{rec}.json'))
    r6={g:np.mean([1-100/e['filter'][s][str(g)]['tried'] for s in e['filter']]) for g in (0,3,6,9)}
    rows.append(dict(recipe=rec, rej01_g3=fs['3']['rejection_rate'], rej01_g4=fs['4']['rejection_rate'], rej01_max=max(v['rejection_rate'] for v in fs.values()),
       rej06_1000=r6[6], rej06_10000=r6[9], rej06_max=max(1-100/e['filter'][s][str(g)]['tried'] for s in e['filter'] for g in range(10))))
f=pd.DataFrame(rows); o=o.merge(f,on='recipe')
print(o[['recipe','z_c6','zA_c','rej01_g3','rej01_g4','rej01_max','rej06_1000','rej06_10000','rej06_max']].round(3).to_string())
f.to_csv('filter.csv',index=False)
