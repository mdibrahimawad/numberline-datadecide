import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.beta_fine import band
R='/home/user/numberline-datadecide/results/beta_fine/step69369-seed-default/'
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
recipes=list(pd.read_csv('/home/user/numberline-datadecide/results/datadecide/alpha_beta.csv').recipe)
ref='dolma1_7-no-flan'
J=json.load(open(R+ref+'.json'))
print('ref tried/accepted', {s:sum(J['filter'][s][g]['tried'] for g in J['filter'][s]) for s in J['filter']})
Zr=np.load(R+ref+'.npz')
rows=[]
for r in recipes:
    Z=np.load(R+r+'.npz'); Jm=json.load(open(R+r+'.json'))
    for s in (45,46,47):
        t0=Zr[f'seed{s}_targets']; g0=Zr[f'seed{s}_groups']; t=Z[f'seed{s}_targets']; g=Z[f'seed{s}_groups']
        assert (g==g0).all()
        for i in range(len(t)):
            rows.append(dict(recipe=r,seed=s,group=int(g0[i]),n0=int(t0[i]),first_ok=bool(t[i]==t0[i])))
    # check: number of first-try rejections <= rejections logged
df=pd.DataFrame(rows); df.to_csv(D+'e06_firsttry.csv',index=False)
# consistency check vs json rejection counts
chk=[]
for r in recipes:
    Jm=json.load(open(R+r+'.json'))
    rejtot=sum(Jm['filter'][s][g]['tried']-Jm['filter'][s][g]['accepted'] for s in Jm['filter'] for g in Jm['filter'][s])
    chk.append((r,int((~df[df.recipe==r].first_ok).sum()),rejtot))
print(pd.DataFrame(chk,columns=['recipe','first_try_rejects','all_rejects']).to_string())
