import json, numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
R='/home/user/numberline-datadecide/'
ab=pd.read_csv(R+'results/datadecide/alpha_beta.csv',usecols=['recipe','selected_layer','rho_mean','ev_mean'])
SEEDS=[45,46,47]; L=17
rows=[]; layerrows=[]
for _,r in ab.iterrows():
    rec=r.recipe
    d=np.load(R+f'results/beta_fine/step69369-seed-default/{rec}.npz')
    j=json.load(open(R+f'results/beta_fine/step69369-seed-default/{rec}.json'))
    e1=json.load(open(R+f'results/datadecide/{rec}.json'))
    L6=j['selection']['selected_layer']; L1=int(r.selected_layer)
    # per layer/seed stats
    ev=np.zeros((3,L,5)); rk=np.full((3,L,5),np.nan); r2_1=np.full((3,L),np.nan); r2_5=np.full((3,L),np.nan)
    mono=np.full((3,L),np.nan); gm=np.full((3,L,10),np.nan)
    for si,s in enumerate(SEEDS):
        t=d[f'seed{s}_targets']; g=d[f'seed{s}_groups']; sc=d[f'seed{s}_pc_scores'].astype(np.float64); ev[si]=d[f'seed{s}_pc_evr']
        u=np.log10(t)
        for l in range(1,L):
            for k in range(5): rk[si,l,k]=abs(spearmanr(t,sc[l,:,k])[0])
            X1=np.c_[np.ones(len(u)),sc[l,:,0]]; X5=np.c_[np.ones(len(u)),sc[l,:,:5]]
            for X,arr in ((X1,r2_1),(X5,r2_5)):
                b=np.linalg.lstsq(X,u,rcond=None)[0]; res=u-X@b; arr[si,l]=1-res.var()/u.var()
            m=np.array([sc[l,g==q,0].mean() for q in range(10)]); gm[si,l]=m
            dm=np.diff(m[[0,3,6,9]]); mono[si,l]=(dm>0).all()
    for l in range(1,L):
        layerrows.append(dict(recipe=rec,layer=l,ev1=np.median(ev[:,l,0]),ev2=np.median(ev[:,l,1]),
            ev1_min=ev[:,l,0].min(),ev1_max=ev[:,l,0].max(),
            rho1=np.median(rk[:,l,0]),rho2=np.median(rk[:,l,1]),rhomax_other=np.median(rk[:,l,1:].max(1)),
            argmax_pc=[int(np.argmax(rk[si,l])+1) for si in range(3)],
            r2_1=np.median(r2_1[:,l]),r2_5=np.median(r2_5[:,l]),ratio12=np.median(ev[:,l,0]/ev[:,l,1])))
    # E01 per-seed (selection 42-44) ev/rho per layer, signed rho
    e1ev=np.array([[ps['layers'][str(l)]['ev'] for l in range(1,L)] for ps in e1['selection']['per_seed']])
    e1rho=np.array([[ps['layers'][str(l)]['rho'] for l in range(1,L)] for ps in e1['selection']['per_seed']])
    evrun=[ru['explained_variance'] for ru in e1['evaluation']['runs']]; rhorun=[ru['rho'] for ru in e1['evaluation']['runs']]
    def at(l):
        cc=np.corrcoef(gm[:,l,:])  # cross-seed corr of 10 group means
        return dict(ev1=np.median(ev[:,l,0]),ev2=np.median(ev[:,l,1]),ev1_cv=ev[:,l,0].std(ddof=1)/ev[:,l,0].mean(),
                    ratio12_min=(ev[:,l,0]/ev[:,l,1]).min(),rho1=np.median(rk[:,l,0]),rhoPC2=np.median(rk[:,l,1]),
                    rho_other_max=rk[:,l,1:].max(),argmax_pc=''.join(str(int(np.argmax(rk[si,l])+1)) for si in range(3)),
                    r2_1=np.median(r2_1[:,l]),r2_5=np.median(r2_5[:,l]),share=np.median(r2_1[:,l]/r2_5[:,l]),
                    gm_corr_min=cc[np.triu_indices(3,1)].min(),mono4=int(mono[:,l].sum()))
    a1=at(L1); a6=at(L6)
    row=dict(recipe=rec,L_E01=L1,L_E06=L6)
    row.update({f'{k}@E01L':v for k,v in a1.items()}); row.update({f'{k}@E06L':v for k,v in a6.items()})
    row.update(e01_ev_sel_seeds=list(np.round(e1ev[:,L1-1],3)),e01_rho_sel_seeds=list(np.round(e1rho[:,L1-1],3)),
               e01_ev_eval=list(np.round(evrun,3)),e01_rho_eval=list(np.round(rhorun,3)),
               e01_neg_rho_any_layer=int((e1rho<0).sum()),
               e01_ev1_max_all_layers=float(np.nanmax(e1ev)),e06_ev1_max_all_layers=float(ev[:,1:,0].max()),
               e06_ev1_argmax_layer=int(np.argmax(np.median(ev[:,1:,0],0))+1),
               e01_ev_cv_max=float(np.nanmax(e1ev.std(0,ddof=1)/e1ev.mean(0))),
               e06_ev_cv_max=float(np.nanmax(ev[:,1:,0].std(0,ddof=1)/ev[:,1:,0].mean(0))),
               n_layers_numberaxis_not_pc1=int(sum(np.median(rk[:,l,1:].max(1))>np.median(rk[:,l,0]) for l in range(1,L))))
    rows.append(row)
df=pd.DataFrame(rows); ld=pd.DataFrame(layerrows)
W='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/14_work/'
df.to_pickle(W+'models.pkl'); ld.to_pickle(W+'layers.pkl'); df.to_csv(W+'models.csv',index=False); ld.to_csv(W+'layers.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
print(df.round(3).to_string())
