import numpy as np, pandas as pd, json, scipy.stats as st, sys
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.spacing_fit import fit_spacing_direct
R='/home/user/numberline-datadecide/'
df=pd.read_pickle('data.pkl'); out=pd.read_csv('per_model2.csv')
rows=[]
for i,r in df.iterrows():
    z=np.load(R+f'results/beta_fine/step69369-seed-default/{r.recipe}.npz'); L=r.L
    rec=[];rec4=[];recb=[]
    for s in (45,46,47):
        t=z[f'seed{s}_targets']; g=z[f'seed{s}_groups']; P=z[f'seed{s}_pc_scores'][L].astype(np.float64)
        pc1=P[:,0]
        if np.corrcoef(np.log10(t),pc1)[0,1]<0: pc1=-pc1
        m=[pc1[g==k].mean() for k in (0,3,6,9)]; rec.append(fit_spacing_direct(np.abs(np.diff(m)))[0])
        sel=np.isin(g,(0,3,6,9)); Q=P[sel]-P[sel].mean(0); u,sv,vt=np.linalg.svd(Q,full_matrices=False); p=Q@vt[0]
        if np.corrcoef(np.log10(t[sel]),p)[0,1]<0: p=-p
        gg=g[sel]; m4=[p[gg==k].mean() for k in (0,3,6,9)]; rec4.append(fit_spacing_direct(np.abs(np.diff(m4)))[0])
        # emulate E01 k=40: bootstrap-free subsample first 40 per coarse group in stored order
        idx=np.concatenate([np.where(g==k)[0][:40] for k in (0,3,6,9)])
        mb=[pc1[idx][g[idx]==k].mean() for k in (0,3,6,9)]; recb.append(fit_spacing_direct(np.abs(np.diff(mb)))[0])
    rows.append(dict(recipe=r.recipe,c6_recomp=np.mean(rec),c6_4pca=np.mean(rec4),c6_k40=np.mean(recb),c6_4pca_seeds=rec4,c6_k40_seeds=recb))
x=pd.DataFrame(rows); o=out.merge(x,on='recipe')
print('recompute check max abs diff',np.max(abs(o.c6_recomp-o.c6)).round(4))
def inter(X):
    m,s=X.shape; res=X-X.mean(1,keepdims=True)-X.mean(0,keepdims=True)+X.mean(); return np.sqrt((res**2).sum()/((m-1)*(s-1)))
X4=np.vstack(x.c6_4pca_seeds.values); Xk=np.vstack(x.c6_k40_seeds.values)
print('interaction sd 4pca',inter(X4).round(4),'k40 subsample',inter(Xk).round(4))
sE=0.0365
for col,X in [('c6',None),('c6_4pca',X4),('c6_k40',Xk)]:
    y=o[col]; ts=st.theilslopes(y,o.ev); res=y-(ts[1]+ts[0]*o.ev); sy=0.0178 if X is None else inter(X)
    se=np.sqrt(sy**2/3+ts[0]**2*sE**2/3); zz=res/se
    print(f'{col}: r={st.pearsonr(o.ev,y)[0]:.3f} slope {ts[0]:.3f} resid sd {res.std(ddof=2):.4f} noise {se:.4f}  flagged', list(o.recipe[abs(zz)>3]), 'z:', dict(zip(o.recipe[abs(zz)>2.5], np.round(zz[abs(zz)>2.5],2))))
    o['z_'+col]=zz
print(o[['recipe','ev','c6','c6_4pca','z_c6','z_c6_4pca']].round(3).to_string())
o.to_csv('per_model3.csv',index=False)
