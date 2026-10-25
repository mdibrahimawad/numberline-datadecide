# E06 at E01's selected layer: signed coarse gaps per seed (10,100,1000,10000 bands, +-10%),
# from (a) the stored group means (PCA over all 10 groups), (b) PCA refit on the 4 coarse groups
# within the stored top-5 PC subspace; and digit-boundary steps inside each coarse band.
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
R='/home/user/numberline-datadecide/results/'
ab=pd.read_csv(R+'datadecide/alpha_beta.csv')
out=[]; steps=[]
for rec, L in zip(ab.recipe, ab.selected_layer):
    j=json.load(open(R+f'beta_fine/step69369-seed-default/{rec}.json'))
    z=np.load(R+f'beta_fine/step69369-seed-default/{rec}.npz')
    for ps in j['per_seed']:
        s=ps['seed']; gm=ps['layers'][str(L)]['group_means']
        m=np.array([gm[str(g)] for g in (0,3,6,9)]); ga=np.diff(m)
        t=z[f'seed{s}_targets']; g=z[f'seed{s}_groups']; X=z[f'seed{s}_pc_scores'][L].astype(np.float64)
        pc1=X[:,0].copy()
        if np.corrcoef(np.log10(t),pc1)[0,1]<0: pc1=-pc1
        # check reproduction of stored means
        m_chk=np.array([pc1[g==k].mean() for k in (0,3,6,9)])
        sel=np.isin(g,(0,3,6,9))
        p=PCA(1).fit(X[sel]); y=p.transform(X[sel])[:,0]
        if np.corrcoef(np.log10(t[sel]),y)[0,1]<0: y=-y
        mb=np.array([y[g[sel]==k].mean() for k in (0,3,6,9)]); gb=np.diff(mb)
        out.append(dict(recipe=rec,layer=L,seed=s,a1=ga[0],a2=ga[1],a3=ga[2],b1=gb[0],b2=gb[1],b3=gb[2],
                        chk=np.abs(m_chk-m).max()))
        # digit-boundary step within each coarse band: PC1 ~ a + b*log10 n + c*[n>=10^k]
        for k,band in zip((1,2,3,4),(0,3,6,9)):
            mm=g==band; tt=t[mm]; yy=pc1[mm]
            D=np.column_stack([np.ones(mm.sum()),np.log10(tt),(tt>=10**k).astype(float)])
            coef,res,*_=np.linalg.lstsq(D,yy,rcond=None)
            r=yy-D@coef; sig=np.sqrt(r@r/(len(yy)-3)); cov=sig**2*np.linalg.inv(D.T@D)
            steps.append(dict(recipe=rec,seed=s,boundary=10**k,n_lo=int((tt<10**k).sum()),n_hi=int((tt>=10**k).sum()),
                              step=coef[2],se=np.sqrt(cov[2,2]),slope=coef[1]))
df=pd.DataFrame(out); st=pd.DataFrame(steps)
P='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/gaps/'
df.to_csv(P+'e06_coarse_gaps_oldlayer.csv',index=False); st.to_csv(P+'e06_digit_steps.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print('max mean-reproduction error', df.chk.max())
print('min signed gap (stored means)',df[['a1','a2','a3']].min().round(3).to_dict(),' (refit PCA)',df[['b1','b2','b3']].min().round(3).to_dict())
print(df.drop(columns='chk').round(2).to_string())
