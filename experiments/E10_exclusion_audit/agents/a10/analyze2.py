import numpy as np, pandas as pd, scipy.stats as st
df=pd.read_pickle('data.pkl'); out=pd.read_csv('per_model.csv')
A=lambda c: np.vstack(df[c].values)
def inter(X):
    m,s=X.shape; res=X-X.mean(1,keepdims=True)-X.mean(0,keepdims=True)+X.mean()
    return res, np.sqrt((res**2).sum()/((m-1)*(s-1)))
S,E,C,K=A('sel'),A('ev'),A('c6'),A('k6')
r6,s6=inter(np.hstack([S,E])); rS,sS=inter(S); rE,sE=inter(E); rC,sC=inter(C); rK,sK=inter(K)
print('interaction sds: E01 6seed %.4f sel %.4f eval %.4f E06c %.4f E06k %.4f'%(s6,sS,sE,sC,sK))
print('F sel vs eval', (sE/sS)**2, 1-st.f.cdf((sE/sS)**2,48,48))
print('per-seed residual sd (6-seed model):',np.round(np.sqrt((r6**2).sum(0)/24*6/5),4))
# per-model interaction sd and relation with beta
pm=np.sqrt((r6**2).sum(1)/5*25/24)
print('spearman(per-model interaction sd, beta)',st.spearmanr(pm,out.ev))
print('per-model interaction sd max', out.recipe[np.argmax(pm)], pm.max().round(3), 'median',np.median(pm).round(3))
# A
dA=out.sel-out.ev; dAc=dA-dA.mean()
seA=s6*np.sqrt(2/3); seA2=np.sqrt(sS**2/3+sE**2/3)
out['zA_c']=dAc/seA; out['zA_c2']=dAc/seA2
print('A: mean shift %.4f (= seed-set mean diff %.4f); centred diff sd %.4f vs expected %.4f / %.4f'%(dA.mean(),S.mean()-E.mean(),dAc.std(ddof=1),seA,seA2))
print('A chi2',(out.zA_c**2).sum().round(1),'df24 p',(1-st.chi2.cdf((out.zA_c**2).sum(),24)).round(4))
# B
for nm,ycol,sy in [('B',C,sC),('C',K,sK)]:
    y=ycol.mean(1)
    for xnm,x,sx,n in [('ev',E.mean(1),sE,3),('e01_6',np.hstack([S,E]).mean(1),s6,6)]:
        d=y-x; off=np.median(d); se_off=np.sqrt(sy**2/3+sx**2/n)
        ts=st.theilslopes(y,x); res=y-(ts[1]+ts[0]*x); se_lin=np.sqrt(sy**2/3+ts[0]**2*sx**2/n)
        z_off=(d-off)/se_off; z_lin=res/se_lin
        tau=np.sqrt(max(0,res.var(ddof=2)-se_lin**2))
        print(f'{nm} vs {xnm}: slope {ts[0]:.3f} icpt {ts[1]:.3f}; offset {off:.4f}; resid sd lin {res.std(ddof=2):.4f} noise {se_lin:.4f} tau {tau:.4f}; chi2 lin {np.sum(z_lin**2):.1f} (df23)  n|z|>3 off {np.sum(abs(z_off)>3)} lin {np.sum(abs(z_lin)>3)}')
        out[f'z{nm}_{xnm}_off']=z_off; out[f'z{nm}_{xnm}_lin']=z_lin
        if xnm=='ev': out[f'res{nm}']=res
# robust variant: z using the MAD of residuals (empirical spread) to flag outliers relative to the population of models
for c in ['zB_ev_lin','zC_ev_lin']:
    r=out[c]; out[c+'_robust']=(r-np.median(r))/(1.4826*np.median(abs(r-np.median(r))))
pd.set_option('display.width',250)
cols=['recipe','L','sel','ev','c6','k6','zA_c','zA_c2','zB_ev_off','zB_ev_lin','zB_e01_6_lin','zB_ev_lin_robust','zC_ev_lin','zC_e01_6_lin','zC_ev_lin_robust']
print(out[cols].round(2).to_string())
out.to_csv('per_model2.csv',index=False)
