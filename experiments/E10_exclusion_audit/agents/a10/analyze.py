import numpy as np, pandas as pd, scipy.stats as st
df=pd.read_pickle('data.pkl')
A=lambda c: np.vstack(df[c].values)
S,E,C,K=A('sel'),A('ev'),A('c6'),A('k6')
def pooled(X): return np.sqrt(np.mean(X.var(axis=1,ddof=1)))
sp={n:pooled(X) for n,X in dict(sel=S,ev=E,c6=C,k6=K).items()}
print('pooled sd (ddof1, 50 df):',{k:round(v,4) for k,v in sp.items()})
sp01=np.sqrt((np.sum(S.var(1,ddof=1))+np.sum(E.var(1,ddof=1)))/50)
print('pooled E01 sd (sel+eval, 100df)',round(sp01,4))
for n,X in dict(sel=S,ev=E,c6=C,k6=K).items():
    print(n,'bartlett',st.bartlett(*X).pvalue.round(4),'levene',st.levene(*X).pvalue.round(4), 'spearman(sd,mean)',st.spearmanr(X.std(1,ddof=1),X.mean(1)))
# relative (log-scale) noise
for n,X in dict(ev=E,c6=C).items(): print(n,'pooled sd of log beta',round(pooled(np.log(X)),4))
out=pd.DataFrame({'recipe':df.recipe,'L':df.L,'sel':S.mean(1),'sel_sd':S.std(1,ddof=1),'ev':E.mean(1),'ev_sd':E.std(1,ddof=1),'c6':C.mean(1),'c6_sd':C.std(1,ddof=1),'k6':K.mean(1),'k6_sd':K.std(1,ddof=1)})
# A: sel vs eval
out['dA']=out.sel-out.ev
out['zA']=out.dA/(sp01*np.sqrt(2/3))
out['zA_own']=out.dA/np.sqrt(out.sel_sd**2/3+out.ev_sd**2/3)
print('A mean diff',out.dA.mean().round(4),'sd',out.dA.std().round(4),'expected sd',(sp01*np.sqrt(2/3)).round(4),'t-test',st.ttest_1samp(out.dA,0))
print('A sum z^2',(out.zA**2).sum().round(1),'df 25 p',1-st.chi2.cdf((out.zA**2).sum(),25))
# B: eval vs c6
se_B=np.sqrt(sp['ev']**2/3+sp['c6']**2/3)
out['dB_raw']=out.c6-out.ev
off=np.median(out.dB_raw); print('B raw diff median',off.round(4),'mean',out.dB_raw.mean().round(4),'sd',out.dB_raw.std().round(4),'expected sd',se_B.round(4))
out['zB_off']=(out.dB_raw-off)/se_B
ts=st.theilslopes(out.c6,out.ev); print('B theil-sen',ts[:2])
out['resB']=out.c6-(ts[1]+ts[0]*out.ev)
# noise of residual: var(c6)+b^2 var(ev)
seB2=np.sqrt(sp['c6']**2/3+ts[0]**2*sp['ev']**2/3)
out['zB_lin']=out.resB/seB2
print('B pearson',st.pearsonr(out.ev,out.c6)[0].round(3),'spearman',st.spearmanr(out.ev,out.c6)[0].round(3))
for z in ['zB_off','zB_lin']: print(z,'sum z^2',(out[z]**2).sum().round(1),'MAD-sd of z',(1.4826*np.median(abs(out[z]-np.median(out[z])))).round(2))
# excess (protocol x model) sd tau estimated by method of moments on linear residuals
tau2=max(0,out.resB.var(ddof=2)-seB2**2); print('B excess tau',np.sqrt(tau2).round(4),'resid sd',out.resB.std(ddof=2).round(4),'noise',seB2.round(4))
# log-ratio version
out['lrB']=np.log(out.c6/out.ev); se_lr=np.sqrt(pooled(np.log(E))**2/3+pooled(np.log(C))**2/3)
out['zB_log']=(out.lrB-np.median(out.lrB))/se_lr
print('B log ratio median',np.median(out.lrB).round(4),'sd',out.lrB.std().round(4),'expected',se_lr.round(4))
# C: eval vs cont
tsC=st.theilslopes(out.k6,out.ev); print('C theil-sen',tsC[:2],'pearson',st.pearsonr(out.ev,out.k6)[0].round(3),'spearman',st.spearmanr(out.ev,out.k6)[0].round(3))
out['resC']=out.k6-(tsC[1]+tsC[0]*out.ev); seC=np.sqrt(sp['k6']**2/3+tsC[0]**2*sp['ev']**2/3)
out['zC_lin']=out.resC/seC
print('C resid sd',out.resC.std(ddof=2).round(4),'noise',seC.round(4),'tau',np.sqrt(max(0,out.resC.var(ddof=2)-seC**2)).round(4))
# D: c6 vs k6 within E06 (same prompts; diagnostic of within-E06 consistency)
tsD=st.theilslopes(out.k6,out.c6); out['resD']=out.k6-(tsD[1]+tsD[0]*out.c6)
print('D c6 vs k6 pearson',st.pearsonr(out.c6,out.k6)[0].round(3),'resid sd',out.resD.std(ddof=2).round(4))
# 6-seed E01 vs c6
out['e01_6']=(out.sel+out.ev)/2
ts6=st.theilslopes(out.c6,out.e01_6); se6=np.sqrt(sp['c6']**2/3+ts6[0]**2*sp01**2/6)
out['zB6_lin']=(out.c6-(ts6[1]+ts6[0]*out.e01_6))/se6
# rank shifts
out['rank_ev']=out.ev.rank(ascending=False).astype(int); out['rank_c6']=out.c6.rank(ascending=False).astype(int); out['rank_sel']=out.sel.rank(ascending=False).astype(int)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(out[['recipe','L','sel','sel_sd','ev','ev_sd','c6','c6_sd','k6','k6_sd','dA','zA','zA_own','dB_raw','zB_off','zB_lin','zB_log','zB6_lin','zC_lin','rank_sel','rank_ev','rank_c6']].round(3).to_string())
out.to_csv('per_model.csv',index=False)
