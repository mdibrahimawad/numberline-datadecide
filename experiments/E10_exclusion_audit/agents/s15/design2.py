import numpy as np, pandas as pd
d=pd.read_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/s15/design_table.csv')
x=d.alpha_ols.values; n=len(x); f=d.family
nf=f.map(f.value_counts())
# FE design: y ~ family dummies + alpha ; leverage = 1/n_f + a_w^2/Sw ; p=7
Sw=(d.a_within**2).sum(); d['h_FE']=1/nf+d.a_within**2/Sw; print('FE p=7 thr',2*7/n)
print('within SS by family share:',(d.a_within**2).groupby(f).sum()/Sw)
# balanced weights 1/n_f
w=1/nf.values; xb=(w*x).sum()/w.sum(); d['h_bal']=w/w.sum()+w*(x-xb)**2/(w*(x-xb)**2).sum()
print('balanced: effective n',w.sum()**2/(w**2).sum())
# family means design
fm=d.groupby('family').alpha_ols.mean(); hm=1/len(fm)+(fm-fm.mean())**2/((fm-fm.mean())**2).sum(); print('family-mean leverage (thr 2p/k=%.2f)'%(4/len(fm))); print(hm.round(3))
fm5=fm.drop('c4'); print('without c4', (1/5+(fm5-fm5.mean())**2/((fm5-fm5.mean())**2).sum()).round(3))
# hull rule
mins=d.groupby('family').alpha_ols.min().sort_values(); maxs=d.groupby('family').alpha_ols.max().sort_values(ascending=False)
L=mins.iloc[1]; U=maxs.iloc[1]; print('hull L',L,mins.index[1],'U',U,maxs.index[1])
d['keep_hull']=(x>=L-1e-12)&(x<=U+1e-12)
d['keep_strict2']=d.fam_cover>=2
for k in ['keep_hull','keep_strict2']:
    print(k,'dropped',list(d.recipe[~d[k]]))
    s=d[d[k]]; print('  kept n',len(s),'families',s.family.value_counts().to_dict(),'sd',s.alpha_ols.std(ddof=1),'eta2', 1-((s.alpha_ols-s.groupby('family').alpha_ols.transform('mean'))**2).sum()/((s.alpha_ols-s.alpha_ols.mean())**2).sum())
    xs=s.alpha_ols.values; h=1/len(xs)+(xs-xs.mean())**2/((xs-xs.mean())**2).sum(); print('  max h',h.max().round(3),'thr',round(4/len(xs),3),'flag',list(s.recipe[h>4/len(xs)]))
# LOO: sd change
full=x.std(ddof=1)
d['sd_loo_pct']=[100*(np.delete(x,i).std(ddof=1)/full-1) for i in range(n)]
# top cumulative share
s=d.sort_values('share_Sxx',ascending=False); print(s[['recipe','share_Sxx']].assign(cum=s.share_Sxx.cumsum()).head(8).round(3).to_string())
# dolma-with-code group
grp=d.recipe.isin(['dolma1_7','dolma1_7-no-reddit','dolma1_7-no-flan','dolma1_6plus']); print('4 code-dolma share Sxx',d.share_Sxx[grp].sum(), 'range of others',x[~grp].min(),x[~grp].max(),'sd others',x[~grp].std(ddof=1))
# code share>0 recipes
print(d.loc[d.code_share>0,['recipe','code_share','math_share']])
print('corr code,math',np.corrcoef(d.code_share,d.math_share)[0,1])
# alpha_mle leverage flags
print('h_mle>0.16',list(d.recipe[d.h_mle>0.16]),'h_mle>0.24',list(d.recipe[d.h_mle>0.24]))
d.to_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/s15/design_table2.csv',index=False)
print(d[['recipe','h_FE','h_bal','sd_loo_pct','keep_hull','keep_strict2']].round(3).to_string())
