import json, numpy as np, pandas as pd
R='/home/user/numberline-datadecide/results/'
ab=pd.read_csv(R+'datadecide/alpha_beta.csv')[['recipe','alpha_ols','alpha_mle','alpha_r2']]
cf=pd.read_csv(R+'datadecide/count_features.csv')
cf=cf.drop(columns=['old_direct','beta_resid'])  # beta columns never used
conf=pd.DataFrame(json.load(open(R+'datadecide/confounders.json'))['recipes'])[['recipe','number_density','code_share','math_share']]
d=ab.merge(cf[['recipe','family']],on='recipe').merge(conf,on='recipe')
for c in ['number_density','code_share','math_share']: d[c]=d[c].astype(float)
n=len(d); x=d.alpha_ols.values
print('n',n,'mean',x.mean(),'sd',x.std(ddof=1),'median',np.median(x))
mad=np.median(abs(x-np.median(x)))*1.4826
d['z']=(x-x.mean())/x.std(ddof=1); d['robust_z']=(x-np.median(x))/mad
Sxx=((x-x.mean())**2).sum()
d['h_ols']=1/n+(x-x.mean())**2/Sxx
xm=d.alpha_mle.values; d['h_mle']=1/n+(xm-xm.mean())**2/((xm-xm.mean())**2).sum()
d['share_Sxx']=(x-x.mean())**2/Sxx
# studentized deleted residual of alpha vs rest (outlier test): t_i = (x_i - mean_{-i}) / (sd_{-i} sqrt(1+1/(n-1)))
t=[]
for i in range(n):
    o=np.delete(x,i); t.append((x[i]-o.mean())/(o.std(ddof=1)*np.sqrt(1+1/(n-1))))
d['t_del']=t
from scipy import stats
crit=stats.t.ppf(1-0.05/(2*n),n-2); print('Bonferroni crit |t_del|',crit)
print('hat threshold 2p/n',4/n,'3p/n',6/n)
# Grubbs
G=abs(d.z).max(); tc=stats.t.ppf(1-0.05/(2*n),n-2); Gc=(n-1)/np.sqrt(n)*np.sqrt(tc**2/(n-2+tc**2)); print('Grubbs G',G,'crit',Gc)
print('Shapiro',stats.shapiro(x))
s=d.sort_values('alpha_ols').reset_index(drop=True)
s['gap_to_next']=s.alpha_ols.diff().shift(-1)
print(s[['recipe','family','alpha_ols','gap_to_next']].to_string())
g=np.diff(np.sort(x)); print('median gap',np.median(g),'mean gap',g.mean(),'max gap',g.max())
# family structure
fam=d.groupby('family').alpha_ols.agg(['count','mean','min','max','std']); print(fam)
grand=x.mean(); SSb=sum(len(v)*(v.mean()-grand)**2 for _,v in d.groupby('family').alpha_ols); print('eta2 family',SSb/Sxx,'within share',1-SSb/Sxx)
# one-way anova
print(stats.f_oneway(*[v.values for _,v in d.groupby('family').alpha_ols]))
# overlap region: number of families whose [min,max] contains x_i (excluding own family -> count other families)
rng=fam[['min','max']]
def cover(v,excl=None):
    return [f for f,r in rng.iterrows() if r['min']<=v<=r['max'] and f!=excl]
d['other_fam_cover']=[len(cover(v,f)) for v,f in zip(x,d.family)]
d['fam_cover']=[len(cover(v)) for v in x]
# within-family demeaned alpha leverage (family fixed effects design)
d['a_within']=x-d.groupby('family').alpha_ols.transform('mean')
Sw=(d.a_within**2).sum(); d['h_within_share']=d.a_within**2/Sw
# multivariate leverage alpha + confounders
X=np.column_stack([np.ones(n),x,d.code_share,d.math_share,d.number_density])
H=X@np.linalg.pinv(X.T@X)@X.T; d['h_multi']=np.diag(H); print('multi thr 2p/n',2*5/n)
# collinearity
from numpy.linalg import lstsq
def r2(y,Z):
    Z=np.column_stack([np.ones(len(y)),Z]); b=lstsq(Z,y,rcond=None)[0]; e=y-Z@b; return 1-(e@e)/((y-y.mean())@(y-y.mean()))
for c in ['code_share','math_share','number_density']:
    print(c,'r',np.corrcoef(x,d[c])[0,1],'spearman',stats.spearmanr(x,d[c])[0])
r2c=r2(x,d[['code_share','math_share','number_density']].values); print('R2 alpha~code+math+density',r2c,'VIF',1/(1-r2c))
D=pd.get_dummies(d.family,drop_first=True).values.astype(float)
r2f=r2(x,D); r2fc=r2(x,np.column_stack([D,d[['code_share','math_share','number_density']].values]))
print('R2 alpha~family',r2f,'VIF',1/(1-r2f),'R2 family+content',r2fc,'VIF',1/(1-r2fc))
print('alpha_ols vs alpha_mle r',np.corrcoef(x,xm)[0,1],'spearman',stats.spearmanr(x,xm)[0])
# count_features non-beta cols
feats=[c for c in cf.columns if c.startswith('F')]
cf2=cf.set_index('recipe').loc[d.recipe]
rows=[]
for c in feats:
    v=cf2[c].values.astype(float)
    if np.std(v)==0: continue
    rows.append((c,np.corrcoef(x,v)[0,1],stats.spearmanr(x,v)[0]))
print(pd.DataFrame(rows,columns=['feat','r','rho']).sort_values('r',key=abs,ascending=False).to_string())
d.to_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/s15/design_table.csv',index=False)
pd.set_option('display.width',250)
print(d[['recipe','family','alpha_ols','alpha_mle','alpha_r2','z','robust_z','t_del','h_ols','h_mle','share_Sxx','a_within','h_within_share','h_multi','fam_cover','other_fam_cover','code_share','math_share','number_density']].round(4).to_string())
