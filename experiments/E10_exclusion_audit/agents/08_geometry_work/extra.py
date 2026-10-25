import numpy as np, pandas as pd, sys
from scipy.optimize import minimize_scalar
R="/home/user/numberline-datadecide"; D=f"{R}/results/beta_fine/step69369-seed-default"
rec=pd.read_csv(f"{R}/results/datadecide/alpha_beta.csv")[["recipe","selected_layer"]]
C=[0,3,6,9]
def nd(t): return np.array([len(str(int(x))) for x in t])
def contfit(u,y):
    def sse(lb):
        z=np.exp(lb*u) if abs(lb)>1e-6 else u; X=np.column_stack([np.ones_like(u),z]); c,*_=np.linalg.lstsq(X,y,rcond=None); return ((y-X@c)**2).sum(),X@c
    r=minimize_scalar(lambda l:sse(l)[0],bounds=(-6,6),method="bounded"); return sse(r.x)[1]
def ratio(u,y,dg):
    ov=np.polyfit(u,y,1)[0]; cl=[c for c in (2,3,4)]
    xs=np.concatenate([u[dg==c]-u[dg==c].mean() for c in cl]); ys=np.concatenate([y[dg==c]-y[dg==c].mean() for c in cl])
    return (xs@ys/(xs@xs))/ov
rows=[]
for recipe,sl in rec.itertuples(index=False):
    z=np.load(f"{D}/{recipe}.npz")
    for tag,L in (("sel",int(sl)),("L8",8)):
        acc=[]
        for s in (45,46,47):
            S=z[f"seed{s}_pc_scores"][L].astype(float); t=z[f"seed{s}_targets"]; g=z[f"seed{s}_groups"]; u=np.log10(t)-1; dg=nd(t)
            M=np.array([S[g==k].mean(0) for k in range(10)]); n=np.array([(g==k).sum() for k in range(10)])
            pc1g=np.abs(np.diff(M[C,0]))
            off=[]
            for a,b in zip(C[:-1],C[1:]):
                sq=((M[b,1:]-M[a,1:])**2).sum()-np.trace(np.cov(S[g==a][:,1:].T))/n[a]-np.trace(np.cov(S[g==b][:,1:].T))/n[b]
                off.append(np.sqrt(max(sq,0)))
            yhat=contfit(u,S[:,0])
            acc.append(list(pc1g)+off+[ratio(u,S[:,0],dg), ratio(u,yhat,dg)])
        m=np.median(np.array(acc),0)
        rows.append(dict(recipe=recipe,which=tag,layer=L,g1=m[0],g2=m[1],g3=m[2],off1=m[3],off2=m[4],off3=m[5],ratio_obs=m[6],ratio_smooth_null=m[7]))
df=pd.DataFrame(rows); df.to_csv(sys.argv[1],index=False)
pd.set_option('display.width',250)
df['recipe']=df.recipe.str.replace('dclm-baseline','dclm').str.replace('falcon-and-cc','fcc').str.replace('dolma1.7','d17')
for c in ['off1','off2','off3']: df[c+'/g']=df[c]/df['g'+c[-1]]
print(df.round(3).to_string(index=False))
