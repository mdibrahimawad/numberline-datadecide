import numpy as np, pandas as pd, sys
sys.argv=[sys.argv[0],"x"]; exec(open("pass1.py").read().split("rows=[]")[0])   # reuse feats, cnt
F=["spread_raw","step_raw","H0_total","H0_max","H1_n","H1_total","H1_entropy"]
tests={"years 1900-2025":np.arange(1900,2026),"years 1900-1999":np.arange(1900,2000),"years 1926-2025":np.arange(1926,2026)}
out=[]
for L in LAYERS:
    X=np.load(VEC.format(L))["X"].astype(np.float32)
    gscale=np.median(pdist(X[np.random.default_rng(0).choice(10000,800,replace=False)]))
    for name,idx in tests.items():
        m=len(idx)
        starts=[s for s in range(2100,10000-m,50)]           # all same-length windows outside 1500-2099
        g=feats(X,idx); null=pd.DataFrame([feats(X,np.arange(s,s+m)) for s in starts])
        for f in ["spread_raw","step_raw"]: g[f]/=gscale; null[f]/=gscale
        for f in F:
            z=(g[f]-null[f].mean())/null[f].std(); pct=(null[f]<g[f]).mean()
            out.append(dict(layer=L,test=name,metric=f,value=g[f],null_mean=null[f].mean(),z=z,pct_below=pct,n_null=len(null)))
R=pd.DataFrame(out); R.to_csv("pass1b.csv",index=False)
P=R.pivot_table(index=["test","metric"],columns="layer",values="z").round(1)
print("z-score of the year set vs ~150 same-length windows of ordinary 4-digit numbers (|z|>3 notable)")
print(P.to_string()); print("n_null =",R.n_null.iloc[0])
