"""Is the year structure 'clustered by decade'? ratio = mean distance between different decades / within same decade."""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
VEC="/home/user/mdibrahimawad/topology_number_line/data/crystal_vectors/layer_{:02d}.npz"
def ratio(X,idx):
    D=squareform(pdist(X[idx-1])); dec=idx//10; same=(dec[:,None]==dec[None,:]); off=~np.eye(len(idx),dtype=bool)
    return D[same&off].mean()/D[~same].mean()
rows=[]
for L in [8,16,24,27,32]:
    X=np.load(VEC.format(L))["X"].astype(np.float32)
    for name,idx in {"years 1900-1999":np.arange(1900,2000),"years 1926-2025":np.arange(1926,2026)}.items():
        m=len(idx); null=np.array([ratio(X,np.arange(s,s+m)) for s in range(2100,10000-m,50) if s%10==idx[0]%10])
        g=ratio(X,idx); rows.append(dict(layer=L,test=name,within_over_between_years=round(g,3),null_mean=round(null.mean(),3),
                                         z=round((g-null.mean())/null.std(),1),null_below=f"{(null<g).sum()}/{len(null)}"))
print(pd.DataFrame(rows).to_string(index=False))
