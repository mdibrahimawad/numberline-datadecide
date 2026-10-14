"""Exploratory pass 1 (stage A, discovery data = Crystal final checkpoint, one prompt per number).
Groups vs magnitude/digit-matched 'twins'; PH (Rips, Euclidean, full 4096-d) + geometry."""
import numpy as np, pandas as pd, json, sys
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
VEC="/home/user/mdibrahimawad/topology_number_line/data/crystal_vectors/layer_{:02d}.npz"
CNT="/home/user/numberline-datadecide/results/corpus_alpha_full/slimpajama-627b_reupload/counts_0_to_10000.csv"
cnt=pd.read_csv(CNT).set_index("number")["count"].reindex(range(10001)).fillna(0).values
LAYERS=[8,16,24,27,32]
yearzone=lambda n:(n>=1500)&(n<=2099)

def sets():
    S={}
    yrs=np.arange(1900,2026); S["years"]=[yrs]+[yrs+s for s in range(2000,8000,1000)]
    base=np.array([n for n in range(1000,10000) if n%50==0 and not yearzone(n)])
    offs=[3,7,11,17,23,29,33,41]
    S["round"]=[base]+[base+k for k in offs]
    return S

def feats(X, idx):
    P=X[idx-1]; D=squareform(pdist(P)); med=np.median(D[np.triu_indices(len(idx),1)])
    # resolution: distance n -> n+1 (n+1 taken from full data), relative to layer's global scale
    res=np.linalg.norm(X[idx]-X[idx-1],axis=1)   # X[idx] is number idx+1
    dg=ripser(D/med,distance_matrix=True,maxdim=1)["dgms"]
    h0=dg[0][np.isfinite(dg[0][:,1]),1]; h1=dg[1]; l1=h1[:,1]-h1[:,0]
    p=l1/l1.sum() if l1.sum()>0 else np.array([1.0])
    return dict(spread_raw=med, step_raw=np.median(res),
                H0_total=h0.sum()/len(idx), H0_max=h0.max(),
                H1_n=len(l1), H1_total=l1.sum(), H1_max=l1.max() if len(l1) else 0,
                H1_entropy=float(-(p*np.log(p)).sum()) if len(l1) else 0)

rows=[]
for L in LAYERS:
    X=np.load(VEC.format(L))["X"].astype(np.float32)
    gscale=np.median(pdist(X[np.random.default_rng(0).choice(10000,800,replace=False)]))
    for name,group in sets().items():
        for j,idx in enumerate(group):
            f=feats(X,idx); f["spread_raw"]/=gscale; f["step_raw"]/=gscale
            rows.append(dict(layer=L,family=name,role="group" if j==0 else "twin",set=f"{idx[0]}..",
                             median_count=float(np.median(cnt[idx])),**f))
df=pd.DataFrame(rows); df.to_csv(sys.argv[1],index=False); print("saved",len(df))
