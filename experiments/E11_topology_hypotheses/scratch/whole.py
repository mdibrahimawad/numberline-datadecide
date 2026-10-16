import numpy as np, sys, time
from ripser import ripser
from scipy.spatial.distance import pdist
from src.e11_tier0 import gaussian_like
D="/home/user/mdibrahimawad/topology_number_line/data/"
def summ(P, n1=1000, n2=300):
    s=np.median(pdist(P[np.random.default_rng(0).choice(len(P),1000,replace=False)]))
    r=ripser(P/s, maxdim=1, n_perm=n1); h0=r["dgms"][0]; h0=np.sort(h0[np.isfinite(h0[:,1]),1])[::-1]; h1=r["dgms"][1]
    l1=np.sort(h1[:,1]-h1[:,0])[::-1]
    r2=ripser(P/s, maxdim=2, n_perm=n2); h2=r2["dgms"][2]; l2=np.sort(h2[:,1]-h2[:,0])[::-1] if len(h2) else np.zeros(1)
    return h0[:4], l1[:3], l2[:2], len(h1)
for m,L in [(a,int(b)) for a,b in (x.split(":") for x in sys.argv[1:])]:
    X=np.load(f"{D}{m}_vectors/layer_{L:02d}.npz")["X"].astype(np.float64); t=time.time()
    h0,l1,l2,n=summ(X); g0,g1,g2,gn=summ(gaussian_like(X,np.random.default_rng(1)))
    print(f"{m} L{L}: biggest H0 merges {np.round(h0,2)} (noise {np.round(g0,2)}) | longest loops {np.round(l1,2)} (noise {np.round(g1,2)}), n loops {n} (noise {gn}) | longest voids {np.round(l2,2)} (noise {np.round(g2,2)})  [{time.time()-t:.0f}s]", flush=True)
