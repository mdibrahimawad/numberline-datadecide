import numpy as np, src.e10_explore as ex
from src.e11_tier0 import twonn
from scipy.stats import spearmanr
D="/home/user/mdibrahimawad/topology_number_line/data/"
for m,L in [("crystal",27),("crystal",8),("qwen2.5-1.5b",24),("olmo2-1b",14),("pythia-1.4b",20)]:
    X=np.load(f"{D}{m}_vectors/layer_{L:02d}.npz")["X"].astype(np.float32)
    nums=ex.FOUR[~ex.year_zone(ex.FOUR)]; V=X[nums-1].copy()
    for blk in np.unique(nums//100):
        i=nums//100==blk; V[i]-=V[i].mean(0)
    C=np.stack([V[nums%100==e].mean(0) for e in range(100)]).astype(np.float64); C-=C.mean(0)
    U,S,Vt=np.linalg.svd(C,full_matrices=False); ev=S**2/np.sum(S**2); Z=C@Vt[:3].T
    t=np.arange(100)//10; u=np.arange(100)%10
    best=lambda lab: max(abs(spearmanr(Z[:,k],lab)[0]) for k in range(3))
    print(f"{m} L{L}: PC1-3 share {ev[0]:.2f} {ev[1]:.2f} {ev[2]:.2f} (top2 {ev[:2].sum():.2f}, top10 {ev[:10].sum():.2f}); TwoNN dim {twonn(C):.1f}; |rho| tens with a top-3 PC {best(t):.2f}, units {best(u):.2f}")
