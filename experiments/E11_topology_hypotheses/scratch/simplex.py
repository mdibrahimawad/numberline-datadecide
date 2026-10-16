import numpy as np, src.e10_explore as ex
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
D="/home/user/mdibrahimawad/topology_number_line/data/"
for m,L in [("crystal",27),("crystal",8),("qwen2.5-1.5b",24),("openllama-3b",22),("olmo2-1b",14),("pythia-1.4b",20)]:
    X=np.load(f"{D}{m}_vectors/layer_{L:02d}.npz")["X"].astype(np.float32)
    nums=ex.FOUR[~ex.year_zone(ex.FOUR)]; V=X[nums-1].copy()
    for blk in np.unique(nums//100):
        i=nums//100==blk; V[i]-=V[i].mean(0)
    out=[]
    for name,lab in (("units",nums%10),("tens",(nums//10)%10)):
        C=np.stack([V[lab==k].mean(0) for k in range(10)]).astype(np.float64); d=pdist(C)
        gap=np.abs(np.subtract.outer(np.arange(10),np.arange(10)))[np.triu_indices(10,1)]
        ev=np.linalg.svd(C-C.mean(0),compute_uv=False)**2; ev/=ev.sum()
        out.append(f"{name}: dist spread (sd/mean) {d.std()/d.mean():.2f}, rho(dist,|a-b|) {spearmanr(d,gap)[0]:+.2f}, top PC {ev[0]:.2f}")
    print(m,L,' | '.join(out))
