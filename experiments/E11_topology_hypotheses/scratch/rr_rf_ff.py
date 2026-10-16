"""Rare-rare, rare-frequent and frequent-frequent distances inside each block of 100 (block + digit matched thirds,
no special-pattern numbers), and whether rare numbers' nearest neighbours are rare or frequent. Exploratory."""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, cdist
import src.e10_explore as ex
from src.e10_graded import DEPTH, layer_files
DATA = "/home/user/mdibrahimawad/topology_number_line/data"
mt = ex.matched_tiers(ex.load_counts(), "block_digits_plain")
rows = []
for m in ["crystal", "openllama-3b", "qwen2.5-1.5b", "pythia-1.4b", "olmo2-1b"]:
    for d, f in enumerate(layer_files(DATA, m)):
        X = np.load(f)["X"].astype(np.float32)
        rs = np.random.default_rng(0).choice(ex.FOUR, 1000, replace=False); s = np.median(pdist(X[rs - 1]))
        rr, ff, rf, nnf_r, nnf_f, nr, nf = [], [], [], [], [], [], []
        for b, g in mt.groupby("block"):
            R = g.n[g.tier == 0].values; F = g.n[g.tier == 2].values
            A, B = X[R - 1], X[F - 1]
            rr.append(pdist(A).mean() / s); ff.append(pdist(B).mean() / s); rf.append(cdist(A, B).mean() / s)
            P = np.vstack([A, B]); lab = np.r_[np.zeros(len(R)), np.ones(len(F))]
            D = cdist(P, P); np.fill_diagonal(D, np.inf); nn = np.argsort(D, 1)[:, :5]
            share_f = lab[nn].mean(1)                                   # share of 5 nearest that are frequent
            nnf_r.append(share_f[lab == 0].mean()); nnf_f.append(share_f[lab == 1].mean())
            nr.append(np.linalg.norm(A, axis=1).mean()); nf.append(np.linalg.norm(B, axis=1).mean())
        rows.append(dict(model=m, depth=DEPTH[d], RR=np.mean(rr), RF=np.mean(rf), FF=np.mean(ff),
                         rare_nn_frequent=np.mean(nnf_r), freq_nn_frequent=np.mean(nnf_f), length_ratio=np.mean(nr) / np.mean(nf)))
T = pd.DataFrame(rows); T.to_csv("results/e11_tier0/RR_RF_FF_distances.csv", index=False)
print(T.round(3).to_string(index=False))
