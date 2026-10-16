"""Per-number local intrinsic dimension (Levina-Bickel MLE, k = 20 neighbours among all 4-digit numbers) vs the
count left after block + tens digit + last digit (ordinary numbers, no special patterns). Exploratory."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import src.e10_explore as ex
from src.e10_graded import MODELS, DEPTH, layer_files, levels
DATA = "/home/user/mdibrahimawad/topology_number_line/data"
cnt = ex.load_counts(); mt = levels(cnt, 10); n = mt["n"].values; key = mt["key"].values; lev = mt["level"].values
rows = []
for m in MODELS:
    for d, f in enumerate(layer_files(DATA, m)):
        X = np.load(f)["X"].astype(np.float32); F = X[ex.FOUR - 1]; sq = (F ** 2).sum(1); lid = []
        for i in range(0, len(n), 1000):
            q = X[n[i:i + 1000] - 1]; d2 = (q ** 2).sum(1)[:, None] + sq[None, :] - 2 * q @ F.T
            T = np.sqrt(np.maximum(np.sort(d2, 1)[:, 1:21], 1e-12))
            lid.append(1 / np.mean(np.log(T[:, -1:] / T[:, :-1]), 1))
        lid = np.concatenate(lid); nrm = np.linalg.norm(X[n - 1], axis=1)
        r = spearmanr(key, lid)
        # partial: residualise ranks of LID and frequency on the rank of vector length
        rk = lambda v: pd.Series(v).rank().values
        res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
        rp = spearmanr(res(rk(lid), rk(nrm)), res(rk(key), rk(nrm)))
        rows.append(dict(model=m, depth=DEPTH[d], rho=r.correlation, p=r.pvalue, rho_without_length=rp.correlation,
                         p_without_length=rp.pvalue, lid_rarest=lid[lev == 0].mean(), lid_most_frequent=lid[lev == 9].mean(),
                         lid_mean=lid.mean()))
        print(m, DEPTH[d], f"rho {r.correlation:+.3f} (p {r.pvalue:.1e}) | without length {rp.correlation:+.3f} | LID rarest {lid[lev==0].mean():.1f} vs frequent {lid[lev==9].mean():.1f}", flush=True)
pd.DataFrame(rows).to_csv("results/e11_tier0/LID_per_number.csv", index=False)
