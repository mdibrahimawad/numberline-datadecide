"""E12b: smooth part + additive digit part (experiments/E12_one_dimensional_manifold/PREREG_B.md).

Z(n) = f(n) + D(n) + E(n): f = Gaussian kernel smoother over n (bandwidth >= 1 % of the range, chosen by 5-fold CV of the
additive model), D = additive centred one-hot effects of the tens and units digits (+ hundreds for Q4), fitted by
backfitting. T2 / T3 / T4 on f, on the digit-removed data Z - D, and on fake curves / sheets at noise 0.1x, 0.3x, 1x.
Resumable: one JSON per dataset in <out>/cells/.

  python -m src.e12b_smooth_digits --data <topology_number_line>/data --out results/e12b_smooth_digits
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from src.e12_manifold import (DEPTH, KS, MODELS, SETS, _gauss_like, _orthonormal, id_mle, layer_file, pca, plateau,
                              set_numbers, standardise, t3_order, t4_spacing)

H_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
NOISE = [0.1, 0.3, 1.0]


def positions_for(sname, units_only=False):
    if units_only:
        return [0]
    return [0, 1, 2] if sname == "Q4" else [0, 1]          # 0 = units, 1 = tens, 2 = hundreds


def digit_design(n, positions, means=None):
    cols = [((n // 10 ** p) % 10)[:, None] == np.arange(10)[None, :] for p in positions]
    X = np.hstack(cols).astype(float)
    means = X.mean(0) if means is None else means
    return X - means, means


def smooth_at(Ztr, ntr, h, neval):
    W = np.exp(-0.5 * ((neval[:, None] - ntr[None, :]) / h) ** 2)
    return (W @ Ztr) / W.sum(1, keepdims=True)


def fit_additive(Z, n, h, positions, iters=30):
    X, means = digit_design(n, positions)
    D = np.zeros_like(Z)
    for _ in range(iters):
        f = smooth_at(Z - D, n, h, n)
        coef = np.linalg.lstsq(X, Z - f, rcond=None)[0]
        D = X @ coef
    f = smooth_at(Z - D, n, h, n)
    return f, D, dict(coef=coef, means=means)


def choose_h(Z, n, positions, seed=0):
    R = n.max() - n.min(); errs = {}
    for fr in H_FRACS:
        h = fr * R; e = []
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(n):
            f, D, m = fit_additive(Z[tr], n[tr], h, positions)
            Xte, _ = digit_design(n[te], positions, m["means"])
            pred = smooth_at(Z[tr] - D, n[tr], h, n[te]) + Xte @ m["coef"]
            e.append(((Z[te] - pred) ** 2).sum(1).mean())
        errs[fr] = float(np.mean(e))
    best = min(errs, key=errs.get)
    return best * R, best, errs


def shares(Z, f, D, E):
    v = lambda M: float(((M - M.mean(0)) ** 2).sum())
    tot = v(Z)
    return dict(share_f=v(f) / tot, share_digit=v(D) / tot, share_resid=v(E) / tot,
                r2_f=1 - v(Z - f) / tot, r2_f_plus_digit=1 - v(Z - f - D) / tot)


def t4_exact(f, n, seed):
    """Spacing from the exact arc length along f (sorted by n): models a + b n, a + b log n, a + b n^p."""
    o = np.argsort(n); ns = n[o]; seg = np.linalg.norm(np.diff(f[o], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]; s = (s - s.mean()) / s.std()
    grid = np.round(np.arange(0.05, 2.0001, 0.05), 2)
    fit = lambda x, y: np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
    best_p = lambda nn, ss: grid[int(np.argmin([((ss - np.c_[np.ones(len(nn)), nn ** p] @ fit(nn ** p, ss)) ** 2).mean() for p in grid]))]
    mse = {"linear": [], "log": [], "power": []}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(ns):
        for name, g in (("linear", lambda v: v), ("log", np.log)):
            c = fit(g(ns[tr]), s[tr]); mse[name].append(((s[te] - np.c_[np.ones(len(te)), g(ns[te])] @ c) ** 2).mean())
        p = best_p(ns[tr], s[tr]); c = fit(ns[tr] ** p, s[tr]); mse["power"].append(((s[te] - np.c_[np.ones(len(te)), ns[te] ** p] @ c) ** 2).mean())
    mse = {k: float(np.mean(v)) for k, v in mse.items()}
    rng = np.random.default_rng(seed); boot = []
    for _ in range(200):
        b = rng.integers(0, len(ns), len(ns)); boot.append(best_p(ns[b], s[b]))
    return dict(T4x_best=min(mse, key=mse.get), T4x_mse_linear=mse["linear"], T4x_mse_log=mse["log"], T4x_mse_power=mse["power"],
                T4x_p=float(best_p(ns, s)), T4x_p_lo=float(np.quantile(boot, 0.025)), T4x_p_hi=float(np.quantile(boot, 0.975)))


def sheet(n, f, wfrac, rng):
    x = (n - n.min()) / (n.max() - n.min()); S2 = np.c_[x, rng.uniform(0, wfrac, len(n))]; S2 -= S2.mean(0)
    S = S2 @ _orthonormal(f.shape[1], 2, rng).T
    return S * np.sqrt(((f - f.mean(0)) ** 2).sum(1).mean() / (S ** 2).sum(1).mean())


def t2(M):
    ids = id_mle(M)
    return {**{f"T2_mle_k{k}": ids[k] for k in KS}, "T2_twonn": ids["twonn"], "T2_plateau": bool(plateau(ids))}


def run_dataset(args):
    data, model, sname, d, dsid, units_only, out = args
    tag = f"{model}__{sname}__{d}__{'units' if units_only else 'main'}"
    path = os.path.join(out, "cells", tag + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    n = set_numbers(model, sname)
    Z = pca(standardise(np.load(layer_file(data, model, d))["X"].astype(np.float64)[n - 1]))
    pos = positions_for(sname, units_only)
    h, hfrac, errs = choose_h(Z, n, pos)
    f, D, _ = fit_additive(Z, n, h, pos); E = Z - f - D
    meta = dict(model=model, set=sname, depth=DEPTH[d], N=len(n), variant="units_only" if units_only else "main",
                digits="+".join({0: "units", 1: "tens", 2: "hundreds"}[p] for p in pos), h=h, h_frac=hfrac, **shares(Z, f, D, E))
    rows = []
    seed = 2000 * dsid + (7 if units_only else 0)
    for cond, M in (("f", f), ("Z_minus_digits", Z - D)):
        r = dict(meta, condition=cond, noise=np.nan, **t2(M), **t3_order(M, n, seed), **t4_spacing(M, n, seed))
        if cond == "f":
            r.update(t4_exact(f, n, seed))
        rows.append(r)
    power = []
    if not units_only:
        rng = np.random.default_rng(seed + 1)
        for c in NOISE:
            fk = {"curve": f + c * _gauss_like(E, len(n), rng), "sheet_thin": sheet(n, f, 0.1, rng) + c * _gauss_like(E, len(n), rng),
                  "sheet_fat": sheet(n, f, 0.5, rng) + c * _gauss_like(E, len(n), rng)}
            for cond, M in fk.items():
                rows.append(dict(meta, condition=cond, noise=c, **t2(M), **t3_order(M, n, seed), **t4_spacing(M, n, seed)))
            for rep in range(5):
                for cond, wf in (("curve", None), ("sheet_thin", 0.1), ("sheet_fat", 0.5)):
                    base = f if wf is None else sheet(n, f, wf, rng)
                    power.append(dict(meta, condition=cond, noise=c, rep=rep, plateau=bool(plateau(id_mle(base + c * _gauss_like(E, len(n), rng))))))
    res = dict(rows=rows, power=power)
    json.dump(res, open(path + ".tmp", "w"), default=float); os.replace(path + ".tmp", path)
    print(f"{tag} done (h = {hfrac:.2f} R, share f = {meta['share_f']:.2f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e12b_smooth_digits")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    jobs = []
    for model in MODELS:
        for sname in SETS[model]:
            for d in range(5):
                jobs.append((a.data, model, sname, d, len(jobs) + 1, False, a.out))
    for model, sname in (("qwen2.5-1.5b", "Q2"), ("pythia-1.4b", "P2")):          # secondary: units digit only
        for d in range(5):
            jobs.append((a.data, model, sname, d, 100 + len(jobs), True, a.out))
    jobs.sort(key=lambda j: j[2] != "Q4")                                            # longest first
    with Pool(a.workers) as pool:
        res = pool.map(run_dataset, jobs, chunksize=1)
    pd.DataFrame([r for x in res for r in x["rows"]]).to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame([p for x in res for p in x["power"]]).to_csv(os.path.join(a.out, "power_replicates.csv"), index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
