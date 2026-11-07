"""E12c: held-out R^2 of smooth models of Z(n) (experiments/E12_one_dimensional_manifold/PREREG_C.md).

Models: curve (kernel smoother over n), curve + digits (additive one-hot tens/units [+hundreds for Q4]), 2-D surface over
(n, n mod 10) (product kernel, circular in the last digit), shuffled-n control. Splits: random 80/20 (3 repeats) and a
held-out block. Standardisation and PCA are fitted on the training numbers only. Resumable (one JSON per dataset).

  python -m src.e12c_heldout_r2 --data <topology_number_line>/data --out results/e12c_heldout_r2
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
from sklearn.model_selection import KFold

from src.e12_manifold import DEPTH, MODELS, SETS, layer_file, set_numbers
from src.e12b_smooth_digits import digit_design, positions_for

HN = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
HU = [0.5, 1, 2, 5]
BLOCKS = {"Q2": (70, 79), "P2": (70, 79), "Q3": (700, 799), "P3": (700, 799), "P1": (700, 799), "Q4": (7000, 7999)}


def prep(Xtr, Xte, D=50):
    mu = Xtr.mean(0); sd = Xtr.std(0); keep = sd > 1e-8
    A = (Xtr[:, keep] - mu[keep]) / sd[keep]; B = (Xte[:, keep] - mu[keep]) / sd[keep]
    D = min(D, len(Xtr) - 1)
    m = A.mean(0); Vt = np.linalg.svd(A - m, full_matrices=False)[2][:D]
    return (A - m) @ Vt.T, (B - m) @ Vt.T


def kernel_pred(Ztr, ntr, nev, h, utr=None, uev=None, hu=None):
    L = -0.5 * ((nev[:, None] - ntr[None, :]) / h) ** 2
    if hu is not None:
        du = np.abs(uev[:, None] - utr[None, :]); du = np.minimum(du, 10 - du)
        L = L - 0.5 * (du / hu) ** 2
    W = np.exp(L - L.max(1, keepdims=True))                          # log-space normalisation, never all zero
    return (W @ Ztr) / W.sum(1, keepdims=True)


def fit_cd(Ztr, ntr, h, pos, iters=15):
    X, means = digit_design(ntr, pos); D = np.zeros_like(Ztr)
    for _ in range(iters):
        f = kernel_pred(Ztr - D, ntr, ntr, h)
        coef = np.linalg.lstsq(X, Ztr - f, rcond=None)[0]; D = X @ coef
    return dict(base=Ztr - D, coef=coef, means=means)


def pred_cd(m, ntr, nev, h, pos):
    Xe, _ = digit_design(nev, pos, m["means"])
    return kernel_pred(m["base"], ntr, nev, h) + Xe @ m["coef"]


def predictor(model, Ztr, ntr, nev, h, hu, pos):
    if model == "curve":
        return kernel_pred(Ztr, ntr, nev, h)
    if model == "curve_digits":
        return pred_cd(fit_cd(Ztr, ntr, h, pos), ntr, nev, h, pos)
    if model == "surface2d":
        return kernel_pred(Ztr, ntr, nev, h, ntr % 10, nev % 10, hu)
    raise ValueError(model)


def choose(model, Ztr, ntr, pos, seed):
    R = max(ntr.max() - ntr.min(), 1); grid = [(max(1.0, f * R), hu) for f in HN for hu in (HU if model == "surface2d" else [None])]
    best = None
    for h, hu in grid:
        err = 0.0
        for a, b in KFold(5, shuffle=True, random_state=seed).split(ntr):
            err += ((Ztr[b] - predictor(model, Ztr[a], ntr[a], ntr[b], h, hu, pos)) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, h, hu)
    return best[1], best[2]


def r2(Zte, P, mean_tr):
    return float(1 - ((Zte - P) ** 2).sum() / ((Zte - mean_tr) ** 2).sum())


def evaluate(X, n, tr, te, pos, seed):
    Ztr, Zte = prep(X[tr], X[te]); ntr, nte = n[tr], n[te]; mtr = Ztr.mean(0); out = {}
    for model in ("curve", "curve_digits", "surface2d"):
        h, hu = choose(model, Ztr, ntr, pos, seed)
        out[f"R2_{model}"] = r2(Zte, predictor(model, Ztr, ntr, nte, h, hu, pos), mtr)
        out[f"h_{model}"] = h / max(n.max() - n.min(), 1); out[f"hu_{model}"] = hu
    rng = np.random.default_rng(seed + 77); sh = []
    for _ in range(5):
        ns = rng.permutation(ntr); h, _ = choose("curve", Ztr, ns, pos, seed)
        sh.append(r2(Zte, kernel_pred(Ztr, ns, nte, h), mtr))
    out["R2_shuffled"] = float(np.mean(sh)); out["R2_shuffled_max"] = float(np.max(sh))
    return out


def run_dataset(args):
    data, model, sname, d, out = args
    path = os.path.join(out, "cells", f"{model}__{sname}__{d}.json")
    if os.path.exists(path):
        return json.load(open(path))
    n = set_numbers(model, sname); X = np.load(layer_file(data, model, d))["X"].astype(np.float64)[n - 1]
    pos = positions_for(sname); rows = []
    for rep in range(3):
        te = np.random.default_rng(rep).random(len(n)) < 0.2
        rows.append(dict(model=model, set=sname, depth=DEPTH[d], split="random", rep=rep, **evaluate(X, n, ~te, te, pos, rep)))
    lo, hi = BLOCKS[sname]; te = (n >= lo) & (n <= hi)
    rows.append(dict(model=model, set=sname, depth=DEPTH[d], split=f"block {lo}-{hi}", rep=0, n_test=int(te.sum()),
                     **evaluate(X, n, ~te, te, pos, 10)))
    json.dump(rows, open(path + ".tmp", "w"), default=float); os.replace(path + ".tmp", path)
    print(f"{model} {sname} {DEPTH[d]} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e12c_heldout_r2")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    jobs = [(a.data, m, s, d, a.out) for m in MODELS for s in SETS[m] for d in range(5)]
    jobs.sort(key=lambda j: j[2] != "Q4")
    with Pool(a.workers) as pool:
        res = pool.map(run_dataset, jobs, chunksize=1)
    T = pd.DataFrame([r for rr in res for r in rr]); T.to_csv(os.path.join(a.out, "r2_all.csv"), index=False)
    cols = ["R2_curve", "R2_curve_digits", "R2_surface2d", "R2_shuffled"]
    S = T.assign(split=T.split.where(T.split == "random", "block")).groupby(["model", "set", "depth", "split"])[cols].mean().reset_index()
    S.to_csv(os.path.join(a.out, "r2_summary.csv"), index=False)
    print(S.round(3).to_string(index=False)); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
