"""E11 shape search (pre-registered in experiments/E11_topology_hypotheses/SHAPE_SEARCH.md).

Discovery: score 52 fixed number sets for topological / geometric shape in the discovery models against nulls,
lock the top 10 per model. Test: re-score the locked items in held-out models of the same tokenizer type.

  python -m src.e11_shape_search --data <topology_number_line>/data --phase discover --out results/e11_shape_search
  python -m src.e11_shape_search --data <…>/data --phase test --out results/e11_shape_search
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import src.e10_explore as ex
from src.e10_graded import DEPTH, layer_files
from src.e11_tier0 import gaussian_like

DISCOVERY = {"crystal": "digit", "pythia-1.4b": "chunk"}
HELD_OUT = {"digit": ["openllama-3b", "starcoderbase-3b", "qwen2.5-1.5b", "smollm2-1.7b"],
            "chunk": ["olmo2-1b", "redpajama-3b", "llama3.2-1b"]}
SCORES = ["S0", "S1", "S1b", "S2", "Spoly", "Sline"]
DATA = None


def _primes(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return np.nonzero(s)[0]


def candidates():
    """name -> ("points", numbers) or ("centroids", population, labels, group-key for removing higher digits)."""
    rng = np.random.default_rng(29)
    cap = lambda a: np.sort(rng.choice(a, 300, replace=False)) if len(a) > 300 else np.asarray(a)
    C = {}
    for a, b in ((1, 9), (1, 12), (1, 24), (1, 31), (1, 59), (1, 100), (10, 99), (100, 199), (1900, 2025)):
        C[f"range {a}-{b}"] = ("points", np.arange(a, b + 1))
    N = np.arange(1, 10001)
    for k in (10, 25, 50, 100, 1000):
        C[f"multiples of {k}"] = ("points", cap(N[N % k == 0]))
    C["squares"] = ("points", np.arange(1, 101) ** 2)
    C["primes"] = ("points", cap(_primes(10000)))
    C["powers of 2"] = ("points", 2 ** np.arange(14))
    C["repdigits"] = ("points", np.array([int(str(d) * L) for L in (2, 3, 4) for d in range(1, 10)]))
    low, four = np.arange(1, 1000), ex.FOUR[~ex.year_zone(ex.FOUR)]
    for k in (4, 5, 6, 7, 8, 9, 11, 12, 20, 24, 25, 60, 100):
        C[f"mod {k} (1-999)"] = ("centroids", low, low % k, low // 100)
        C[f"mod {k} (1000-9999)"] = ("centroids", four, four % k, four // 100)
    for p, name in enumerate(("units", "tens", "hundreds", "thousands")):
        C[f"{name} digit (1000-9999)"] = ("centroids", four, (four // 10 ** p) % 10, four // 10 ** (p + 1))
    return C


def residual(X, pop, grp):
    """Vectors of `pop` with the mean of their higher-digit group removed (if there is more than one group)."""
    V = X[pop - 1].astype(np.float64).copy()
    if len(np.unique(grp)) > 1:
        for g in np.unique(grp):
            i = grp == g; V[i] -= V[i].mean(0)
    return V


def class_means(V, lab, classes):
    H = (lab[:, None] == classes[None, :]).astype(np.float64)
    return (H.T @ V) / H.sum(0)[:, None]


def scores(P, classes=None):
    D = squareform(pdist(P)); n = len(P)
    D = D / np.median(D[np.triu_indices(n, 1)])
    dg = ripser(D, distance_matrix=True, maxdim=2 if n <= 100 else 1)["dgms"]
    h0 = np.sort(dg[0][np.isfinite(dg[0][:, 1]), 1])[::-1]
    life = lambda d: np.sort(d[:, 1] - d[:, 0])[::-1] if len(d) else np.zeros(0)
    l1 = life(dg[1]); l2 = life(dg[2]) if len(dg) > 2 else np.zeros(0)
    out = dict(S0=float(h0[:3].mean() / np.median(h0)) if len(h0) >= 3 else np.nan,
               S1=float(l1[0]) if len(l1) else 0.0, S1b=float(l1[1]) if len(l1) > 1 else 0.0,
               S2=(float(l2[0]) if len(l2) else 0.0) if n <= 100 else np.nan, Spoly=np.nan, Sline=np.nan)
    if classes is not None:
        K = n; Dn = D + np.diag(np.full(K, np.inf))
        nn2 = np.argsort(Dn, 1)[:, :2]
        share = float(np.mean([set(nn2[i]) == {(i - 1) % K, (i + 1) % K} for i in range(K)]))
        wraps = (K - 1 in nn2[0]) and (0 in nn2[K - 1])                  # a polygon closes; a line does not
        out["Spoly"] = share if wraps else 0.0
        Z = P - P.mean(0); U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        out["Sline"] = float(abs(spearmanr(Z @ Vt[0], classes)[0]) * S[0] ** 2 / np.sum(S ** 2))
    return out


def _cell(args):
    m, d, name, spec, n_null, seed = args
    rng = np.random.default_rng(seed)
    X = np.load(layer_files(DATA, m)[d])["X"].astype(np.float32)
    if spec[0] == "points":
        P = X[spec[1] - 1].astype(np.float64)
        obs = scores(P)
        null = [scores(gaussian_like(P, rng)) for _ in range(n_null)]
    else:
        _, pop, lab, grp = spec
        V = residual(X, pop, grp); cl = np.unique(lab)
        obs = scores(class_means(V, lab, cl), cl)
        null = [scores(class_means(V, rng.permutation(lab), cl), cl) for _ in range(n_null)]
    null = pd.DataFrame(null)
    rows = []
    for s in SCORES:
        if not np.isfinite(obs[s]):
            continue
        nv = null[s].values
        sd = nv.std()
        rows.append(dict(model=m, depth=DEPTH[d], depth_idx=d, set=name, kind=spec[0], score=s, observed=obs[s],
                         null_mean=float(nv.mean()), z=float((obs[s] - nv.mean()) / sd) if sd > 0 else 0.0,
                         p=float((np.sum(nv >= obs[s]) + 1) / (len(nv) + 1)), n_null=n_null))
    return rows


def _init(data):
    global DATA
    DATA = data


def lock(disc, k=10):
    rows = []
    for m in DISCOVERY:
        d = disc[(disc.model == m) & (disc.p <= 1 / 101 + 1e-12)].sort_values("z", ascending=False).head(k)
        rows += d.to_dict("records")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--phase", choices=["discover", "test"], required=True)
    ap.add_argument("--out", default="results/e11_shape_search")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = candidates()
    with Pool(a.workers, initializer=_init, initargs=(a.data,)) as pool:
        if a.phase == "discover":
            jobs = [(m, d, name, spec, 100, 1000 * i + d) for i, (name, spec) in enumerate(C.items())
                    for m in DISCOVERY for d in range(5)]
            disc = pd.DataFrame([r for rows in pool.imap_unordered(_cell, jobs) for r in rows])
            disc.to_csv(os.path.join(a.out, "discovery.csv"), index=False)
            locked = lock(disc)
            json.dump(locked, open(os.path.join(a.out, "locked.json"), "w"), indent=2)
            print(pd.DataFrame(locked)[["model", "depth", "set", "score", "observed", "null_mean", "z", "p"]].to_string(index=False))
        else:
            locked = json.load(open(os.path.join(a.out, "locked.json")))
            jobs, meta = [], []
            for i, it in enumerate(locked):
                for m in HELD_OUT[DISCOVERY[it["model"]]]:
                    if os.path.isdir(os.path.join(a.data, f"{m}_vectors")):
                        jobs.append((m, it["depth_idx"], it["set"], C[it["set"]], 1000, 7000 + i)); meta.append((i, it["score"]))
            res = pool.map(_cell, jobs)
            rows = []
            for (i, s), rr in zip(meta, res):
                r = next(x for x in rr if x["score"] == s)
                rows.append(dict(item=i, discovered_in=locked[i]["model"], **r, significant=r["p"] < 0.005))
            T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "test.csv"), index=False)
            print(T[["item", "discovered_in", "model", "depth", "set", "score", "observed", "z", "p", "significant"]].to_string(index=False))


if __name__ == "__main__":
    main()
