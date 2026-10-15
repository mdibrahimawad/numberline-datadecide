"""E10 stage C (exploratory): questions only topology can ask, on all three models.

  T0  PH profile across depth (H0 spacing, H1 loops) of random 4-digit sets: does topology
      simplify with depth?
  T1  The last-digit circle: do the 10 last-digit classes (…0, …1, …, …9) form a loop (H1) in
      digit order, as the "number helix" papers predict? Does it differ for frequent vs rare
      numbers?
  T2  Years: do the 10 year-endings (19x0 … 19x9) form a loop, compared with the same endings in
      ordinary numbers? Are the decades of 1900-2025 a line or a loop?
  T3  RTD (Representation Topology Divergence, Barannikov et al. ICML 2022) between models on
      the same numbers: do models agree more on the topology of frequent numbers?

All of this is exploratory (all three models' vectors have already been seen in stages A/B-A).

Run:  python -m src.e10_topology --data <topology_number_line>/data --out results/e10_topology
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

import src.e10_explore as ex
from src.topo_features import barcode_features, diagrams

MODELS = {  # same relative depths in every model (stage B-A pre-registration)
    "crystal": ("crystal_vectors", [8, 16, 24, 27, 32]),
    "openllama-3b": ("openllama-3b_vectors", [7, 13, 20, 22, 26]),
    "starcoderbase-3b": ("starcoderbase-3b_vectors", [9, 18, 27, 30, 36]),
}
DEPTH = ["~25%", "~50%", "~75%", "~84%", "last"]


def load(data, model, layer):
    folder, _ = MODELS[model]
    return np.load(os.path.join(data, folder, f"layer_{layer:02d}.npz"))["X"].astype(np.float32)


# --------------------------------------------------------------------------- #
# T0: PH profile across depth
# --------------------------------------------------------------------------- #

def t0_profile(data, K=20, m=300, seed=0):
    rng = np.random.default_rng(seed)
    pool = ex.FOUR[~ex.year_zone(ex.FOUR)]
    rows = []
    for model, (_, layers) in MODELS.items():
        for d, L in enumerate(layers):
            X = load(data, model, L)
            for k in range(K):
                idx = rng.choice(pool, m, replace=False)
                P = X[idx - 1]
                med = np.median(pdist(P))
                f = barcode_features(diagrams(P, med))              # scale-free: set median = 1
                rows.append(dict(model=model, depth=DEPTH[d], layer=L, rep=k, H0_death_mean=f["H0_death_mean"],
                                 H0_death_max=f["H0_death_max"], H1_count=f["H1_count"],
                                 H1_life_max=f["H1_life_max"], H1_total=f["H1_total"], H1_entropy=f["H1_entropy"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# T1 / T2: circles of 10 class centroids
# --------------------------------------------------------------------------- #

def demean(X, nums, remove):
    """Vectors of `nums` with the mean of each `remove` group (e.g. the block of 100) subtracted,
    so that magnitude does not dominate the class centroids."""
    V = X[nums - 1].copy()
    for g in np.unique(remove):
        i = remove == g
        V[i] -= V[i].mean(0)
    return V


def class_means(V, cls, k=10):
    onehot = (cls[None, :] == np.arange(k)[:, None]).astype(np.float32)
    return (onehot @ V) / onehot.sum(1, keepdims=True)


def centroids(X, nums, cls, remove):
    """(10, d) class centroids after removing group means (see `demean`)."""
    return class_means(demean(X, nums, remove), cls)


def circle_stats(C):
    """H1 of 10 centroids: relative persistence of the most persistent loop (life / death, 0 = no
    loop, near 1 = a clear ring), and whether the ring is in digit order: share of the 10 cyclic
    neighbour pairs (c, c+1 mod 10) that are among each other's 2 nearest neighbours."""
    D = squareform(pdist(C.astype(np.float64)))
    h1 = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    rel = float(((h1[:, 1] - h1[:, 0]) / h1[:, 1]).max()) if len(h1) else 0.0
    nn = np.argsort(D, axis=1)[:, 1:3]
    cyc = np.mean([((c + 1) % 10 in nn[c]) or (c in nn[(c + 1) % 10]) for c in range(10)])
    wrap = bool((9 in nn[0]) or (0 in nn[9]))
    return rel, float(cyc), wrap


def t1_digit_circle(data, n_null=200, seed=1):
    """Last-digit circle over ordinary 4-digit numbers (no 1500-2099), block means removed.
    Also tens digit. Also last-digit circle from the frequent vs the rare third separately
    (block_digits thirds, which are balanced in last digit)."""
    rng = np.random.default_rng(seed)
    cnt = ex.load_counts()
    nums = ex.FOUR[~ex.year_zone(ex.FOUR)]
    blocks = nums // 100
    mt = ex.matched_tiers(cnt, "block_digits").set_index("n")["tier"]
    rows = []
    for model, (_, layers) in MODELS.items():
        for d, L in enumerate(layers):
            X = load(data, model, L)
            V = demean(X, nums, blocks)
            for name, cls in (("last digit", nums % 10), ("tens digit", (nums // 10) % 10)):
                rel, cyc, wrap = circle_stats(class_means(V, cls))
                null = np.array([circle_stats(class_means(V, rng.permutation(cls)))[0] for _ in range(n_null)])
                rows.append(dict(model=model, depth=DEPTH[d], layer=L, classes=name, subset="all", loop_rel_persistence=rel,
                                 null_mean=null.mean(), null_95=np.quantile(null, 0.95), p=(np.sum(null >= rel) + 1) / (n_null + 1),
                                 digit_order_share=cyc, wraps_9_to_0=wrap))
            # frequent vs rare third (ordinary numbers that are not multiples of 10: classes 1..9 only)
            for t, lab in ((2, "high third"), (0, "low third")):
                sel = np.array([mt.get(n, -1) == t for n in nums])
                n_t, b_t = nums[sel], blocks[sel]
                V = X[n_t - 1].copy()
                for g in np.unique(b_t):
                    i = b_t == g; V[i] -= V[i].mean(0)
                C = np.stack([V[n_t % 10 == c].mean(0) for c in range(1, 10)])
                Dm = squareform(pdist(C.astype(np.float64)))
                h1 = ripser(Dm, distance_matrix=True, maxdim=1)["dgms"][1]
                rel = float(((h1[:, 1] - h1[:, 0]) / h1[:, 1]).max()) if len(h1) else 0.0
                rows.append(dict(model=model, depth=DEPTH[d], layer=L, classes="last digit 1-9", subset=lab,
                                 loop_rel_persistence=rel))
    return pd.DataFrame(rows)


def t2_years(data, n_null=200, seed=2):
    """Year endings: centroids of 19x0..19x9 / 20x0..20x9 (decade means removed) in years
    1900-2025, vs the same construction on 'twin' ranges 3900-4025 … 8900-9025. Decades:
    H1 of the 13 decade centroids 1900s..2020s (line = no loop)."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, (_, layers) in MODELS.items():
        for d, L in enumerate(layers):
            X = load(data, model, L)
            for start in [1900] + list(range(3900, 9000, 1000)):
                nums = np.arange(start, start + 126); dec = nums // 10
                V = demean(X, nums, dec)
                rel, cyc, wrap = circle_stats(class_means(V, nums % 10))
                null = [circle_stats(class_means(V, rng.permutation(nums % 10)))[0] for _ in range(n_null)]
                Cd = np.stack([X[np.arange(10 * k, 10 * k + 10) - 1].mean(0) for k in np.unique(dec)])
                h1 = ripser(squareform(pdist(Cd.astype(np.float64))), distance_matrix=True, maxdim=1)["dgms"][1]
                dec_loop = float(((h1[:, 1] - h1[:, 0]) / h1[:, 1]).max()) if len(h1) else 0.0
                rows.append(dict(model=model, depth=DEPTH[d], layer=L, range=f"{start}-{start + 125}",
                                 is_years=start == 1900, ending_loop=rel, ending_null_95=float(np.quantile(null, 0.95)),
                                 ending_p=(np.sum(np.array(null) >= rel) + 1) / (n_null + 1), ending_digit_order=cyc,
                                 decade_loop=dec_loop))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# T3: RTD between models
# --------------------------------------------------------------------------- #

def _norm_dist(P):
    D = squareform(pdist(P.astype(np.float64)))
    return D / np.quantile(D[np.triu_indices(len(D), 1)], 0.9)


def rtd_one_way(Da, Db):
    """R-Cross-Barcode_1 total persistence (Barannikov et al. 2022): 2N vertices; A-A edges 0,
    A_i-B_i 0, A_i-B_j = Da_ij, B_i-B_j = min(Da_ij, Db_ij)."""
    n = len(Da)
    M = np.zeros((2 * n, 2 * n))
    M[n:, :n] = Da; M[:n, n:] = Da
    np.fill_diagonal(M[n:, :n], 0); np.fill_diagonal(M[:n, n:], 0)
    M[n:, n:] = np.minimum(Da, Db)
    h1 = ripser(M, distance_matrix=True, maxdim=1)["dgms"][1]
    return float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) else 0.0


def rtd(Pa, Pb):
    Da, Db = _norm_dist(Pa), _norm_dist(Pb)
    return 0.5 * (rtd_one_way(Da, Db) + rtd_one_way(Db, Da))


def t3_rtd(data, K=15, seed=3):
    """RTD between every pair of models at the same relative depth, on the same 120 numbers,
    for the high and the low frequency third (block_digits_plain, 2 numbers per block), plus a
    'mismatched' baseline (model B's points shuffled, so the correspondence is broken)."""
    rng = np.random.default_rng(seed)
    cnt = ex.load_counts(); mt = ex.matched_tiers(cnt, "block_digits_plain")
    pairs = [("crystal", "openllama-3b"), ("crystal", "starcoderbase-3b"), ("openllama-3b", "starcoderbase-3b")]
    rows = []
    for d in range(5):
        Xs = {m: load(data, m, MODELS[m][1][d]) for m in MODELS}
        for k in range(K):
            sub = ex.tier_subsample(mt, mt["tier"].values, rng)
            for t, lab in ((2, "high third"), (0, "low third")):
                idx = sub[t]
                for a, b in pairs:
                    Pa, Pb = Xs[a][idx - 1], Xs[b][idx - 1]
                    rows.append(dict(depth=DEPTH[d], pair=f"{a} vs {b}", subset=lab, rep=k, rtd=rtd(Pa, Pb),
                                     rtd_mismatched=rtd(Pa, Pb[rng.permutation(len(idx))])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="topology_number_line/data")
    ap.add_argument("--out", default="results/e10_topology")
    ap.add_argument("--only", default="0123")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for key, fn, name in (("0", t0_profile, "T0_depth_profile"), ("1", t1_digit_circle, "T1_digit_circle"),
                          ("2", t2_years, "T2_years"), ("3", t3_rtd, "T3_rtd")):
        if key in a.only:
            fn(a.data).to_csv(os.path.join(a.out, f"{name}.csv"), index=False); print(name, "done", flush=True)
    json.dump(dict(models={k: v[1] for k, v in MODELS.items()}, depth=DEPTH), open(os.path.join(a.out, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
