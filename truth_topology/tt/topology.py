"""Persistent homology pieces: normalisation, subsamples, Vietoris-Rips diagrams, the 41-feature
barcode summary (Fay et al. 2026), Wasserstein-2 / L-inf diagram distance, effect sizes.

The 41 features follow the definition in the brief (= Fay et al.): for each of
{H0 deaths, H1 births, H1 deaths, H1 persistence, H1 birth/death ratio} the 7 statistics
{mean, min, q1, median, q3, max, std} (35), plus total persistence, number of bars and
persistent entropy for H0 and H1 (6). The Malhotra et al. repo's summary.py uses a different
41 (skew, kurtosis, gaps, ...), so it is not reused; only its conventions are (finite bars
only; empty -> 0).
"""
from __future__ import annotations

import numpy as np
from ripser import ripser
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

STATS = ["mean", "min", "q1", "median", "q3", "max", "std"]
GROUPS = ["h0_death", "h1_birth", "h1_death", "h1_pers", "h1_bdratio"]
FEATURE_NAMES = [f"{g}_{s}" for g in GROUPS for s in STATS] + [
    "h0_total_pers", "h0_n_bars", "h0_entropy", "h1_total_pers", "h1_n_bars", "h1_entropy"]
assert len(FEATURE_NAMES) == 41


# ---------------------------------------------------------------- geometry

def distances(X: np.ndarray) -> np.ndarray:
    """Full Euclidean distance matrix in float64."""
    X = np.asarray(X, dtype=np.float64)
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.from_numpy(X).cuda()
            return torch.cdist(t, t).cpu().numpy()
    except ImportError:
        pass
    return squareform(pdist(X))


def norm_scale(D: np.ndarray, ref_idx: np.ndarray) -> float:
    """Median pairwise distance among the fixed reference points (1.0 if the cloud is degenerate)."""
    sub = D[np.ix_(ref_idx, ref_idx)]
    med = float(np.median(sub[np.triu_indices(len(ref_idx), 1)]))
    return med if med > 1e-12 else 1.0


# ---------------------------------------------------------------- subsamples

def stratified_subsamples(strata: np.ndarray, pool: np.ndarray, B: int, m: int, rng) -> np.ndarray:
    """B index sets of size m drawn without replacement from ``pool`` (item indices), with each
    stratum represented in proportion to its share of the pool. Returns (B, m) int array."""
    s = strata[pool]
    keys, counts = np.unique(s, return_counts=True)
    quota = np.floor(counts / counts.sum() * m).astype(int)
    for i in np.argsort(-(counts / counts.sum() * m - quota))[: m - quota.sum()]:
        quota[i] += 1
    out = np.empty((B, m), dtype=int)
    for b in range(B):
        out[b] = np.concatenate([rng.choice(pool[s == k], q, replace=False) for k, q in zip(keys, quota)])
    return out


# ---------------------------------------------------------------- PH

def diagrams(D_sub: np.ndarray) -> list[np.ndarray]:
    """[H0, H1] finite bars of the Vietoris-Rips filtration of a distance matrix."""
    dg = ripser(D_sub, distance_matrix=True, maxdim=1)["dgms"]
    return [d[np.isfinite(d[:, 1])] if len(d) else np.empty((0, 2)) for d in dg[:2]]


def _stats(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * 7
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return [float(x.mean()), float(x.min()), float(q1), float(med), float(q3), float(x.max()), float(x.std())]


def _entropy(lengths: np.ndarray) -> float:
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return 0.0
    p = lengths / lengths.sum()
    return float(-(p * np.log(p)).sum())


def summary(dgms: list[np.ndarray]) -> np.ndarray:
    h0, h1 = dgms
    d0 = h0[:, 1]
    b1, e1 = h1[:, 0], h1[:, 1]
    p1 = e1 - b1
    ratio = b1[e1 > 0] / e1[e1 > 0]
    v = _stats(d0) + _stats(b1) + _stats(e1) + _stats(p1) + _stats(ratio)
    v += [float(d0.sum()), float(d0.size), _entropy(d0), float(p1.sum()), float(p1.size), _entropy(p1)]
    return np.array(v)


# ---------------------------------------------------------------- diagram distance

def wasserstein2(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-2 distance between two persistence diagrams with L-inf ground metric
    (points may be matched to the diagonal at cost (death - birth) / 2)."""
    n, k = len(a), len(b)
    if n == 0 and k == 0:
        return 0.0
    da = (a[:, 1] - a[:, 0]) / 2 if n else np.empty(0)
    db = (b[:, 1] - b[:, 0]) / 2 if k else np.empty(0)
    if n == 0:
        return float(np.sqrt((db ** 2).sum()))
    if k == 0:
        return float(np.sqrt((da ** 2).sum()))
    cost = np.zeros((n + k, n + k))
    cost[:n, :k] = np.maximum(np.abs(a[:, None, 0] - b[None, :, 0]), np.abs(a[:, None, 1] - b[None, :, 1])) ** 2
    cost[:n, k:] = np.inf
    cost[np.arange(n), k + np.arange(n)] = da ** 2
    cost[n:, :k] = np.inf
    cost[n + np.arange(k), np.arange(k)] = db ** 2
    # diagonal-to-diagonal block stays 0
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(cost[r, c].sum()))


# ---------------------------------------------------------------- statistics

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    J = 1 - 3 / (4 * (nx + ny) - 9)
    return float(J * (x.mean() - y.mean()) / sp)


def hedges_g_boot(x: np.ndarray, y: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    g = hedges_g(x, y)
    bs = [hedges_g(x[rng.integers(len(x), size=len(x))], y[rng.integers(len(y), size=len(y))]) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return g, float(lo), float(hi)


def prune_correlated(F: np.ndarray, names: list[str], thr: float) -> list[int]:
    """Greedy: keep a feature unless |corr| > thr with an already kept one; constant features dropped."""
    keep = []
    sd = F.std(0)
    Z = (F - F.mean(0)) / np.where(sd > 0, sd, 1)
    R = np.abs(np.corrcoef(Z.T))
    for j in range(F.shape[1]):
        if sd[j] == 0:
            continue
        if all(R[j, k] <= thr for k in keep):
            keep.append(j)
    return keep
