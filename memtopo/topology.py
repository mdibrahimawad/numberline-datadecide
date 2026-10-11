"""Persistent homology of token point clouds, the 41-number barcode summary, diagram
distances and same-cloud geometry (brief sections 6.1 B6 and 6.2).

Everything here is numpy/scipy only, except ``diagrams`` which calls Ripser.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

STATS = ("mean", "min", "q1", "median", "q3", "max", "std")
GROUPS = ("h0_death", "h1_birth", "h1_death", "h1_pers", "h1_ratio")
FEATURE_NAMES = [f"{g}_{s}" for g in GROUPS for s in STATS] + [
    f"{h}_{k}" for h in ("h0", "h1") for k in ("total_pers", "n_bars", "entropy")
]
assert len(FEATURE_NAMES) == 41


# --------------------------------------------------------------------------- clouds
def normalise(cloud: np.ndarray) -> np.ndarray:
    """Centre the cloud and divide by its median pairwise distance (fixed in advance)."""
    x = np.asarray(cloud, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    med = np.median(pdist(x))
    return x / med if med > 0 else x


def diagrams(cloud: np.ndarray, maxdim: int = 1) -> list[np.ndarray]:
    """Vietoris-Rips persistence (Euclidean) of one cloud; the infinite H0 bar is dropped."""
    from ripser import ripser

    d = squareform(pdist(np.asarray(cloud, dtype=np.float64)))
    dgms = ripser(d, maxdim=maxdim, distance_matrix=True)["dgms"]
    return [dg[np.isfinite(dg[:, 1])].astype(np.float64) for dg in dgms]


# --------------------------------------------------------------------------- summary
def _stats(v: np.ndarray) -> list[float]:
    if v.size == 0:                   # empty H1: all statistics 0 (n_bars = 0 carries the fact)
        return [0.0] * 7
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return [v.mean(), v.min(), q1, med, q3, v.max(), v.std()]


def _entropy(pers: np.ndarray) -> float:
    pers = pers[pers > 0]
    if pers.size == 0:
        return 0.0
    p = pers / pers.sum()
    return float(-(p * np.log(p)).sum())


def summary(dgms: list[np.ndarray]) -> np.ndarray:
    """41 features (Fay et al. 2026): 7 statistics x {H0 deaths, H1 births, H1 deaths,
    H1 persistences, H1 birth/death ratios} + {total persistence, number of bars,
    persistent entropy} for H0 and H1."""
    h0, h1 = dgms[0], dgms[1]
    h0_death = h0[:, 1]
    h0_pers = h0[:, 1] - h0[:, 0]
    b, d = h1[:, 0], h1[:, 1]
    h1_pers = d - b
    ratio = np.divide(b, d, out=np.zeros_like(b), where=d > 0)
    out = []
    for v in (h0_death, b, d, h1_pers, ratio):
        out += _stats(v)
    for pers in (h0_pers, h1_pers):
        out += [float(pers.sum()), float(pers.size), _entropy(pers)]
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- distances
def wasserstein(d1: np.ndarray, d2: np.ndarray, p: float = 2.0) -> float:
    """p-Wasserstein distance between persistence diagrams, Euclidean ground metric, points
    may be matched to the diagonal (distance (death - birth) / sqrt 2). With p = 1 this is
    exactly persim.wasserstein; persim has no p = 2, hence this implementation."""
    d1 = np.asarray(d1, dtype=np.float64).reshape(-1, 2)
    d2 = np.asarray(d2, dtype=np.float64).reshape(-1, 2)
    m, n = len(d1), len(d2)
    if m == 0 and n == 0:
        return 0.0
    big = np.zeros((m + n, m + n))
    if m and n:
        big[:m, :n] = np.sqrt(((d1[:, None, :] - d2[None, :, :]) ** 2).sum(2)) ** p
    big[:m, n:] = np.inf
    big[m:, :n] = np.inf
    if m:
        big[np.arange(m), n + np.arange(m)] = ((d1[:, 1] - d1[:, 0]) / np.sqrt(2)) ** p
    if n:
        big[m + np.arange(n), np.arange(n)] = ((d2[:, 1] - d2[:, 0]) / np.sqrt(2)) ** p
    big[m:, n:] = 0.0                      # diagonal to diagonal
    r, c = linear_sum_assignment(big)
    return float(big[r, c].sum() ** (1.0 / p))


# --------------------------------------------------------------------------- geometry (B6)
GEOM_NAMES = ["total_var", "mean_pdist", "participation_ratio", "effective_rank",
              "mean_consec_cos", "centroid_norm"]


def geometry(cloud: np.ndarray) -> np.ndarray:
    x = np.asarray(cloud, dtype=np.float64)
    centroid = x.mean(axis=0)
    xc = x - centroid
    sv = np.linalg.svd(xc, compute_uv=False)
    lam = sv ** 2 / max(len(x) - 1, 1)
    total_var = lam.sum()
    pr = total_var ** 2 / (lam ** 2).sum() if total_var > 0 else 0.0
    ps = sv / sv.sum() if sv.sum() > 0 else sv
    ps = ps[ps > 0]
    erank = float(np.exp(-(ps * np.log(ps)).sum())) if ps.size else 0.0
    nrm = np.linalg.norm(x, axis=1)
    cos = (x[1:] * x[:-1]).sum(1) / np.maximum(nrm[1:] * nrm[:-1], 1e-12)
    return np.array([total_var, pdist(x).mean(), pr, erank, cos.mean(),
                     np.linalg.norm(centroid)])


# --------------------------------------------------------------------------- per-cloud job
def cloud_features(cloud: np.ndarray, prompt_len: int, also_raw: bool = True,
                   keep_diagrams: bool = True) -> dict:
    """Everything computed from one 64-token cloud at one layer and checkpoint:
    T1 (normalised; full 64 tokens and continuation tokens), optional raw T1, B6 geometry,
    and the normalised full-cloud diagrams (for T2/T3 distances)."""
    x = np.asarray(cloud, dtype=np.float64)
    out = {"geom": geometry(x)}
    dg_full = diagrams(normalise(x))
    out["t1_full"] = summary(dg_full)
    out["t1_cont"] = summary(diagrams(normalise(x[prompt_len:])))
    if also_raw:
        out["t1_full_raw"] = summary(diagrams(x - x.mean(0)))
    if keep_diagrams:
        out["dgm_h0"] = dg_full[0].astype(np.float32)
        out["dgm_h1"] = dg_full[1].astype(np.float32)
    return out


def gaussian_floor(n: int, d: int, reps: int, seed: int = 0) -> np.ndarray:
    """T1 summaries of isotropic Gaussian clouds matched in n and d (Malhotra's control)."""
    rng = np.random.default_rng(seed)
    return np.stack([summary(diagrams(normalise(rng.standard_normal((n, d)))))
                     for _ in range(reps)])
