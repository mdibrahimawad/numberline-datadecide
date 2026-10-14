"""Barcode summary features for a point cloud (E10).

`barcode_features` reconstructs the 41-feature barcode summary of Fay et al. (ICLR 2026,
"The Shape of Adversarial Influence"), from the paper's description: summary statistics of
the births, deaths and lifetimes of H0 and H1 bars, birth/death ratios, loop midpoints, bar counts, total
persistence and persistent entropy. Their appendix was not accessible when this was written,
so the exact list is ours (FEATURE_NAMES), not a verbatim copy.

Filtration: Vietoris-Rips on Euclidean distances in the full hidden space, divided by a
caller-supplied scale (so values are comparable across layers but group scale differences
are kept).
"""
from __future__ import annotations

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform


def _stats(x: np.ndarray, prefix: str) -> dict:
    if len(x) == 0:
        x = np.zeros(1)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std(), f"{prefix}_median": med,
            f"{prefix}_q1": q1, f"{prefix}_q3": q3, f"{prefix}_min": x.min(), f"{prefix}_max": x.max()}


def _entropy(life: np.ndarray) -> float:
    life = life[life > 0]
    if len(life) == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def diagrams(points: np.ndarray, scale: float = 1.0) -> list[np.ndarray]:
    """H0 and H1 persistence diagrams (Rips, Euclidean / scale)."""
    D = squareform(pdist(points.astype(np.float64))) / scale
    return ripser(D, distance_matrix=True, maxdim=1)["dgms"]


def barcode_features(dgms: list[np.ndarray]) -> dict:
    """41 summary features of an (H0, H1) diagram pair."""
    h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
    h1 = dgms[1]
    d0 = h0[:, 1]                                   # H0: birth = 0, so death = lifetime
    b1, e1 = h1[:, 0], h1[:, 1]
    l1 = e1 - b1
    f = {"H0_count": len(d0)}
    f.update(_stats(d0, "H0_death"))                # 7
    f["H0_total"] = d0.sum()
    f["H0_entropy"] = _entropy(d0)
    f["H1_count"] = len(l1)
    f.update(_stats(b1, "H1_birth"))                # 7
    f.update(_stats(e1, "H1_death"))                # 7
    f.update(_stats(l1, "H1_life"))                 # 7
    ratio = b1 / e1 if len(e1) else np.zeros(1)     # scale-invariant birth/death ratio
    f.update({k: v for k, v in _stats(ratio, "H1_ratio").items()
              if k.split("_")[-1] in ("mean", "std", "median", "q1", "q3")})   # 5
    mid = (b1 + e1) / 2 if len(e1) else np.zeros(1)  # where on the scale loops live
    f["H1_mid_mean"], f["H1_mid_std"] = mid.mean(), mid.std()
    f["H1_total"] = l1.sum()
    f["H1_entropy"] = _entropy(l1)
    return {k: float(v) for k, v in f.items()}


FEATURE_NAMES = list(barcode_features([np.array([[0, 1.0], [0, np.inf]]), np.array([[0.5, 1.0]])]).keys())
assert len(FEATURE_NAMES) == 41, len(FEATURE_NAMES)
