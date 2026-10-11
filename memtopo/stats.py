"""Metrics and uncertainty (brief section 7): AUROC, prevalence-reweighted AUPRC and
recall at fixed precision, paired bootstrap, DeLong, Hedges' g, Cliff's delta."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _weighted_pr(y, s, w):
    o = np.argsort(-np.asarray(s), kind="mergesort")
    y, s, w = np.asarray(y)[o].astype(bool), np.asarray(s)[o], np.asarray(w, float)[o]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * ~y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]       # one point per distinct threshold
    tp, fp = tp[last], fp[last]
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / max(tp[-1], 1e-300)
    return prec, rec


def auprc(y, s, w=None) -> float:
    """Average precision with sample weights (negatives up-weighted to natural prevalence)."""
    w = np.ones(len(y)) if w is None else w
    if np.asarray(y).sum() == 0:
        return float("nan")
    prec, rec = _weighted_pr(y, s, w)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def recall_at_precision(y, s, w=None, p: float = 0.9) -> float:
    w = np.ones(len(y)) if w is None else w
    if np.asarray(y).sum() == 0:
        return float("nan")
    prec, rec = _weighted_pr(y, s, w)
    ok = prec >= p
    return float(rec[ok].max()) if ok.any() else 0.0


def pr_curve(y, s, w=None):
    w = np.ones(len(y)) if w is None else w
    return _weighted_pr(y, s, w)


def precision_recall_binary(y, pred, w=None) -> tuple[float, float]:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    tp, fp, fn = (w * (y & pred)).sum(), (w * (~y & pred)).sum(), (w * (y & ~pred)).sum()
    return (tp / (tp + fp) if tp + fp else float("nan")), (tp / (tp + fn) if tp + fn else float("nan"))


def bootstrap_auroc(y, scores: dict, n: int = 2000, seed: int = 0) -> dict:
    """Percentile CIs for each model's AUROC from the same resamples (paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = {k: [] for k in scores}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        for k, s in scores.items():
            out[k].append(auroc(y[i], np.asarray(s)[i]))
    return {k: np.asarray(v) for k, v in out.items()}


def ci(v: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


# ---------------------------------------------------------------------- DeLong (Sun & Xu 2014)
def _midrank(x):
    return rankdata(x)


def delong(y, s1, s2) -> tuple[float, float]:
    """Paired DeLong test: returns (AUROC1 - AUROC2, two-sided p)."""
    y = np.asarray(y).astype(bool)
    pos = [np.asarray(s)[y] for s in (s1, s2)]
    neg = [np.asarray(s)[~y] for s in (s1, s2)]
    m, n = len(pos[0]), len(neg[0])
    v10, v01, aucs = [], [], []
    for p, q in zip(pos, neg):
        tx, ty, tz = _midrank(p), _midrank(q), _midrank(np.r_[p, q])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
        aucs.append(auc)
    sx, sy = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (sx[0, 0] + sx[1, 1] - 2 * sx[0, 1]) / m + (sy[0, 0] + sy[1, 1] - 2 * sy[0, 1]) / n
    d = aucs[0] - aucs[1]
    if var <= 0:
        return float(d), float("nan")
    return float(d), float(2 * norm.sf(abs(d) / np.sqrt(var)))


# ---------------------------------------------------------------------- effect sizes
def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * (a.mean() - b.mean()) / sp)


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b), via ranks (O(n log n))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = rankdata(np.r_[a, b])
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2          # Mann-Whitney U (ties count 1/2)
    return float(2 * u / (len(a) * len(b)) - 1)


def bootstrap_two_sample(f, a, b, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    v = [f(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(n)]
    lo, hi = ci(np.asarray(v))
    return f(a, b), lo, hi
