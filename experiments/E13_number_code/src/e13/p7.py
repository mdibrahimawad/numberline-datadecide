"""P7 fixed positive control: days (7) and months (12), last token of the item in 20 templates, averaged over
templates; projection onto the top-k PCs (k = 2..6) of the item centroids; circular-order score and a circular probe.
The same detector runs on the units digit of numbers."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src import e13_stage1 as E1

from . import config as C

OLD_SLOTS = [2, 4, 6, 7, 8]          # the five E10 depths among the nine stored layers


def circular_order(cent, k):
    """Share of items whose two nearest items (in the top-k PCs of the centroids) are their cyclic neighbours."""
    K = len(cent); X = cent - cent.mean(0); k = min(k, K - 1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False); P = X @ Vt[:k].T
    D = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(D, np.inf); hit = 0
    for i in range(K):
        nn = set(np.argsort(D[i])[:2]); hit += nn == {(i - 1) % K, (i + 1) % K}
    return hit / K


def circular_probe(Zg, labels, K, groups):
    """Ridge from vectors to (cos, sin)(2 pi i / K); leave-one-group-out; decoded by the nearest item angle."""
    acc = []
    ang = 2 * np.pi * labels / K; Y = np.c_[np.cos(ang), np.sin(ang)]
    for g in np.unique(groups):
        tr = groups != g; m = Ridge(alpha=1.0).fit(Zg[tr], Y[tr]); p = m.predict(Zg[~tr])
        dec = np.round(np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi) / (2 * np.pi) * K).astype(int) % K
        acc.append(np.mean(dec == labels[~tr]))
    return float(np.mean(acc))


def detect(arr):
    """arr: [templates, items, d] for one layer."""
    T, K, d = arr.shape; X = arr.reshape(T * K, d).astype(np.float64)
    Z = E1.prep(X); Zr = Z.reshape(T, K, -1); cent = Zr.mean(0)
    scores = {k: circular_order(cent, k) for k in range(2, 7)}
    labels = np.tile(np.arange(K), T); groups = np.repeat(np.arange(T), K)
    probe = circular_probe(Z, labels, K, groups)
    n_hi = sum(v >= 0.75 for v in scores.values())
    return dict(order_score=scores, n_k_pass=int(n_hi), probe_acc=probe, chance=1 / K,
                passed=bool(n_hi >= 3 and probe >= 0.5))


def run(model_dir):
    import os
    out = {}
    for name in ("days", "months"):
        p = os.path.join(model_dir, f"p7_{name}.npy")
        if not os.path.exists(p):
            continue
        A = np.load(p); res = {int(sl): detect(A[:, :, sl]) for sl in range(A.shape[2])}
        old = [res[sl]["passed"] for sl in OLD_SLOTS if sl in res]
        out[name] = dict(by_slot=res, passes_old_depths=int(sum(old)), works=bool(sum(old) >= 2))
    out["control_works"] = bool(all(out.get(k, {}).get("works") for k in ("days", "months")))
    return out


def numbers_units(md, slot, s):
    """The same detector on the units digit: centroids of the 10 units classes (neutral average), probe by number."""
    n = md.set_numbers(s, ["AVG5"])
    if len(n) < 50:
        return None
    if s == "S4":
        from .data import subsample_s4
        n = subsample_s4(n)
    Z = E1.prep(md.X(("AVG5", "num"), slot, n)); u = n % 10
    cent = np.stack([Z[u == v].mean(0) for v in range(10)])
    scores = {k: circular_order(cent, k) for k in range(2, 7)}
    groups = np.random.default_rng(0).integers(0, 5, len(n))
    probe = circular_probe(Z, u, 10, groups)
    return dict(order_score=scores, n_k_pass=int(sum(v >= 0.75 for v in scores.values())), probe_acc=probe, chance=0.1)
