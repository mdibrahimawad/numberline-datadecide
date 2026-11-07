"""H3 quantity / composition / lexicon: residual reliability across context halves, frequency and special-number
regressions, and the E10/E11 rare-vs-frequent crowding re-run with the residual removed."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from src import e13_stage1 as E1
from src.e12c_heldout_r2 import fit_cd, pred_cd

from . import config as C
from .h0 import _space
from .panel import choose_h, n_digits, positions_for

FAMOUS = {7, 404, 911, 666, 1337}


def counts():
    c = pd.read_csv(C.COUNTS).set_index("number")["count"].reindex(range(10001)).fillna(0)
    return c.values.astype(float)


def flags(n):
    s = pd.Series(n.astype(str))
    return pd.DataFrame(dict(year=((n >= 1900) & (n <= 2030)).astype(float), mult100=(n % 100 == 0).astype(float),
                             mult1000=(n % 1000 == 0).astype(float), repdigit=s.map(lambda x: len(set(x)) == 1 and len(x) > 1).astype(float).values,
                             pow2=np.isin(n, [2 ** k for k in range(1, 14)]).astype(float), famous=np.isin(n, list(FAMOUS)).astype(float)))


def residual_reliability(ZA, ZB, n, s, tr, te, seed=0):
    """Fit f + D on each half's training numbers; residuals of held-out numbers; mean per-number correlation."""
    L = n_digits(s); pos = positions_for(L); res = []
    for Z in (ZA, ZB):
        h = choose_h(Z[tr], n[tr], seed); m = fit_cd(Z[tr], n[tr], h, pos)
        res.append(Z[te] - pred_cd(m, n[tr], n[te], h, pos))
    ra = res[0] - res[0].mean(0); rb = res[1] - res[1].mean(0)
    num = (ra * rb).sum(1); den = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def reliability(md, slot, n, s, n_splits=5, seed=0):
    halves = list(itertools.combinations(C.NEUTRAL, 2))
    rng = np.random.default_rng(seed); pick = rng.choice(len(halves), min(n_splits, len(halves)), replace=False)
    te = np.random.default_rng(seed + 1).random(len(n)) < 0.2; tr = ~te; vals = []
    for k in pick:
        A = list(halves[k]); B = [c for c in C.NEUTRAL if c not in A]
        XA = md.half_average(A, slot, n); XB = md.half_average(B, slot, n)
        ZA, ZB = _space([XA, XB], fit_rows=np.where(tr)[0])
        vals.append(residual_reliability(ZA, ZB, n, s, tr, te, seed))
    return dict(reliability=float(np.mean(vals)), per_split=vals)


def crossfit_residual(Z, n, s, seed=0, k=5):
    L = n_digits(s); pos = positions_for(L); R = np.zeros_like(Z); Fh = np.zeros_like(Z)
    from sklearn.model_selection import KFold
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(n):
        h = choose_h(Z[tr], n[tr], seed); m = fit_cd(Z[tr], n[tr], h, pos)
        Fh[te] = pred_cd(m, n[tr], n[te], h, pos); R[te] = Z[te] - Fh[te]
    return R, Fh


def frequency_regression(Z, R, n, B=500, seed=0):
    cnt = counts(); lf = np.log10(cnt[n] + 1); F = flags(n); out = {}
    for name, y in (("resid_norm", np.linalg.norm(R, axis=1)), ("vector_norm", np.linalg.norm(Z - Z.mean(0), axis=1))):
        cols = [c for c in F if F[c].sum() >= 3]
        X = np.column_stack([np.ones(len(n)), lf] + [F[c].values for c in cols])
        b = np.linalg.lstsq(X, y, rcond=None)[0]; rng = np.random.default_rng(seed); bs = []
        for _ in range(B):
            i = rng.integers(0, len(n), len(n)); bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
        bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        out[name] = dict(slope_logfreq=float(b[1]), lo=float(lo[1]), hi=float(hi[1]), sd_y=float(y.std()),
                         flags={c: dict(coef=float(b[2 + q]), lo=float(lo[2 + q]), hi=float(hi[2 + q])) for q, c in enumerate(cols)})
    return out


def mst_deaths(P):
    return minimum_spanning_tree(squareform(pdist(P))).data


def crowding(Z, Fh, n, n_sub=50, seed=0):
    """S4 only. E10/E11 'block' tiers (rarest vs most frequent third inside each block of 100); median H0 death
    (= MST edge length) per tier subsample, in Z and in Z with the residual removed (= f + D)."""
    import src.e10_explore as ex
    tiers = ex.matched_tiers(counts(), "block"); tiers = tiers[tiers.n.isin(n)]
    if len(tiers) < 200:
        return None
    pos = {int(v): i for i, v in enumerate(n)}; rng = np.random.default_rng(seed); eff = {"full": [], "removed": []}
    for _ in range(n_sub):
        sub = ex.tier_subsample(tiers, tiers["tier"].values, rng)
        for name, M in (("full", Z), ("removed", Fh)):
            med = {t: np.median(mst_deaths(M[[pos[int(v)] for v in sub[t]]])) for t in (0, 2)}
            eff[name].append((med[2] - med[0]) / ((med[2] + med[0]) / 2))
    f = float(np.mean(eff["full"])); r = float(np.mean(eff["removed"]))
    return dict(effect_full=f, effect_removed=r, effect_full_ci=[float(x) for x in np.percentile(eff["full"], [2.5, 97.5])],
                reduction=float(1 - r / f) if f != 0 else None, lexical=bool(f != 0 and (1 - r / f) >= 0.5))


def run(md, slot, s):
    n = md.set_numbers(s, C.NEUTRAL)
    if len(n) < 30:
        return dict(skipped=f"only {len(n)} numbers")
    out = dict(N=int(len(n)))
    nr = n
    if s == "S4":
        from .data import subsample_s4
        nr = subsample_s4(n)
    out["reliability"] = reliability(md, slot, nr, s)
    if s == "S4":
        Zs = E1.prep(md.X(("AVG5", "num"), slot, nr)); R, _ = crossfit_residual(Zs, nr, s)
        out["frequency"] = frequency_regression(Zs, R, nr)
        Z = E1.prep(md.X(("AVG5", "num"), slot, n)); out["crowding"] = crowding(Z, _full_fit(Z, n, s), n)
    else:
        Z = E1.prep(md.X(("AVG5", "num"), slot, n)); R, _ = crossfit_residual(Z, n, s)
        out["frequency"] = frequency_regression(Z, R, n)
    return out


def _full_fit(Z, n, s, seed=0):
    """f + D fitted on all S4 numbers of the 3,000-number subsample, evaluated at every number (for crowding)."""
    from .data import subsample_s4
    L = n_digits(s); pos = positions_for(L); ns = subsample_s4(n); idx = np.searchsorted(n, ns)
    h = choose_h(Z[idx], ns, seed); m = fit_cd(Z[idx], ns, h, pos)
    return pred_cd(m, ns, n, h, pos)
