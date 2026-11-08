"""E13-1D: test battery for "a 1-D manifold encodes the number line" (PREREG_1D.md).

Shape tests T1-T4 (+T5 contexts, T6 across models, T10 checkpoints) on stored vectors; causal tests T7-T9 in
`causal_1d` (needs the model; CUDA, Apple MPS or CPU)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

from src.e12c_heldout_r2 import kernel_pred

from . import config as C
from .panel import choose_h

SETS = {"S2": (10, 99), "S3": (100, 999), "S4": (1000, 9999)}
BLOCK_LEADS = (3, 5, 7)


# --------------------------------------------------------------------------- preparation
def prep_fit(Xtr, D=50):
    mu = Xtr.mean(0); sd = Xtr.std(0); keep = sd > 1e-6 * max(sd.max(), 1e-12)
    A = (Xtr[:, keep] - mu[keep]) / sd[keep]; D = min(D, A.shape[0] - 1, A.shape[1])
    m0 = A.mean(0); Vt = np.linalg.svd(A - m0, full_matrices=False)[2][:D]
    return dict(mu=mu, sd=sd, keep=keep, m0=m0, Vt=Vt)


def prep_apply(T, X):
    return ((X[:, T["keep"]] - T["mu"][T["keep"]]) / T["sd"][T["keep"]] - T["m0"]) @ T["Vt"].T


# --------------------------------------------------------------------------- the curve
def fit_curve(Ztr, ntr, lo, hi, seed=0, h=None):
    h = h or choose_h(Ztr, ntr, seed)
    grid = np.arange(lo, hi + 1)
    F = np.vstack([kernel_pred(Ztr, ntr, grid[k:k + 2000], h) for k in range(0, len(grid), 2000)])
    mu = F.mean(0); Fc = F - mu
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False); share = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
    m = int(min(5, np.searchsorted(share, 0.90) + 1)); V = Vt[:m]
    return dict(h=float(h), m=m, V=V, mu=mu, gamma=Fc @ V.T, grid=grid, F=F)


def place(curve, Z, chunk=256):
    P = (Z - curve["mu"]) @ curve["V"].T; out = np.empty(len(Z)); g = curve["gamma"]
    for k in range(0, len(Z), chunk):
        d = ((P[k:k + chunk, None, :] - g[None]) ** 2).sum(-1); out[k:k + chunk] = curve["grid"][d.argmin(1)]
    return out


def r2(y, p, ref):
    return float(1 - ((y - p) ** 2).sum() / ((y - ref) ** 2).sum())


# --------------------------------------------------------------------------- T1 + T2
def t1_t2(X, n, s, reps=3, shuffled=True, prepped=False):
    lo, hi = SETS[s] if s in SETS else (int(n.min()), int(n.max())); rows = []
    for rep in range(reps):
        te = np.random.default_rng(rep).random(len(n)) < 0.2; tr = ~te
        T = None if prepped else prep_fit(X[tr]); Z = X if prepped else prep_apply(T, X)
        cur = fit_curve(Z[tr], n[tr], lo, hi, rep); nh = place(cur, Z[te]); y = n[te].astype(float); ref = n[tr].mean()
        knn = KNeighborsRegressor(10).fit(Z[tr], n[tr]).predict(Z[te])
        rid = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Z[tr], n[tr]).predict(Z[te])
        row = dict(rho=float(spearmanr(nh, y)[0]), R2_curve=r2(y, nh, ref), R2_knn=r2(y, knn, ref), R2_ridge=r2(y, rid, ref), m=cur["m"], h=cur["h"])
        if shuffled and rep == 0:
            ns = np.random.default_rng(99).permutation(n[tr]); cs = fit_curve(Z[tr], ns, lo, hi, rep)
            row["rho_shuffled"] = float(spearmanr(place(cs, Z[te]), y)[0])
        rows.append(row)
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("rho", "R2_curve", "R2_knn", "R2_ridge", "m", "h")}
    out["rho_shuffled"] = rows[0].get("rho_shuffled")
    best = max(out["R2_knn"], out["R2_ridge"])
    out["T1_pass"] = bool(out["rho"] >= 0.90)
    out["T2_pass"] = bool(out["R2_curve"] >= 0.80 and out["R2_curve"] >= 0.90 * best)
    return out


def shuffle_null(X, n, s, perms=20, seed=7):
    """Amendment 1: T1 must beat the 95th percentile of the order obtained with n shuffled among training numbers
    (a curve built on shuffled labels can still give a high order by chance, see PREREG_1D.md)."""
    lo, hi = SETS[s]; te = np.random.default_rng(0).random(len(n)) < 0.2; tr = ~te
    T = prep_fit(X[tr]); Z = prep_apply(T, X); rng = np.random.default_rng(seed); rh = []; r2s = []
    h = choose_h(Z[tr], n[tr], 0)
    for _ in range(perms):
        ns = rng.permutation(n[tr]); cur = fit_curve(Z[tr], ns, lo, hi, 0, h=h); nh = place(cur, Z[te])
        rh.append(float(spearmanr(nh, n[te])[0]) if np.ptp(nh) > 0 else 0.0); r2s.append(r2(n[te].astype(float), nh, n[tr].mean()))
    return dict(rho_q95=float(np.percentile(rh, 95)), rho_mean=float(np.mean(rh)), R2_q95=float(np.percentile(r2s, 95)),
                T2_false_pass=float(np.mean([v >= 0.80 for v in r2s])))


# --------------------------------------------------------------------------- T3 continuity
def lead(n):
    return np.array([int(str(int(v))[0]) for v in n])


def t3(Z, n, s, B=500, seed=0, prepped=True):
    o = np.argsort(n); Z, n = Z[o], n[o]; R = max(n.max() - n.min(), 1); b = max(3, int(round(0.05 * R)))
    if s == "S4":
        b = max(3, int(round(0.05 * R * len(n) / (R + 1))))          # subsample: same share of the numbers
    L = lead(n); blocks = []
    for g in np.unique(L):
        idx = np.where(L == g)[0]; k = max(1, len(idx) // b)
        blocks += [(g, part) for part in np.array_split(idx, k) if len(part) >= 2]
    if len(blocks) < 4:
        return dict(ratio=None, T3_pass=None, note="too few blocks")

    def stats(sel):
        cent = np.stack([Z[p].mean(0) for _, p in sel]); cnt = np.array([len(p) for _, p in sel])
        within = sum(((Z[p] - Z[p].mean(0)) ** 2).sum() for _, p in sel) / max(sum(cnt) - len(sel), 1)
        d2 = ((cent[1:] - cent[:-1]) ** 2).sum(1) - within * (1 / cnt[1:] + 1 / cnt[:-1])
        d = np.sqrt(np.clip(d2, 0, None)); bd = np.array([sel[k][0] != sel[k + 1][0] for k in range(len(sel) - 1)])
        if bd.sum() == 0 or (~bd).sum() == 0:
            return None, None
        return float(np.median(d[bd]) / max(np.median(d[~bd]), 1e-12)), float(np.median(d[~bd]))
    ratio, within_step = stats(blocks)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        sel = [(g, rng.choice(p, len(p), replace=True)) for g, p in blocks]; r_, _ = stats(sel)
        if r_ is not None:
            bs.append(r_)
    hi = float(np.percentile(bs, 97.5)) if bs else None
    return dict(ratio=ratio, ratio_hi=hi, block=int(b), n_blocks=len(blocks), within_step=within_step,
                T3_pass=bool(ratio is not None and ratio <= 1.5 and hi is not None and hi <= 2.0))


# --------------------------------------------------------------------------- T4 unseen blocks
def t4(X, n, s, prepped=False):
    lo, hi = SETS[s]; width = (hi + 1 - lo) // 9 if s != "S2" else 10; res = {}; passed = 0
    for d in BLOCK_LEADS:
        blo = d * (10 ** (len(str(lo)) - 1)); bhi = blo + width - 1
        te = (n >= blo) & (n <= bhi); tr = ~te
        if te.sum() < 5:
            continue
        T = None if prepped else prep_fit(X[tr]); Z = X if prepped else prep_apply(T, X)
        cur = fit_curve(Z[tr], n[tr], lo, hi, d); nh = place(cur, Z[te])
        rho = float(spearmanr(nh, n[te])[0]) if np.ptp(nh) > 0 else 0.0
        inside = float(np.mean((nh >= blo - width / 2) & (nh <= bhi + width / 2)))
        ok = bool(rho >= 0.80 and inside >= 0.80); passed += ok
        res[str(d)] = dict(block=[int(blo), int(bhi)], rho=rho, inside=inside, median_placed=float(np.median(nh)), passed=ok)
    return dict(blocks=res, n_pass=int(passed), T4_pass=bool(passed >= 2))


# --------------------------------------------------------------------------- power (planted curve / planted tear)
def power(X, n, s, tag, reps=10):
    T0 = prep_fit(X); Z = prep_apply(T0, X); lo, hi = SETS[s]
    h = choose_h(Z, n, 0); Tn = kernel_pred(Z, n, n, h); E = Z - Tn
    base = t3(Z, n, s, B=0); ws = base.get("within_step") or 0.0; L = lead(n); o = np.argsort(n)
    hits = dict(T1=0, T2=0, T4=0, T3_no_tear_on_curve=0, T3_detects_tear=0)
    import zlib
    seed = zlib.crc32(tag.encode()) % (2 ** 31)
    for rep in range(reps):
        r = np.random.default_rng(seed + rep); noise = E[r.permutation(len(n))]
        Zc = Tn + noise
        a = t1_t2(Zc, n, s, reps=1, shuffled=False, prepped=True); hits["T1"] += a["T1_pass"]; hits["T2"] += a["T2_pass"]
        hits["T4"] += t4(Zc, n, s, prepped=True)["T4_pass"]
        hits["T3_no_tear_on_curve"] += bool(t3(Zc, n, s, B=100)["T3_pass"])
        J = np.zeros_like(Z); cum = np.zeros(Z.shape[1]); prev = None
        for k in o:
            if prev is not None and L[k] != prev:
                u = r.standard_normal(Z.shape[1]); cum = cum + 3 * ws * u / np.linalg.norm(u)
            J[k] = cum; prev = L[k]
        hits["T3_detects_tear"] += not bool(t3(Zc + J, n, s, B=100)["T3_pass"])
    rate = {k: v / reps for k, v in hits.items()}
    rate["power_T1"] = rate["T1"] >= 0.8; rate["power_T2"] = rate["T2"] >= 0.8; rate["power_T4"] = rate["T4"] >= 0.8
    rate["power_T3"] = rate["T3_detects_tear"] >= 0.8
    return rate


# --------------------------------------------------------------------------- T5 contexts (new prompts)
def t5(get_X, contexts, n, s):
    lo, hi = SETS[s]; te = np.random.default_rng(0).random(len(n)) < 0.2; tr = ~te; raw = {}
    for c in contexts:
        X = get_X(c); raw[c] = X - X[tr].mean(0)
    rows = []
    for A in contexts:
        T = prep_fit(raw[A][tr]); ZA = prep_apply(T, raw[A]); cur = fit_curve(ZA[tr], n[tr], lo, hi, 0)
        for B in contexts:
            if B == A:
                continue
            nh = place(cur, prep_apply(T, raw[B])[te]); rows.append(dict(A=A, B=B, rho=float(spearmanr(nh, n[te])[0])))
    share = float(np.mean([r["rho"] >= 0.90 for r in rows])) if rows else None
    return dict(pairs=rows, share_ge_090=share, T5_pass=bool(share is not None and share >= 0.70))


# --------------------------------------------------------------------------- one (model, reading, set) job
def dev_split(n):
    rng = np.random.default_rng(C.SEED_DEV); full = rng.random(C.N_MAX + 2) < C.DEV_FRAC
    return full[np.asarray(n)]


def subsample(n, s, size=3000):
    if s == "S4" and len(n) > size:
        return np.sort(np.random.default_rng(0).choice(n, size, replace=False))
    return n


def run_job(job, layers_X, n, s, tag, do_power=True):
    """layers_X: list of (layer_label, X [len(n), d]). Chooses the layer on dev numbers, evaluates on the rest."""
    dev = dev_split(n); out = dict(job=job, set=s, N=int(len(n)), N_dev=int(dev.sum()), layers={})
    best = None
    for lab, X in layers_X:
        X = X() if callable(X) else X
        good = np.isfinite(X).all(1)
        if X[good].std(0).max() < 1e-6 or good.sum() < 30:
            continue
        Xd, nd = X[dev & good], n[dev & good]; Xe, ne = X[~dev & good], n[~dev & good]
        if len(nd) < 15 or len(ne) < 30:
            continue
        dv = t1_t2(Xd, nd, s, reps=3, shuffled=False)["rho"]
        a = t1_t2(Xe, ne, s, reps=3)
        T = prep_fit(Xe); Z = prep_apply(T, Xe)
        c3 = t3(Z, ne, s, B=200); c4 = t4(Xe, ne, s)
        out["layers"][str(lab)] = dict(dev_rho=dv, **a, **{k: v for k, v in c3.items()}, T4=c4, T4_pass=c4["T4_pass"])
        if best is None or dv > best[0]:
            best = (dv, lab, Xe, ne)
    if best is None:
        out["skipped"] = "no usable layer"; return out
    out["chosen_layer"] = best[1]; L = out["layers"][str(best[1])]
    out["shuffle_null"] = shuffle_null(best[2], best[3], s)
    L["T1_pass"] = bool(L["T1_pass"] and L["rho"] > out["shuffle_null"]["rho_q95"])        # Amendment 1
    out["T3_full"] = t3(prep_apply(prep_fit(best[2]), best[2]), best[3], s, B=500)
    if do_power:
        out["power"] = power(best[2], best[3], s, tag)
    p = out.get("power", {})
    out["verdict"] = dict(T1=L["T1_pass"] if p.get("power_T1", True) else "no power",
                          T2=L["T2_pass"] if p.get("power_T2", True) else "no power",
                          T3=out["T3_full"]["T3_pass"] if p.get("power_T3", True) else "no power",
                          T4=L["T4_pass"] if p.get("power_T4", True) else "no power")
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
