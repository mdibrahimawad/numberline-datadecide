"""The per-cell test panel (METHODOLOGY Part 3): M1 smooth part, M2 additive digits, M4 probes, H1, H2, P8 knee,
and the exponent probe. A cell = (model, condition, layer, set). tier="full" adds every control and the planted
power checks (E13 Stage-1 code); tier="light" keeps point estimates and the cheap controls (DEVIATIONS.md D5)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold

from src import e13_stage1 as E1
from src.e12c_heldout_r2 import fit_cd, kernel_pred, pred_cd

from . import config as C

HN = [0.01, 0.02, 0.05, 0.10, 0.20]          # bandwidth grid, fractions of the range; minimum 1 % (M1)
RIDGE_ALPHAS = np.logspace(-2, 4, 13)


def n_digits(s):
    return {"S1": 1, "S2": 2, "S3": 3, "S4": 4}[s]


def positions_for(L, units_only=False):
    """Powers of ten of the digit positions in the additive model (0 = units ... L-1 = leading)."""
    return [0] if units_only else list(range(L))


def r2(Zte, P, mu):
    return float(1 - ((Zte - P) ** 2).sum() / ((Zte - mu) ** 2).sum())


def choose_h(Ztr, ntr, seed, pos=None, model="curve"):
    R = max(ntr.max() - ntr.min(), 1); best = None
    for f in HN:
        h = max(1.0, f * R); err = 0.0
        for a, b in KFold(5, shuffle=True, random_state=seed).split(ntr):
            if model == "curve":
                P = kernel_pred(Ztr[a], ntr[a], ntr[b], h)
            else:
                P = pred_cd(fit_cd(Ztr[a], ntr[a], h, pos), ntr[a], ntr[b], h, pos)
            err += ((Ztr[b] - P) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, h)
    return best[1]


# --------------------------------------------------------------------------- M1 + M2
def splits(n, s, reps=3):
    out = []
    for rep in range(reps):
        te = np.random.default_rng(rep).random(len(n)) < 0.2; out.append((f"random{rep}", ~te, te))
    if s in C.BLOCKS:
        lo, hi = C.BLOCKS[s]; te = (n >= lo) & (n <= hi)
        if te.sum() >= 3 and (~te).sum() >= 10:
            out.append(("block", ~te, te))
    return out


def m12(X, n, s, full, seed=0):
    L = n_digits(s); pos = positions_for(L); pos_u = positions_for(L, True); res = {}
    for name, tr, te in splits(n, s):
        Z = E1.prep(X, np.where(tr)[0]); Ztr, Zte = Z[tr], Z[te]; ntr, nte = n[tr], n[te]; mu = Ztr.mean(0)
        h = choose_h(Ztr, ntr, seed); out = dict(h_curve=h / max(n.max() - n.min(), 1))
        out["R2_curve"] = r2(Zte, kernel_pred(Ztr, ntr, nte, h), mu)
        hc = choose_h(Ztr, ntr, seed, pos, "cd") if full else h
        out["R2_curve_digits"] = r2(Zte, pred_cd(fit_cd(Ztr, ntr, hc, pos), ntr, nte, hc, pos), mu)
        out["R2_curve_units"] = r2(Zte, pred_cd(fit_cd(Ztr, ntr, h, pos_u), ntr, nte, h, pos_u), mu)
        rng = np.random.default_rng(seed + 77); sh = []
        for _ in range(5 if full else 1):
            ns = rng.permutation(ntr); hs = choose_h(Ztr, ns, seed) if full else h
            sh.append(r2(Zte, kernel_pred(Ztr, ns, nte, hs), mu))
        out["R2_shuffled"] = float(np.mean(sh))
        res[name] = out
    rnd = [v for k, v in res.items() if k.startswith("random")]
    summ = {k: float(np.mean([v[k] for v in rnd])) for k in rnd[0]}
    if "block" in res:
        summ.update({f"block_{k}": v for k, v in res["block"].items()})
    # variance shares of f, digits, residual (in-sample backfit on all numbers, M2)
    Z = E1.prep(X); h = choose_h(Z, n, seed); m = fit_cd(Z, n, h, pos)
    f = kernel_pred(m["base"], n, n, h); from src.e12b_smooth_digits import digit_design
    Dd = digit_design(n, pos, m["means"])[0] @ m["coef"]; Eres = Z - f - Dd; tot = ((Z - Z.mean(0)) ** 2).sum()
    summ.update(share_f=float(((f - f.mean(0)) ** 2).sum() / tot), share_digits=float((Dd ** 2).sum() / tot),
                share_resid=float((Eres ** 2).sum() / tot))
    return dict(splits=res, **summ)


# --------------------------------------------------------------------------- M4 probes
def _ridge(Ztr, ytr):
    return RidgeCV(alphas=RIDGE_ALPHAS).fit(Ztr, ytr)


def sinusoid_decode(Zte, models, periods, cand):
    """Choose, for each held-out row, the candidate n maximising sum_T <(cos, sin) prediction, (cos, sin)(2 pi n / T)>."""
    score = np.zeros((len(Zte), len(cand)))
    for T in periods:
        cs = models[T].predict(Zte); ang = 2 * np.pi * cand / T
        score += cs[:, :1] * np.cos(ang)[None] + cs[:, 1:2] * np.sin(ang)[None]
    return cand[score.argmax(1)]


def m4(X, n, s, seed=0):
    L = n_digits(s); te = np.random.default_rng(seed).random(len(n)) < 0.2; tr = ~te
    Z = E1.prep(X, np.where(tr)[0]); Ztr, Zte = Z[tr], Z[te]; out = {}
    for name, y in (("n", n.astype(float)), ("logn", np.log(n.astype(float)))):
        p = _ridge(Ztr, y[tr]).predict(Zte); yt = y[te]
        out[f"probe_{name}_R2"] = float(1 - ((yt - p) ** 2).sum() / ((yt - y[tr].mean()) ** 2).sum())
        out[f"probe_{name}_spearman"] = float(spearmanr(p, yt)[0])
    R = n.max() - n.min() + 1
    periods = [T for T in (2, 5, 10, 100, 1000, 10000) if T <= 10 * R]
    models = {T: _ridge(Ztr, np.c_[np.cos(2 * np.pi * n[tr] / T), np.sin(2 * np.pi * n[tr] / T)]) for T in periods}
    cand = np.arange(C.SETS[s][0], C.SETS[s][1] + 1)
    dec = sinusoid_decode(Zte, models, periods, cand)
    out["sin_exact_acc"] = float(np.mean(dec == n[te])); out["sin_mae"] = float(np.mean(np.abs(dec - n[te])))
    scan = {}
    for T in np.unique(np.round(np.geomspace(2, max(3, 2 * R), 25)).astype(int)):
        yy = np.c_[np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)]
        p = _ridge(Ztr, yy[tr]).predict(Zte); scan[int(T)] = float(1 - ((yy[te] - p) ** 2).sum() / ((yy[te] - yy[tr].mean(0)) ** 2).sum())
    out["sin_scan_best_period"] = int(max(scan, key=scan.get)); out["sin_scan"] = scan
    for p in range(L):
        d = (n // 10 ** p) % 10; name = "units" if p == 0 else ("leading" if p == L - 1 else f"pos{p}")
        m = _ridge(Ztr, np.c_[np.cos(2 * np.pi * d[tr] / 10), np.sin(2 * np.pi * d[tr] / 10)]); cs = m.predict(Zte)
        dh = np.round(np.mod(np.arctan2(cs[:, 1], cs[:, 0]) / (2 * np.pi) * 10, 10)).astype(int) % 10
        out[f"circ_acc_{name}"] = float(np.mean(dh == d[te]))
        if len(np.unique(d[tr])) > 1:
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(Ztr, d[tr])
            out[f"lin_acc_{name}"] = float(np.mean(clf.predict(Zte) == d[te]))
    return out


# --------------------------------------------------------------------------- H1 / H2
def h1(Z, n, s, tag, full):
    L = n_digits(s)
    if L < 2:
        return None
    return E1.h1(Z, n, L, tag, B=200 if full else 50, n_perm=200 if full else 50, n_rep=20 if full else 0)


def h2(X, Z, n, s, tag, full):
    L = n_digits(s)
    if L < 2:
        return None
    big = len(n) > 3000
    if full:
        return E1.h2_all(X, Z, n, L, tag, big)
    i, j, c = E1.steps_of(n); out = dict(n_steps={int(k): int((c == k).sum()) for k in np.unique(c)} if len(c) else {})
    out["h2a"] = E1.h2a(Z, n, L, tag, i, j, c, B=100, n_perm=50) if len(c) else None
    b = E1.h2b(X, n, tag, i, j, c, B=100, reps=1) if len(c) else None
    if b is not None:
        b.pop("_boot", None)
    out["h2b"] = b
    out["h2c"] = E1.h2c(X, n, L, tag, big=big, B=200, light=True)
    out["h2c"]["within_group_line_R2"] = E1.within_group_line(Z, n, L, tag)
    return out


# --------------------------------------------------------------------------- P8 knee
def latent_features(n, L, tr):
    S = E1.ns_basis(n, 10); Xd, _ = E1.design(E1.digits(n, L))
    return np.column_stack([Xd, S])


def knee_curve(Z, n, L, tr, te, kmax=30):
    M = latent_features(n, L, tr); b = np.linalg.lstsq(M[tr], Z[tr], rcond=None)[0]
    Ftr = M[tr] @ b; Pte = M[te] @ b; mu = Ftr.mean(0); V = np.linalg.svd(Ftr - mu, full_matrices=False)[2]
    zmu = Z[tr].mean(0); sst = ((Z[te] - zmu) ** 2).sum(); out = {}
    for k in range(1, kmax + 1):
        k2 = min(k, V.shape[0]); Pk = mu + (Pte - mu) @ V[:k2].T @ V[:k2]
        out[k] = float(1 - ((Z[te] - Pk) ** 2).sum() / sst)
    return out


def knee_stats(curve, kref=10, frac=0.90):
    ref = curve[kref]
    if not np.isfinite(ref) or ref <= 0:
        return dict(knee=None, sharpness=None, R2_ref=ref)
    knee = next(k for k in sorted(curve) if k <= kref and curve[k] >= frac * ref) if any(
        curve[k] >= frac * ref for k in curve if k <= kref) else None
    sharp = float(np.mean([max(curve[k], 0) / ref for k in range(1, kref + 1)]))
    return dict(knee=knee, sharpness=sharp, R2_ref=ref)


def p8(X, n, s, tag):
    L = n_digits(s); rng = np.random.default_rng(E1.seed_of(tag, "p8")); te = rng.random(len(n)) < 0.2; tr = ~te
    Z = E1.prep(X, np.where(tr)[0]); out = {}
    cur = knee_curve(Z, n, L, tr, te); out["random"] = dict(curve=cur, **knee_stats(cur))
    if s in C.BLOCKS:
        lo, hi = C.BLOCKS[s]; teb = (n >= lo) & (n <= hi)
        if teb.sum() >= 3:
            Zb = E1.prep(X, np.where(~teb)[0]); cb = knee_curve(Zb, n, L, ~teb, teb); out["block"] = dict(curve=cb, **knee_stats(cb))
    # controls: shuffled n; flat sheet (x = n, y ~ U) with the real residual covariance as noise
    Zs = Z[rng.permutation(len(Z))]; cs = knee_curve(Zs, n, L, tr, te); out["ctrl_shuffled"] = dict(curve=cs, **knee_stats(cs))
    M = latent_features(n, L, tr); b = np.linalg.lstsq(M, Z, rcond=None)[0]; E = Z - M @ b
    u = E1.orthonormal(Z.shape[1], 2, rng); x = (n - n.min()) / max(n.max() - n.min(), 1); y = rng.random(len(n))
    sig = np.sqrt(((M @ b - (M @ b).mean(0)) ** 2).sum(1).mean() / 2)
    Zf = np.c_[x - x.mean(), (y - y.mean())] / np.r_[x.std(), y.std()] * sig @ u.T + E[rng.permutation(len(Z))]
    cf = knee_curve(Zf, n, L, tr, te); out["ctrl_sheet"] = dict(curve=cf, **knee_stats(cf))
    out["expected_H1"] = 1 + 9 * (L - 1)
    sr = out["random"]["sharpness"]
    out["sharper_than_controls"] = bool(sr is not None and all(
        out[k]["sharpness"] is None or sr > out[k]["sharpness"] for k in ("ctrl_shuffled", "ctrl_sheet")))
    return out


# --------------------------------------------------------------------------- exponent (digit count)
def exponent(get_X, sets_numbers, tag):
    """get_X(numbers) -> matrix. Train a ridge probe of length on lengths {2, 3}; test on 1 and 4 (where present)."""
    lens = {s: n for s, n in sets_numbers.items() if len(n) >= 5}
    if not ({"S2", "S3"} <= set(lens)):
        return None
    sub = {s: (np.sort(np.random.default_rng(0).choice(n, 900, replace=False)) if len(n) > 900 else n) for s, n in lens.items()}
    n_all = np.concatenate([sub[s] for s in sub]); y = np.concatenate([[n_digits(s)] * len(sub[s]) for s in sub]).astype(float)
    X = get_X(n_all); tr = np.isin(y, [2, 3]); Z = E1.prep(X, np.where(tr)[0])
    rg = Ridge(alpha=1.0).fit(Z[tr], y[tr]); m2 = rg.predict(Z[y == 2]).mean(); m3 = rg.predict(Z[y == 3]).mean(); out = {}
    for Lh in (1, 4):
        k = y == Lh
        if k.any():
            ph = float(rg.predict(Z[k]).mean()); out[f"pred_len{Lh}"] = ph
            out[f"ordered_len{Lh}"] = bool(ph < m2) if Lh == 1 else bool(ph > m3)
    out.update(train_mean_len2=float(m2), train_mean_len3=float(m3))
    return out


# --------------------------------------------------------------------------- one cell
def run_cell(X, n, s, tag, tier="light"):
    full = tier == "full"; out = dict(N=int(len(n)), tier=tier)
    Xs = X; ns = n
    if s == "S4" and len(n) > C.S4_SUBSAMPLE:
        from .data import subsample_s4
        ns = subsample_s4(n); Xs = X[np.searchsorted(n, ns)]
    out["m12"] = m12(Xs, ns, s, full)
    out["m4"] = m4(Xs, ns, s)
    out["p8"] = p8(Xs, ns, s, tag)
    Z = E1.prep(X)
    out["h1"] = h1(Z, n, s, tag, full)
    out["h2"] = h2(X, Z, n, s, tag, full)
    return out
