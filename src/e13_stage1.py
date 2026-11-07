"""E13 Stage 1: digit-position codes (H1) and carry structure (H2). Pre-registration:
experiments/E13_number_code/PREREG_stage1.md. One job = (model, depth); one JSON per (model, set, depth); resumable.

  python -m src.e13_stage1 --data <topology_number_line>/data --out results/e13_stage1 --workers 4
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.extmath import randomized_svd

HERE = os.path.dirname(os.path.abspath(__file__))
TOKCOUNTS = os.path.join(HERE, "..", "experiments", "E13_number_code", "token_counts_in_prompt.json")
MODELS = {  # name: (family, subfamily)
    "crystal": ("digit", "digit"), "openllama-3b": ("digit", "digit"), "starcoderbase-3b": ("digit", "digit"),
    "qwen2.5-1.5b": ("digit", "digit"), "smollm2-1.7b": ("digit", "digit"),
    "pythia-1.4b": ("chunk", "neox"), "redpajama-3b": ("chunk", "neox"),
    "olmo2-1b": ("chunk", "three"), "llama3.2-1b": ("chunk", "three")}
DISCOVERY = {"qwen2.5-1.5b", "pythia-1.4b"}
DEPTH = ["~25%", "~50%", "~75%", "~84%", "last"]
RANGES = {"S2": (10, 99), "S3": (100, 999), "S4": (1000, 9999), "S4x": (1000, 9999)}
PCA_D = 50


# --------------------------------------------------------------------------- data
def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


_TC = None


def token_counts(model):
    global _TC
    if _TC is None:
        _TC = json.load(open(TOKCOUNTS))
    return np.array(_TC[model])


def sets_for(model):
    sub = MODELS[model][1]
    return ["S2", "S3", "S4"] if sub != "three" else ["S2", "S3", "S4x"]


def set_numbers(model, s):
    lo, hi = RANGES[s]; n = np.arange(lo, hi + 1)
    if MODELS[model][0] == "chunk" and s != "S4x":
        n = n[token_counts(model)[n - 1] == 1]
    return n


def layer_files(data, model):
    fs = sorted(glob.glob(os.path.join(data, f"{model}_vectors", "layer_*.npz")),
                key=lambda f: int(os.path.basename(f)[6:-4]))
    assert len(fs) == 5, fs
    return fs


def digits(n, L):
    return np.stack([(n // 10 ** (L - 1 - p)) % 10 for p in range(L)], 1)


# --------------------------------------------------------------------------- preparation
def prep(X, fit_idx=None, variant="main", D=PCA_D):
    """Centre, z-score (unless variant='nozscore'), PCA to D; transform fitted on rows fit_idx, applied to all rows."""
    X = np.asarray(X, dtype=np.float64)
    if variant == "drop3":
        F = X if fit_idx is None else X[fit_idx]
        keep = np.ones(X.shape[1], bool); keep[np.argsort(F.var(0))[-3:]] = False; X = X[:, keep]
    F = X if fit_idx is None else X[fit_idx]
    mu = F.mean(0)
    if variant == "nozscore":
        sd = np.ones(X.shape[1]); keep = np.ones(X.shape[1], bool)
    else:
        sd = F.std(0); keep = sd > 1e-8
    A = (F[:, keep] - mu[keep]) / sd[keep]; D = min(D, len(F) - 1, A.shape[1])
    if len(A) > 2000 and A.shape[1] > 200:
        _, _, Vt = randomized_svd(A, D, n_iter=6, random_state=0)
    else:
        Vt = np.linalg.svd(A, full_matrices=False)[2][:D]
    return ((X[:, keep] - mu[keep]) / sd[keep]) @ Vt.T


# --------------------------------------------------------------------------- additive digit model
def design(dig):
    N, L = dig.shape; cols = [np.ones(N)]; info = []
    for p in range(L):
        levels = np.unique(dig[:, p]); idx = {}
        for v in levels[1:]:
            idx[int(v)] = len(cols); cols.append((dig[:, p] == v).astype(float))
        info.append((levels.astype(int), idx))
    return np.stack(cols, 1), info


def additive(Z, dig):
    X, info = design(dig)
    G = np.linalg.pinv(X.T @ X); B = G @ (X.T @ Z); F = X @ B; R = Z - F
    rank = np.linalg.matrix_rank(X); sigma2 = float((R ** 2).sum() / max(len(Z) - rank, 1))
    pos = []
    for levels, idx in info:
        K = len(levels); E = np.zeros((K, X.shape[1]))
        for a, v in enumerate(levels):
            if int(v) in idx:
                E[a, idx[int(v)]] = 1.0
        C = E @ B; C = C - C.mean(0)
        M = E @ G @ E.T; dM = np.diag(M); Q = dM[:, None] + dM[None, :] - 2 * M
        pos.append(dict(levels=levels, C=C, Q=Q))
    return dict(pos=pos, F=F, R=R, sigma2=sigma2, B=B, X=X, G=G)


def debiased_d2(C, Q, sigma2):
    sq = (C ** 2).sum(1); D2 = sq[:, None] + sq[None, :] - 2 * C @ C.T
    return D2 - sigma2 * Q


TEMPLATES = ("line", "circle", "simplex")


def template_t(levels, kind):
    a = levels[:, None]; b = levels[None, :]; dv = np.abs(a - b).astype(float)
    if kind == "line":
        return dv
    if kind == "circle":
        return 2 * np.sin(np.pi * dv / 10)
    return np.ones_like(dv)


def pos_stats(D2, levels, tot_var, seed, full=True):
    K = len(levels); iu = np.triu_indices(K, 1); d = np.sqrt(np.clip(D2[iu], 0, None))
    dv = np.abs(levels[iu[0]] - levels[iu[1]])
    o1 = spearmanr(d, dv)[0] if d.std() > 0 else np.nan
    out = dict(O1=float(o1), V=float(D2[iu].mean() / 2 / tot_var))
    if not full:
        return out
    J = np.eye(K) - 1.0 / K; Bm = -0.5 * J @ D2 @ J; w, U = np.linalg.eigh(Bm); w = w[::-1]; U = U[:, ::-1]
    lam = np.clip(w, 0, None)
    out["PR"] = float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else np.nan
    ax = U[:, 0] * np.sqrt(lam[0]) if lam[0] > 0 else np.zeros(K)
    out["A1"] = float(lam[0] / lam.sum() * abs(spearmanr(ax, levels)[0])) if lam.sum() > 0 and ax.std() > 0 else np.nan
    cv = {}
    kf = KFold(5, shuffle=True, random_state=seed % (2 ** 31))
    for kind in TEMPLATES:
        t = template_t(levels, kind)[iu]; err = 0.0
        for tr, te in kf.split(d):
            s = (t[tr] @ d[tr]) / (t[tr] @ t[tr]); err += ((d[te] - s * t[te]) ** 2).sum()
        cv[kind] = err / len(d)
        s = (t @ d) / (t @ t); sst = ((d - d.mean()) ** 2).sum()
        out[f"R2_{kind}"] = float(1 - ((d - s * t) ** 2).sum() / sst) if sst > 0 else np.nan
        out[f"cvmse_{kind}"] = float(cv[kind])
    out["template"] = min(cv, key=cv.get) if d.max() > 0 else "none"
    return out


def o1_all(Z, dig):
    m = additive(Z, dig); tv = float(((Z - Z.mean(0)) ** 2).sum(1).mean())
    return [pos_stats(debiased_d2(p["C"], p["Q"], m["sigma2"]), p["levels"], tv, 0, full=False)["O1"] for p in m["pos"]]


def decay_pass(o1, lo_ci):
    L = len(o1); delta = o1[0] - o1[-1]
    mono = all(o1[p + 1] <= o1[p] + 0.05 for p in range(L - 1))
    return bool(delta > 0.30 and lo_ci > 0 and mono)


def boot_o1(Z, dig, B, rng):
    out = []
    for _ in range(B):
        i = rng.integers(0, len(Z), len(Z))
        try:
            out.append(o1_all(Z[i], dig[i]))
        except Exception:
            continue
    return np.array(out)


# --------------------------------------------------------------------------- fakes
def orthonormal(D, k, rng):
    return np.linalg.qr(rng.standard_normal((D, k)))[0][:, :k]


def planted_template(levels, kind, target_v, D, rng):
    """Centroids (K x D) of a template with mean pairwise squared distance / 2 = target_v (absolute variance)."""
    K = len(levels)
    if kind == "line":
        T = (levels - levels.mean())[:, None] * orthonormal(D, 1, rng).T
    elif kind == "circle":
        ang = 2 * np.pi * levels / 10
        T = np.stack([np.cos(ang), np.sin(ang)], 1) @ orthonormal(D, 2, rng).T
    else:
        T = (np.eye(K) - 1.0 / K) @ orthonormal(D, K, rng).T
    T = T - T.mean(0); sq = (T ** 2).sum(1); D2 = sq[:, None] + sq[None, :] - 2 * T @ T.T
    cur = D2[np.triu_indices(K, 1)].mean() / 2
    return T * np.sqrt(max(target_v, 0) / cur) if cur > 0 else T * 0


def matched_smooth(Z, n, rng, max_pairs=2_000_000):
    """Stationary GP per PCA dimension with the set's variogram over |m - n| (E12 (d)); pair subsample if N > 3000."""
    lo, hi = n.min(), n.max(); G = hi - lo + 1; idx = n - lo; N = len(n); out = np.zeros_like(Z)
    if N <= 3000:
        i, j = np.triu_indices(N, 1)
    else:
        i = rng.integers(0, N, max_pairs); j = rng.integers(0, N, max_pairs); k = i != j; i, j = i[k], j[k]
    dl = np.abs(n[i] - n[j]); cnt = np.bincount(dl, minlength=G); have = cnt > 0
    for c in range(Z.shape[1]):
        z = Z[:, c] - Z[:, c].mean(); var = z.var()
        g = np.bincount(dl, weights=0.5 * (z[i] - z[j]) ** 2, minlength=G) / np.maximum(cnt, 1)
        g = np.interp(np.arange(G), np.arange(G)[have], g[have]); g[0] = 0.0
        g = uniform_filter1d(g, 5, mode="nearest"); g[0] = 0.0
        row = np.r_[var - g, (var - g)[-2:0:-1]]
        lam = np.clip(np.fft.fft(row).real, 0, None)
        xi = rng.standard_normal(len(row)) + 1j * rng.standard_normal(len(row))
        path = np.fft.ifft(np.sqrt(lam) * xi).real[:G]; s = path.std()
        out[:, c] = path[idx] / (s if s > 0 else 1) * np.sqrt(var)
    return out


def trend(Z, n):
    """Gaussian kernel smoother over n, bandwidth by leave-one-out CV from {1,2,5,10,20} % of the range (min 1)."""
    R = max(n.max() - n.min(), 1); best = None; nf = n.astype(float)
    for f in (0.01, 0.02, 0.05, 0.10, 0.20):
        h = max(1.0, f * R)
        W = np.exp(-0.5 * ((nf[:, None] - nf[None, :]) / h) ** 2); np.fill_diagonal(W, 0.0)
        P = (W @ Z) / np.maximum(W.sum(1, keepdims=True), 1e-300); err = float(((Z - P) ** 2).sum())
        del W
        if best is None or err < best[0]:
            best = (err, h)
    h = best[1]; W = np.exp(-0.5 * ((nf[:, None] - nf[None, :]) / h) ** 2)
    T = (W @ Z) / W.sum(1, keepdims=True)
    return T, h


# --------------------------------------------------------------------------- H1
def h1(Z, n, L, tag, B=200, n_perm=200, n_rep=20):
    dig = digits(n, L); rng = np.random.default_rng(seed_of(tag, "h1"))
    tv = float(((Z - Z.mean(0)) ** 2).sum(1).mean()); m = additive(Z, dig)
    res = dict(positions=[])
    for p, P in enumerate(m["pos"]):
        D2 = debiased_d2(P["C"], P["Q"], m["sigma2"])
        st = pos_stats(D2, P["levels"], tv, seed_of(tag, "tmpl", p)); st["K"] = len(P["levels"])
        # digit-label shuffle null for O1
        K = len(P["levels"]); iu = np.triu_indices(K, 1); dv = np.abs(P["levels"][iu[0]] - P["levels"][iu[1]])
        null = []
        for _ in range(n_perm):
            pm = rng.permutation(K); Dp = D2[np.ix_(pm, pm)]; d = np.sqrt(np.clip(Dp[iu], 0, None))
            null.append(spearmanr(d, dv)[0] if d.std() > 0 else 0.0)
        st["O1_label_shuffle_mean"] = float(np.nanmean(null))
        st["O1_label_shuffle_p"] = float((1 + np.sum(np.array(null) >= st["O1"])) / (n_perm + 1)) if np.isfinite(st["O1"]) else np.nan
        res["positions"].append(st)
    # shuffled-n null for the signal V_p
    Vnull = []
    for _ in range(n_perm):
        mp = additive(Z[rng.permutation(len(Z))], dig)
        Vnull.append([float(debiased_d2(P["C"], P["Q"], mp["sigma2"])[np.triu_indices(len(P["levels"]), 1)].mean() / 2 / tv)
                      for P in mp["pos"]])
    Vnull = np.array(Vnull)
    for p, st in enumerate(res["positions"]):
        st["V_shuffle_p"] = float((1 + np.sum(Vnull[:, p] >= st["V"])) / (n_perm + 1))
        st["V_shuffle_mean"] = float(Vnull[:, p].mean())
    # bootstrap CIs
    bo = boot_o1(Z, dig, B, rng)
    o1 = [st["O1"] for st in res["positions"]]
    for p, st in enumerate(res["positions"]):
        st["O1_lo"], st["O1_hi"] = [float(x) for x in np.nanpercentile(bo[:, p], [2.5, 97.5])]
    dl = bo[:, 0] - bo[:, -1]; res["delta"] = float(o1[0] - o1[-1])
    res["delta_lo"], res["delta_hi"] = [float(x) for x in np.nanpercentile(dl, [2.5, 97.5])]
    res["decay_pass"] = decay_pass(o1, res["delta_lo"])
    res["Ra"] = bool(all(o > 0.5 for o in o1) and max(o1) - min(o1) < 0.1)
    res["Rb"] = bool(all(o < 0.3 for o in o1))
    # matched-smoothness fake
    Zf = matched_smooth(Z, n, np.random.default_rng(seed_of(tag, "h1_ms")))
    mf = additive(Zf, dig); tvf = float(((Zf - Zf.mean(0)) ** 2).sum(1).mean()); fk = []
    for p, P in enumerate(mf["pos"]):
        fk.append(pos_stats(debiased_d2(P["C"], P["Q"], mf["sigma2"]), P["levels"], tvf, seed_of(tag, "tmplf", p)))
    res["fake_ms"] = dict(O1=[f["O1"] for f in fk], V=[f["V"] for f in fk], template=[f["template"] for f in fk],
                          delta=float(fk[0]["O1"] - fk[-1]["O1"]))
    # planted power
    Vabs = [max(st["V"] * tv, 0.0) for st in res["positions"]]
    real_parts = [P["C"][np.searchsorted(P["levels"], dig[:, p])] for p, P in enumerate(m["pos"])]
    base = Z.mean(0); Rres = m["R"]; Dd = Z.shape[1]

    def fake_with(plants, r):
        parts = []
        for p, P in enumerate(m["pos"]):
            if p in plants:
                T = planted_template(P["levels"], plants[p], Vabs[p], Dd, r)
                parts.append(T[np.searchsorted(P["levels"], dig[:, p])])
            else:
                parts.append(real_parts[p])
        return base + sum(parts) + Rres[r.permutation(len(Z))]

    rec = {}
    for p, P in enumerate(m["pos"]):
        for kind in TEMPLATES:
            hits = 0
            for rep in range(n_rep):
                r = np.random.default_rng(seed_of(tag, "plant", p, kind, rep)); Zp = fake_with({p: kind}, r)
                mp = additive(Zp, dig); tvp = float(((Zp - Zp.mean(0)) ** 2).sum(1).mean()); Pp = mp["pos"][p]
                st = pos_stats(debiased_d2(Pp["C"], Pp["Q"], mp["sigma2"]), Pp["levels"], tvp, seed_of(tag, "pt", p, rep))
                hits += st["template"] == kind
            rec[f"p{p}_{kind}"] = hits / n_rep
    res["template_recovery"] = rec
    for name, plants in (("decay_power", {0: "line", L - 1: "simplex"}), ("decay_fp", {p: "line" for p in range(L)})):
        ok = 0
        for rep in range(n_rep):
            r = np.random.default_rng(seed_of(tag, name, rep)); Zp = fake_with(plants, r)
            o = o1_all(Zp, dig); b = boot_o1(Zp, dig, 50, r); lo = np.nanpercentile(b[:, 0] - b[:, -1], 2.5)
            ok += decay_pass(o, lo)
        res[name] = ok / n_rep
    res["has_power"] = bool(res["decay_power"] >= 0.8 and res["decay_fp"] <= 0.2)
    u = res["positions"][-1]
    res["Rc"] = bool(u["template"] == "circle" and rec[f"p{L-1}_circle"] >= 0.8)
    res["units_simplex_signal"] = bool(u["template"] == "simplex" and u["V_shuffle_p"] < 0.01)
    res["beyond_smooth"] = bool(res["delta"] - res["fake_ms"]["delta"] > 0.10)
    return res


# --------------------------------------------------------------------------- H2 helpers
def trailing_nines(n):
    c = np.zeros(len(n), int); m = n.copy()
    while True:
        k = (m % 10 == 9) & (m > 0)
        if not k.any():
            break
        c[k] += 1; m = np.where(k, m // 10, -1)
    return c


def steps_of(n):
    pos = {int(v): i for i, v in enumerate(n)}
    i = np.array([pos[int(v)] for v in n if int(v) + 1 in pos], int)
    j = np.array([pos[int(n[k]) + 1] for k in i], int)
    return i, j, trailing_nines(n[i])


def ns_basis(x, df):
    """Natural cubic spline basis (ESL 5.4), df non-constant columns, knots at quantiles."""
    K = df + 1; kn = np.quantile(x, np.linspace(0, 1, K)); x = x.astype(float)

    def dk(k):
        return (np.clip(x - kn[k], 0, None) ** 3 - np.clip(x - kn[-1], 0, None) ** 3) / (kn[-1] - kn[k])
    cols = [x] + [dk(k) - dk(K - 2) for k in range(K - 2)]
    M = np.stack(cols, 1)
    return (M - M.mean(0)) / np.maximum(M.std(0), 1e-12)


def h2_levels(c, min_steps=5):
    return [j for j in range(1, c.max() + 1) if (c == j).sum() >= min_steps] if len(c) else []


# --------------------------------------------------------------------------- H2a
def h2a_fit(Z, n, L, i, j, c, levels, add=None, free_b=False):
    """Amendment 1: primary = offset model, y = s^2 - debiased additive step^2 (coefficient fixed at 1);
    free_b=True gives the originally pre-registered free-coefficient version (secondary)."""
    if add is None:
        add = additive(Z, digits(n, L))
    F, Xd, G, s2n = add["F"], add["X"], add["G"], add["sigma2"]
    s2 = ((Z[j] - Z[i]) ** 2).sum(1); dX = Xd[j] - Xd[i]
    sa = ((F[j] - F[i]) ** 2).sum(1) - s2n * np.einsum("ij,jk,ik->i", dX, G, dX)
    keep = (c == 0) | np.isin(c, levels); S = ns_basis(n[i], 6)
    dums = [(c == lv).astype(float) for lv in levels]
    if free_b:
        X = np.column_stack([np.ones(len(i)), S, sa] + dums); y = s2
    else:
        X = np.column_stack([np.ones(len(i)), S] + dums); y = s2 - sa
    y1 = np.sqrt(s2) - np.sqrt(np.clip(sa, 0, None))
    return X[keep], y[keep], y1[keep], (n[i] // 10)[keep]


def ols_coef(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def h2a(Z, n, L, tag, i, j, c, B=500, n_perm=200, add=None, rng=None, full=True):
    levels = h2_levels(c)
    if (c == 0).sum() < 30 or 1 not in levels:
        return None
    add = add if add is not None else additive(Z, digits(n, L))
    X, y, y1, cl = h2a_fit(Z, n, L, i, j, c, levels, add)
    k0 = X.shape[1] - len(levels); b = ols_coef(X, y); e = b[k0:]
    Xf, yf, _, _ = h2a_fit(Z, n, L, i, j, c, levels, add, free_b=True); bf = ols_coef(Xf, yf)
    out = dict(levels=levels, e=[float(v) for v in e], e_s=[float(v) for v in ols_coef(X, y1)[k0:]],
               e_freeb=[float(v) for v in bf[k0 + 1:]], b_freeb=float(bf[k0]),
               n_steps={int(lv): int((X[:, k0 + a] == 1).sum()) for a, lv in enumerate(levels)})
    out["mean_y_level0"] = float(y[(X[:, k0:] == 0).all(1)].mean())
    rng = rng or np.random.default_rng(seed_of(tag, "h2a"))
    ucl = np.unique(cl); groups = {g: np.where(cl == g)[0] for g in ucl}; bs = []
    for _ in range(B):
        pick = np.concatenate([groups[g] for g in rng.choice(ucl, len(ucl))])
        Xb = X[pick]
        if any(Xb[:, k0 + a].sum() == 0 for a in range(len(levels))):
            continue
        bs.append(ols_coef(Xb, y[pick])[k0:])
    bs = np.array(bs)
    out["e_lo"] = [float(v) for v in np.percentile(bs, 2.5, 0)]; out["e_hi"] = [float(v) for v in np.percentile(bs, 97.5, 0)]
    diffs = [bs[:, a + 1] - bs[:, a] for a in range(len(levels) - 1)]
    out["diff_lo"] = [float(np.percentile(d, 2.5)) for d in diffs]; out["diff_hi"] = [float(np.percentile(d, 97.5)) for d in diffs]
    out["diff"] = [float(e[a + 1] - e[a]) for a in range(len(levels) - 1)]
    has2 = 2 in levels
    out["pass"] = bool(out["e_lo"][0] > 0 and (not has2 or out["diff_lo"][0] > 0))
    out["has_level2"] = bool(has2)
    if full and n_perm:
        D = X[:, k0:].copy(); null = []
        for _ in range(n_perm):
            Xp = X.copy(); Xp[:, k0:] = D[rng.permutation(len(D))]; bp = ols_coef(Xp, y)[k0:]
            null.append([bp[0]] + ([bp[1] - bp[0]] if has2 else []))
        null = np.array(null)
        out["perm_p_e1"] = float((1 + (null[:, 0] >= e[0]).sum()) / (n_perm + 1))
        if has2:
            out["perm_p_diff12"] = float((1 + (null[:, 1] >= e[1] - e[0]).sum()) / (n_perm + 1))
    return out


# --------------------------------------------------------------------------- H2b
def h2b(X, n, tag, i, j, c, B=500, prep_fn=None, reps=3):
    levels = h2_levels(c)
    if (c == 0).sum() < 30 or 1 not in levels:
        return None
    prep_fn = prep_fn or (lambda A, idx: prep(A, idx))
    z0 = np.where(c == 0)[0]; res = []
    for rep in range(reps):
        rng = np.random.default_rng(seed_of(tag, "h2b", rep)); te0 = rng.random(len(z0)) < 0.2
        tr = z0[~te0]; te = z0[te0]
        fit = np.unique(np.r_[i[tr], j[tr]]); Z = prep_fn(X, fit)
        A, Bv = Z[i[tr]], Z[j[tr]]; best = None
        for a in (0.1, 1, 10, 100, 1000):
            err = 0.0
            for ftr, fte in KFold(5, shuffle=True, random_state=rep).split(A):
                mdl = Ridge(alpha=a).fit(A[ftr], Bv[ftr]); err += ((Bv[fte] - mdl.predict(A[fte])) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, a)
        mdl = Ridge(alpha=best[1]).fit(A, Bv); mu = Bv.mean(0); dmu = (Bv - A).mean(0)
        sets = {0: te}; sets.update({lv: np.where(c == lv)[0] for lv in levels}); sets["ge1"] = np.where(c >= 1)[0]
        sets["ge1"] = sets["ge1"][np.isin(c[sets["ge1"]], levels)]
        pieces = {}
        for k, idx in sets.items():
            P = mdl.predict(Z[i[idx]]); Y = Z[j[idx]]
            pieces[k] = (((Y - P) ** 2).sum(1), ((Y - mu) ** 2).sum(1),
                         (((Y - Z[i[idx]]) - (P - Z[i[idx]])) ** 2).sum(1), (((Y - Z[i[idx]]) - dmu) ** 2).sum(1))
        r = {"alpha": best[1]}
        for k, (e, t, ed, td) in pieces.items():
            r[f"R2_{k}"] = float(1 - e.sum() / t.sum()); r[f"R2step_{k}"] = float(1 - ed.sum() / td.sum())
        r["drop"] = r["R2_0"] - r["R2_ge1"]; bs = []
        for _ in range(B):
            a0 = rng.integers(0, len(pieces[0][0]), len(pieces[0][0])); a1 = rng.integers(0, len(pieces["ge1"][0]), len(pieces["ge1"][0]))
            bs.append((1 - pieces[0][0][a0].sum() / pieces[0][1][a0].sum()) - (1 - pieces["ge1"][0][a1].sum() / pieces["ge1"][1][a1].sum()))
        r["boot"] = bs; res.append(r)
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k not in ("boot",)}
    bs = np.mean([r["boot"] for r in res], 0)
    out["drop_lo"], out["drop_hi"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    out["pass"] = bool(out["drop"] > 0.20 and out["drop_lo"] > 0); out["levels"] = levels
    out["_boot"] = bs
    return out


# --------------------------------------------------------------------------- H2c
def prefix_feats(n, L):
    return [n // 10 ** (L - l) for l in range(1, L)]          # prefixes of length 1..L-1


def tree_X(pre, a, b):
    return np.column_stack([np.ones(len(a))] + [(p[a] != p[b]).astype(float) for p in pre])


class FastIso:
    """Increasing isotonic fit for integer-valued x: aggregate ties, weighted PAVA on unique x (same solution as
    sklearn on the raw pairs), predict by linear interpolation with clipping (sklearn's out_of_bounds='clip')."""
    def fit(self, x, y):
        ux, inv = np.unique(x, return_inverse=True); cnt = np.bincount(inv); m = np.bincount(inv, weights=y) / cnt
        self.ux = ux.astype(float)
        self.uy = IsotonicRegression(increasing=True).fit(self.ux, m, sample_weight=cnt).predict(self.ux)
        return self

    def predict(self, x):
        return np.interp(x, self.ux, self.uy)


def fit_models(x_tr, T_tr, y_tr, x_te, T_te):
    """line (isotonic in integer |m-n|), tree (OLS on prefix indicators), both (backfit, 10 alternations)."""
    ux, inv = np.unique(x_tr, return_inverse=True); cnt = np.bincount(inv); uxf = ux.astype(float)
    grid = np.arange(0, int(max(ux.max(), x_te.max())) + 1, dtype=float)

    def iso_table(yv):
        m = np.bincount(inv, weights=yv) / cnt
        uy = IsotonicRegression(increasing=True).fit(uxf, m, sample_weight=cnt).predict(uxf)
        return np.interp(grid, uxf, uy)                 # identical to sklearn predict with out_of_bounds="clip"
    P = np.linalg.pinv(T_tr.T @ T_tr)
    tab = iso_table(y_tr); p_line = tab[x_te]
    bt = P @ (T_tr.T @ y_tr); p_tree = T_te @ bt; g = T_tr @ bt
    for _ in range(10):
        tab2 = iso_table(y_tr - g + g.mean()); f = tab2[x_tr] - g.mean()
        bb = P @ (T_tr.T @ (y_tr - f)); g = T_tr @ bb
    p_both = tab2[x_te] - g.mean() + T_te @ bb
    return p_line, p_tree, p_both


def sample_pairs(idx_a, idx_b, cap, rng, same=False):
    """Random distinct pairs (a from idx_a, b from idx_b); all pairs if few."""
    if same:
        m = len(idx_a)
        if m * (m - 1) // 2 <= cap:
            u, v = np.triu_indices(m, 1); return idx_a[u], idx_a[v]
        a = idx_a[rng.integers(0, m, cap)]; b = idx_a[rng.integers(0, m, cap)]; k = a != b; return a[k], b[k]
    if len(idx_a) * len(idx_b) <= cap:
        a, b = np.meshgrid(idx_a, idx_b, indexing="ij"); a, b = a.ravel(), b.ravel(); k = a != b; return a[k], b[k]
    a = idx_a[rng.integers(0, len(idx_a), cap)]; b = idx_b[rng.integers(0, len(idx_b), cap)]; k = a != b
    return a[k], b[k]


def h2c(X, n, L, tag, prep_fn=None, big=False, B=1000, light=False):
    prep_fn = prep_fn or (lambda A, idx: prep(A, idx))
    rng = np.random.default_rng(seed_of(tag, "h2c")); pre = prefix_feats(n, L); lead = pre[0]; N = len(n)
    cap_tr, cap_te = (100_000, 50_000) if big else (200_000, 200_000)
    out = {}
    # (ii) leave one leading-digit group out
    fold = []
    for g in np.unique(lead):
        te_num = np.where(lead == g)[0]; tr_num = np.where(lead != g)[0]
        if len(te_num) < 2 or len(tr_num) < 2:
            continue
        Z = prep_fn(X, tr_num)
        a, b = sample_pairs(tr_num, None, cap_tr, rng, same=True)
        ta, tb = sample_pairs(te_num, np.arange(N), cap_te, rng)
        y = ((Z[a] - Z[b]) ** 2).sum(1); yt = ((Z[ta] - Z[tb]) ** 2).sum(1)
        pl, pt, pb = fit_models(np.abs(n[a] - n[b]), tree_X(pre, a, b), y, np.abs(n[ta] - n[tb]), tree_X(pre, ta, tb))
        sst = ((yt - y.mean()) ** 2).sum()
        fold.append([((yt - pl) ** 2).sum(), ((yt - pt) ** 2).sum(), ((yt - pb) ** 2).sum(), sst])
    fold = np.array(fold)

    def r2s(F):
        return 1 - F[:, :3].sum(0) / F[:, 3].sum()
    r = r2s(fold); out.update(R2g_line=float(r[0]), R2g_tree=float(r[1]), R2g_both=float(r[2]), dR2g=float(r[1] - r[0]))
    bs = []
    for _ in range(B):
        rr = r2s(fold[rng.integers(0, len(fold), len(fold))]); bs.append(rr[1] - rr[0])
    out["dR2g_lo"], out["dR2g_hi"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    out["pass"] = bool(out["dR2g"] >= 0.10 and out["dR2g_lo"] > 0)
    if light:
        return out
    # (i) random numbers, 5-fold
    F5 = []
    for tr_num, te_num in KFold(5, shuffle=True, random_state=seed_of(tag, "h2c5") % (2 ** 31)).split(np.arange(N)):
        Z = prep_fn(X, tr_num)
        a, b = sample_pairs(tr_num, None, cap_tr, rng, same=True); ta, tb = sample_pairs(te_num, None, cap_te, rng, same=True)
        y = ((Z[a] - Z[b]) ** 2).sum(1); yt = ((Z[ta] - Z[tb]) ** 2).sum(1)
        pl, pt, pb = fit_models(np.abs(n[a] - n[b]), tree_X(pre, a, b), y, np.abs(n[ta] - n[tb]), tree_X(pre, ta, tb))
        F5.append([((yt - pl) ** 2).sum(), ((yt - pt) ** 2).sum(), ((yt - pb) ** 2).sum(), ((yt - y.mean()) ** 2).sum()])
    r = r2s(np.array(F5)); out.update(R2r_line=float(r[0]), R2r_tree=float(r[1]), R2r_both=float(r[2]))
    return out


def within_group_line(Z, n, L, tag):
    lead = n // 10 ** (L - 1); rs = []
    for g in np.unique(lead):
        idx = np.where(lead == g)[0]
        if len(idx) < 6:
            continue
        rng = np.random.default_rng(seed_of(tag, "wg", g)); a, b = sample_pairs(idx, None, 50_000, rng, same=True)
        x = np.abs(n[a] - n[b]); y = ((Z[a] - Z[b]) ** 2).sum(1); e = 0.0; t = 0.0
        for tr, te in KFold(5, shuffle=True, random_state=int(g)).split(x):
            p = FastIso().fit(x[tr], y[tr]).predict(x[te])
            e += ((y[te] - p) ** 2).sum(); t += ((y[te] - y[tr].mean()) ** 2).sum()
        rs.append(1 - e / t)
    return float(np.mean(rs)) if rs else np.nan


def tree_indices(Z, groups, tag, n_samp=20_000):
    ug = np.unique(groups); G = len(ug)
    if G < 4:
        return dict(UVI=np.nan, delta_mean=np.nan, delta_p95=np.nan, G=G)
    cent = np.stack([Z[groups == g].mean(0) for g in ug]); cnt = np.array([(groups == g).sum() for g in ug])
    within = sum(((Z[groups == g] - cent[k]) ** 2).sum() for k, g in enumerate(ug)); s2 = within / max(len(Z) - G, 1)
    sq = (cent ** 2).sum(1); D2 = sq[:, None] + sq[None, :] - 2 * cent @ cent.T - s2 * (1 / cnt[:, None] + 1 / cnt[None, :])
    np.fill_diagonal(D2, 0); Dm = np.sqrt(np.clip(D2, 0, None)); rng = np.random.default_rng(seed_of(tag, "tree_idx"))
    from itertools import combinations
    tri = np.array(list(combinations(range(G), 3))) if G <= 30 else np.array([rng.choice(G, 3, replace=False) for _ in range(n_samp)])
    d = np.sort(np.stack([Dm[tri[:, 0], tri[:, 1]], Dm[tri[:, 0], tri[:, 2]], Dm[tri[:, 1], tri[:, 2]]], 1), 1)
    uvi = float(np.mean((d[:, 2] - d[:, 1]) / np.maximum(d[:, 2], 1e-12)))
    qu = np.array(list(combinations(range(G), 4))) if G <= 20 else np.array([rng.choice(G, 4, replace=False) for _ in range(n_samp)])
    x, y, z, w = qu.T
    S = np.sort(np.stack([Dm[x, y] + Dm[z, w], Dm[x, z] + Dm[y, w], Dm[x, w] + Dm[y, z]], 1), 1)
    mx = np.max(np.stack([Dm[x, y], Dm[x, z], Dm[x, w], Dm[y, z], Dm[y, w], Dm[z, w]], 1), 1)
    rel = (S[:, 2] - S[:, 1]) / 2 / np.maximum(mx, 1e-12)
    return dict(UVI=uvi, delta_mean=float(rel.mean()), delta_p95=float(np.percentile(rel, 95)), G=G)


def level_variances(Z, n, L):
    """Debiased between-group variance per prefix level (1 = leading digit, 2 = two-digit prefix)."""
    out = []; N = len(Z); prev = np.zeros(N, int)
    for l in range(1, min(L, 3)):
        g = n // 10 ** (L - l); ug = np.unique(g)
        if len(ug) >= N:
            break
        cent = {u: Z[g == u].mean(0) for u in ug}; Cg = np.stack([cent[u] for u in g])
        parent = np.stack([Z[prev == u].mean(0) for u in prev]) if l > 1 else np.broadcast_to(Z.mean(0), Z.shape)
        within = ((Z - Cg) ** 2).sum() / max(N - len(ug), 1)
        nparent = len(np.unique(prev)) if l > 1 else 1
        v = (((Cg - parent) ** 2).sum() - (len(ug) - nparent) * within) / N
        out.append((l, max(float(v), 0.0))); prev = g
    return out


# --------------------------------------------------------------------------- secondary: knee, exponent
def knee(X, n, L, tag):
    rng = np.random.default_rng(seed_of(tag, "knee")); te = rng.random(len(n)) < 0.2; tr = ~te
    Z = prep(X, np.where(tr)[0]); dig = digits(n, L)
    S = ns_basis(n, 10); Xd, _ = design(dig); M = np.column_stack([Xd, S])
    b = np.linalg.lstsq(M[tr], Z[tr], rcond=None)[0]; Ftr = M[tr] @ b; Pte = M[te] @ b; mu = Z[tr].mean(0)
    V = np.linalg.svd(Ftr - Ftr.mean(0), full_matrices=False)[2]; sst = ((Z[te] - mu) ** 2).sum(); r2 = {}
    for k in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50):
        k2 = min(k, V.shape[0]); Pk = Ftr.mean(0) + (Pte - Ftr.mean(0)) @ V[:k2].T @ V[:k2]
        r2[k] = float(1 - ((Z[te] - Pk) ** 2).sum() / sst)
    full = r2[50]; kn = next((k for k in r2 if r2[k] >= 0.95 * full), None) if full > 0 else None
    return dict(R2_by_k=r2, knee=kn)


def exponent(Xall, model, tag):
    tc = token_counts(model); ns = []
    for L, (lo, hi) in enumerate([(1, 9), (10, 99), (100, 999), (1000, 9999)], 1):
        nn = np.arange(lo, hi + 1)
        if MODELS[model][0] == "chunk":
            nn = nn[tc[nn - 1] == 1]
        if L == 4 and len(nn) > 900:
            nn = np.sort(np.random.default_rng(seed_of(model, "exp4")).choice(nn, 900, replace=False))
        if len(nn) >= 5:
            ns.append((L, nn))
    n = np.concatenate([v for _, v in ns]); y = np.concatenate([[L] * len(v) for L, v in ns]); X = Xall[n - 1]
    out = dict(lengths=[int(L) for L, _ in ns], counts=[int(len(v)) for _, v in ns])

    def cv_acc(lab):
        pred = np.zeros_like(lab)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
            Z = prep(X, tr); clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z[tr], lab[tr])
            pred[te] = clf.predict(Z[te])
        return float(balanced_accuracy_score(lab, pred))
    out["bal_acc"] = cv_acc(y)
    out["bal_acc_shuffled"] = cv_acc(np.random.default_rng(seed_of(tag, "exp_shuf")).permutation(y))
    Ls = sorted(set(y)); ordered = []; preds = {}
    for Lh in Ls:
        tr = np.where(y != Lh)[0]; te = np.where(y == Lh)[0]; Z = prep(X, tr)
        rg = Ridge(alpha=1.0).fit(Z[tr], y[tr]); ph = float(rg.predict(Z[te]).mean())
        mtr = {L: float(rg.predict(Z[y == L]).mean()) for L in Ls if L != Lh}; preds[int(Lh)] = ph
        lo = [mtr[L] for L in mtr if L < Lh]; hi = [mtr[L] for L in mtr if L > Lh]
        ok = (not lo or ph > mtr[max(L for L in mtr if L < Lh)]) and (not hi or ph < mtr[min(L for L in mtr if L > Lh)])
        ordered.append(bool(ok))
    out["heldout_pred"] = preds; out["all_ordered"] = bool(all(ordered)); out["ordered"] = ordered
    out["explicit"] = bool(out["bal_acc"] >= 0.90 and out["all_ordered"])
    return out


# --------------------------------------------------------------------------- H2 bundle with controls
def h2_all(X, Z, n, L, tag, big):
    i, j, c = steps_of(n); res = dict(n_steps={int(k): int((c == k).sum()) for k in np.unique(c)} if len(c) else {})
    m = additive(Z, digits(n, L)); F = m["F"]
    res["h2a"] = h2a(Z, n, L, tag, i, j, c, add=m)
    res["h2b"] = h2b(X, n, tag, i, j, c)
    res["h2c"] = h2c(X, n, L, tag, big=big)
    res["h2c"]["within_group_line_R2"] = within_group_line(Z, n, L, tag)
    res["h2c"]["tree_lead"] = tree_indices(Z, n // 10 ** (L - 1), tag + "|lead")
    if L >= 3:
        res["h2c"]["tree_pref2"] = tree_indices(Z, n // 10 ** (L - 2), tag + "|pref2")
    zprep = lambda A, idx: prep(A, idx, D=A.shape[1])            # fakes already live in PCA space
    T, h = trend(Z, n); res["trend_bw"] = h; E = Z - T
    run_ab = res["h2a"] is not None
    # controls: additive fake, matched smooth
    ctrl = {}
    rng = np.random.default_rng(seed_of(tag, "h2_add"))
    Zadd = Z.mean(0) + (F - F.mean(0)) + m["R"][rng.permutation(len(Z))]
    Zms = matched_smooth(Z, n, np.random.default_rng(seed_of(tag, "h2_ms")))
    for name, Zf in (("additive", Zadd), ("matched_smooth", Zms)):
        d = {}
        if run_ab:
            a = h2a(Zf, n, L, tag + name, i, j, c, n_perm=0); d["h2a_pass"] = a["pass"]; d["h2a_e"] = a["e"]
            b = h2b(Zf, n, tag + name, i, j, c, prep_fn=zprep); d["h2b_pass"] = b["pass"]; d["h2b_drop"] = b["drop"]
            d["h2b_boot"] = b["_boot"]
        cc = h2c(Zf, n, L, tag + name, prep_fn=zprep, big=big, light=True); d["h2c_pass"] = cc["pass"]; d["h2c_dR2g"] = cc["dR2g"]
        ctrl[name] = d
    if run_ab:
        bd = res["h2b"]["_boot"] - ctrl["additive"]["h2b_boot"]
        res["h2b"]["beyond_digits"] = float(res["h2b"]["drop"] - ctrl["additive"]["h2b_drop"])
        res["h2b"]["beyond_digits_lo"] = float(np.percentile(bd, 2.5))
        res["h2b"]["beyond_digits_pass"] = bool(res["h2b"]["beyond_digits"] > 0.10 and res["h2b"]["beyond_digits_lo"] > 0)
    for d in ctrl.values():
        d.pop("h2b_boot", None)
    res["controls"] = ctrl
    # planted power for H2a / H2b (with and without cumulative carry jumps)
    if run_ab:
        m0 = float(np.median(np.sqrt(((Z[j[c == 0]] - Z[i[c == 0]]) ** 2).sum(1)))); res["median_step0"] = m0
        levels_t = h2_levels(c)
        order = np.argsort(n); lvl_all = trailing_nines(n); nset = set(int(v) for v in n)
        has_next = np.array([int(v) + 1 in nset for v in n])
        pw = {"a_jump": 0, "a_none": 0, "b_jump": 0, "b_none": 0, "a_transient": 0, "b_transient": 0}
        for rep in range(20):
            r = np.random.default_rng(seed_of(tag, "h2plant", rep)); base = T + E[r.permutation(len(Z))]
            J = np.zeros_like(Z); cum = np.zeros(Z.shape[1])
            for k in order:
                J[k] = cum
                lv = lvl_all[k]
                if lv >= 1 and has_next[k]:
                    u = r.standard_normal(Z.shape[1]); cum = cum + 0.5 * lv * m0 * u / np.linalg.norm(u)
            Jt = np.zeros_like(Z)                       # Amendment 1: transient carry event on the number after a carry
            for lv_ in levels_t:
                for k in j[c == lv_]:
                    u = r.standard_normal(Z.shape[1]); Jt[k] = 0.5 * lv_ * m0 * u / np.linalg.norm(u)
            for name, Zf in (("jump", base + J), ("none", base), ("transient", base + Jt)):
                a = h2a(Zf, n, L, tag + "pl", i, j, c, B=100, n_perm=0, rng=r)
                b = h2b(Zf, n, tag + f"pl{rep}{name}", i, j, c, B=100, prep_fn=zprep, reps=1)
                pw["a_" + name] += a["pass"]; pw["b_" + name] += b["pass"]
        pw = {k: v / 20 for k, v in pw.items()}; res["power_ab"] = pw
        res["h2a"]["has_power"] = bool(pw["a_transient"] >= 0.8 and pw["a_none"] <= 0.2)   # Amendment 1
        res["h2a"]["has_power_cumulative"] = bool(pw["a_jump"] >= 0.8 and pw["a_none"] <= 0.2)
        res["h2b"]["has_power"] = bool(pw["b_jump"] >= 0.8 and pw["b_none"] <= 0.2)
    # planted power for H2c (tree vs line)
    lv = level_variances(Z, n, L); hits = {"tree": 0, "line": 0}
    for rep in range(10):
        r = np.random.default_rng(seed_of(tag, "h2cplant", rep)); noise = E[r.permutation(len(Z))]
        Tr = np.zeros_like(Z)
        for l, v in lv:
            g = n // 10 ** (L - l); ug, inv = np.unique(g, return_inverse=True)
            Tr += (r.standard_normal((len(ug), Z.shape[1])) * np.sqrt(v / Z.shape[1]))[inv]
        for name, Zf in (("tree", Z.mean(0) + Tr + noise), ("line", T + noise)):
            hits[name] += h2c(Zf, n, L, tag + f"cp{rep}{name}", prep_fn=zprep, big=big, B=200, light=True)["pass"]
    res["power_c"] = {k: v / 10 for k, v in hits.items()}
    res["h2c"]["has_power"] = bool(res["power_c"]["tree"] >= 0.8 and res["power_c"]["line"] <= 0.2)
    if res["h2b"] is not None:
        res["h2b"].pop("_boot", None)
    return res


# --------------------------------------------------------------------------- robustness (R8), point estimates
def robustness(X, n, L, tag, big):
    out = {}
    for v in ("nozscore", "drop3"):
        Z = prep(X, variant=v); dig = digits(n, L); d = {"O1": o1_all(Z, dig)}
        i, j, c = steps_of(n); a = h2a(Z, n, L, tag + v, i, j, c, B=50, n_perm=0) if len(c) else None
        d["h2a_e"] = a["e"] if a else None
        b = h2b(X, n, tag + v, i, j, c, B=50, prep_fn=lambda A, idx, v=v: prep(A, idx, variant=v), reps=1) if len(c) else None
        d["h2b_drop"] = b["drop"] if b else None
        d["h2c_dR2g"] = h2c(X, n, L, tag + v, prep_fn=lambda A, idx, v=v: prep(A, idx, variant=v), big=big, B=50, light=True)["dR2g"]
        out[v] = d
    return out


# --------------------------------------------------------------------------- driver
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_cell(X, model, s, d, out_dir):
    path = os.path.join(out_dir, "cells", f"{model}__{s}__{d}.json")
    if os.path.exists(path):
        return
    n = set_numbers(model, s); L = 4 if s.startswith("S4") else int(s[1]); tag = f"{model}|{s}|{d}"
    Xs = X[n - 1].astype(np.float64); big = len(n) > 3000
    Z = prep(Xs)
    rec = dict(model=model, family=MODELS[model][0], subfamily=MODELS[model][1], set=s, depth=DEPTH[d], N=int(len(n)),
               token_count=sorted({int(t) for t in token_counts(model)[n - 1]}), discovery=model in DISCOVERY,
               secondary=s == "S4x")
    rec["h1"] = h1(Z, n, L, tag)
    rec["h2"] = h2_all(Xs, Z, n, L, tag, big)
    rec["knee"] = knee(Xs, n, L, tag)
    rec["robust"] = robustness(Xs, n, L, tag, big)
    json.dump(_clean(rec), open(path + ".tmp", "w")); os.replace(path + ".tmp", path)


def run_job(args):
    data, model, d, out_dir = args
    X = np.load(layer_files(data, model)[d])["X"]
    ep = os.path.join(out_dir, "cells", f"{model}__exponent__{d}.json")
    if not os.path.exists(ep):
        r = exponent(X.astype(np.float64), model, f"{model}|exp|{d}"); r.update(model=model, depth=DEPTH[d])
        json.dump(_clean(r), open(ep, "w"))
    for s in sets_for(model):
        run_cell(X, model, s, d, out_dir)
    print(f"{model} {DEPTH[d]} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e13_stage1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    models = a.models.split(",") if a.models else list(MODELS)
    jobs = [(a.data, m, d, a.out) for m in models for d in range(5)]
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(run_job, jobs):
            pass
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
