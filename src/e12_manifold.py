"""E12: do number hidden states lie on a 1-D manifold? Implements experiments/E12_one_dimensional_manifold/PREREG.md.

Datasets: Qwen2.5-1.5B sets Q2/Q3/Q4 and Pythia-1.4B sets P2/P3/P1 (one-token numbers), five depths each.
For every dataset: preparation (standardise, PCA to 50), eight conditions (real + seven fakes), tests T1-T8.

  python -m src.e12_manifold --data <topology_number_line>/data --out results/e12_manifold
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(__file__)
PYTHIA_COUNTS = os.path.join(HERE, "..", "experiments", "E12_one_dimensional_manifold", "pythia_token_counts_1_10000.txt")
MODELS = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "pythia-1.4b": "Pythia-1.4B"}
DEPTH = ["~25%", "~50%", "~75%", "~84%", "last"]
KS = [5, 10, 20, 40, 80]
CONDITIONS = ["real", "curve", "log_curve", "helix", "sheet_thin", "sheet_fat", "shuffled", "matched_smooth"]
PERIODS = [2, 5, 10, 100]


# --------------------------------------------------------------------------- data
def set_numbers(model, name):
    if model.startswith("qwen"):
        if name == "Q2": return np.arange(10, 100)
        if name == "Q3": return np.arange(100, 1000)
        if name == "Q4": return np.sort(np.random.default_rng(0).choice(np.arange(1000, 10000), 3000, replace=False))
    c = np.loadtxt(PYTHIA_COUNTS, dtype=int)                     # token count of n = 1..10000 in prompt context
    one = np.arange(1, 10001)[c == 1]
    if name == "P2": return np.arange(10, 100)
    if name == "P3": return one[(one >= 100) & (one <= 999)]
    if name == "P1": return one[one <= 999]
    raise ValueError(name)


SETS = {"qwen2.5-1.5b": ["Q2", "Q3", "Q4"], "pythia-1.4b": ["P2", "P3", "P1"]}


def layer_file(data, model, d):
    folder = os.path.join(data, f"{model}_vectors")
    files = sorted(f for f in os.listdir(folder) if f.startswith("layer_"))
    return os.path.join(folder, files[d])


def huge_dims(X):
    v = X.var(0); s = np.sort(v)[::-1]
    return dict(top1_var_share=float(s[0] / s.sum()), top5_var_share=float(s[:5].sum() / s.sum()),
                max_over_median_var=float(s[0] / np.median(v)))


def standardise(X):
    X = X - X.mean(0); sd = X.std(0); keep = sd > 1e-8
    return X[:, keep] / sd[keep]


def pca(X, D=50):
    D = min(D, len(X) - 1)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return U[:, :D] * S[:D]


# --------------------------------------------------------------------------- smoothing over n
def kernel_smooth(Z, n, h, n_eval=None):
    n_eval = n if n_eval is None else n_eval
    W = np.exp(-0.5 * ((n_eval[:, None] - n[None, :]) / h) ** 2)
    return (W @ Z) / W.sum(1, keepdims=True)


def cv_bandwidth(Z, n, grid=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    R = n.max() - n.min(); best = None
    for h in [g for g in grid if g <= R / 4]:
        W = np.exp(-0.5 * ((n[:, None] - n[None, :]) / h) ** 2); w0 = np.diag(W).copy()
        pred = (W @ Z - w0[:, None] * Z) / (W.sum(1) - w0)[:, None]      # leave-one-out
        err = float(((Z - pred) ** 2).sum(1).mean())
        if best is None or err < best[1]:
            best = (h, err)
    return best[0]


# --------------------------------------------------------------------------- fakes
def _gauss_like(E, N, rng):
    C = np.cov(E.T); w, V = np.linalg.eigh(C); w = np.clip(w, 0, None)
    return (rng.standard_normal((N, len(w))) * np.sqrt(w)) @ V.T


def _orthonormal(D, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q[:, :k]


def _arclength_curve(T_sorted, u):
    seg = np.linalg.norm(np.diff(T_sorted, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    target = u * s[-1]
    return np.stack([np.interp(target, s, T_sorted[:, j]) for j in range(T_sorted.shape[1])], 1)


def matched_smooth(Z, n, rng):
    """Stationary Gaussian process per PCA dimension with the dataset's own variogram over |m - n| (PREREG (d))."""
    lo, hi = n.min(), n.max(); G = hi - lo + 1; idx = n - lo
    out = np.zeros_like(Z)
    i, j = np.triu_indices(len(n), 1); dl = np.abs(n[i] - n[j])
    for c in range(Z.shape[1]):
        z = Z[:, c] - Z[:, c].mean(); var = z.var()
        g = np.bincount(dl, weights=0.5 * (z[i] - z[j]) ** 2, minlength=G) / np.maximum(np.bincount(dl, minlength=G), 1)
        have = np.bincount(dl, minlength=G) > 0
        g = np.interp(np.arange(G), np.arange(G)[have], g[have]); g[0] = 0.0
        g = uniform_filter1d(g, 5, mode="nearest"); g[0] = 0.0
        k = var - g
        row = np.r_[k, k[-2:0:-1]]                                   # circulant embedding, length 2G - 2
        lam = np.clip(np.fft.fft(row).real, 0, None); M = len(row)
        xi = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        path = np.fft.ifft(np.sqrt(lam) * xi).real[:G]               # rescaled to var_j below
        s = path.std()
        out[:, c] = path[idx] / (s if s > 0 else 1) * np.sqrt(var)
    return out


def make_fakes(Z, n, seed, which=None):
    """All fake conditions for one dataset. Returns {name: (N, D) array}."""
    rng = np.random.default_rng(seed)
    order = np.argsort(n); n_s = n[order]
    h = cv_bandwidth(Z[order], n_s)
    T = np.empty_like(Z); T[order] = kernel_smooth(Z[order], n_s, h)
    E = Z - T; N, D = Z.shape
    fakes, info = {}, dict(bandwidth=h, residual_var=float((E ** 2).sum(1).mean()), trend_var=float(((T - T.mean(0)) ** 2).sum(1).mean()))
    want = lambda k: which is None or k in which
    if want("curve"):
        fakes["curve"] = T + _gauss_like(E, N, rng)
    if want("log_curve"):
        u = (np.log(n) - np.log(n.min())) / (np.log(n.max()) - np.log(n.min()))
        fakes["log_curve"] = _arclength_curve(T[order], u) + _gauss_like(E, N, rng)
    if want("helix"):
        R = n.max() - n.min(); per = [p for p in PERIODS if p <= R / 3]
        Q = _orthonormal(D, 2 * len(per), rng); H = np.zeros_like(Z)
        for a, p in enumerate(per):
            H += np.outer(np.cos(2 * np.pi * n / p), Q[:, 2 * a])
            if p != 2:
                H += np.outer(np.sin(2 * np.pi * n / p), Q[:, 2 * a + 1])
        H -= H.mean(0); H *= np.sqrt(0.5 * info["residual_var"] / max((H ** 2).sum(1).mean(), 1e-12))
        fakes["helix"] = T + H + np.sqrt(0.5) * _gauss_like(E, N, rng)
    for name, wfrac in (("sheet_thin", 0.1), ("sheet_fat", 0.5)):
        if want(name):
            x = (n - n.min()) / (n.max() - n.min()); y = rng.uniform(0, wfrac, N)
            S2 = np.c_[x, y]; S2 -= S2.mean(0)
            S = S2 @ _orthonormal(D, 2, rng).T
            S *= np.sqrt(info["trend_var"] / ((S ** 2).sum(1).mean()))
            fakes[name] = S + _gauss_like(E, N, rng)
    if want("shuffled"):
        fakes["shuffled"] = np.stack([rng.permutation(Z[:, j]) for j in range(D)], 1)
    if want("matched_smooth"):
        fakes["matched_smooth"] = matched_smooth(Z, n, rng)
    return fakes, info


# --------------------------------------------------------------------------- T2 dimension
def id_mle(Z, ks=KS):
    kmax = min(max(ks), len(Z) - 1)
    dist, _ = NearestNeighbors(n_neighbors=kmax + 1).fit(Z).kneighbors(Z)
    T = np.maximum(dist[:, 1:], 1e-12)
    out = {}
    for k in ks:
        if k >= len(Z):
            out[k] = np.nan; continue
        inv = np.mean(np.log(T[:, k - 1:k] / T[:, :k - 1]), 1)          # = 1 / m_k(x)
        out[k] = float(1 / np.mean(inv))
    mu = T[:, 1] / T[:, 0]; mu = mu[mu > 1]
    out["twonn"] = float(len(mu) / np.log(mu).sum())
    return out


def plateau(ids, lo=0.8, hi=1.5):
    v = [ids[k] for k in KS]
    ok = [np.isfinite(x) and lo <= x <= hi for x in v]
    return any(all(ok[i:i + 3]) for i in range(len(ok) - 2))


# --------------------------------------------------------------------------- T3 principal curve
def _polyline_project(Z, P, chunk=400):
    seg = np.diff(P, axis=0); L2 = np.maximum((seg ** 2).sum(1), 1e-12); cum = np.r_[0, np.cumsum(np.sqrt(L2))]
    pos = np.empty(len(Z))
    for s in range(0, len(Z), chunk):
        z = Z[s:s + chunk]
        t = np.clip(np.einsum("nsd,sd->ns", z[:, None, :] - P[None, :-1, :], seg) / L2, 0, 1)
        proj = P[None, :-1, :] + t[..., None] * seg[None]
        d2 = ((z[:, None, :] - proj) ** 2).sum(2); b = d2.argmin(1)
        pos[s:s + chunk] = cum[b] + t[np.arange(len(z)), b] * np.sqrt(L2[b])
    return pos


def principal_curve(Z, iters=10, knots=200, sig_frac=0.02):
    v1 = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    lam = (Z - Z.mean(0)) @ v1
    for _ in range(iters):
        o = np.argsort(lam)
        C = gaussian_filter1d(Z[o], sigma=max(1.0, sig_frac * len(Z)), axis=0, mode="nearest")
        P = C[np.unique(np.linspace(0, len(Z) - 1, min(knots, len(Z))).astype(int))]
        lam = _polyline_project(Z, P)
    return P, lam


def _oriented(P, lam_tr, n_tr):
    if spearmanr(lam_tr, n_tr).correlation < 0:
        return P[::-1], _polyline_project_fn(P[::-1])
    return P, _polyline_project_fn(P)


def _polyline_project_fn(P):
    return lambda Z: _polyline_project(Z, P)


def t3_order(Z, n, seed):
    rng = np.random.default_rng(seed); R = n.max() - n.min(); out = {}
    rhos, fine = [], []
    for r in range(3):
        te = rng.random(len(n)) < 0.2
        P, lam = principal_curve(Z[~te]); P, proj = _oriented(P, lam, n[~te])
        pt = proj(Z[te]); rhos.append(abs(spearmanr(pt, n[te]).correlation))
        w = 0.05 * R; ws = []
        for a in np.arange(n.min(), n.max(), w):
            m = (n[te] >= a) & (n[te] < a + w)
            if m.sum() >= 5:
                ws.append(spearmanr(pt[m], n[te][m]).correlation)
        fine.append(np.nanmedian(ws) if ws else np.nan)
    out["T3_random_rho"] = float(np.mean(rhos)); out["T3_fine_rho"] = float(np.nanmean(fine)) if np.isfinite(fine).any() else np.nan
    te = (n >= n.min() + 0.45 * R) & (n < n.min() + 0.55 * R)
    P, lam = principal_curve(Z[~te]); P, proj = _oriented(P, lam, n[~te])
    out["T3_block_rho"] = float(spearmanr(proj(Z[te]), n[te]).correlation)
    te = n >= n.min() + 0.8 * R
    P, lam = principal_curve(Z[~te]); P, proj = _oriented(P, lam, n[~te])
    pt = proj(Z[te]); ptr = proj(Z[~te])
    out["T3_extrap_rho"] = float(spearmanr(pt, n[te]).correlation)
    out["T3_extrap_beyond_share"] = float(np.mean(pt >= np.quantile(ptr, 0.95)))
    return out


# --------------------------------------------------------------------------- T4 spacing
def t4_spacing(Z, n, seed):
    P, lam = principal_curve(Z)
    s = lam if spearmanr(lam, n).correlation >= 0 else lam.max() - lam
    s = (s - s.mean()) / s.std()
    grid = np.round(np.arange(0.05, 2.0001, 0.05), 2)

    def fit(x, y):
        A = np.c_[np.ones(len(x)), x]; c = np.linalg.lstsq(A, y, rcond=None)[0]; return c

    def best_p(nn, ss):
        errs = [((ss - np.c_[np.ones(len(nn)), nn ** p] @ fit(nn ** p, ss)) ** 2).mean() for p in grid]
        return grid[int(np.argmin(errs))]

    kf = KFold(5, shuffle=True, random_state=seed); mse = {"linear": [], "log": [], "power": []}
    for tr, te in kf.split(n):
        for name, f in (("linear", lambda v: v), ("log", np.log)):
            c = fit(f(n[tr]), s[tr]); mse[name].append(((s[te] - np.c_[np.ones(len(te)), f(n[te])] @ c) ** 2).mean())
        p = best_p(n[tr], s[tr]); c = fit(n[tr] ** p, s[tr])
        mse["power"].append(((s[te] - np.c_[np.ones(len(te)), n[te] ** p] @ c) ** 2).mean())
    mse = {k: float(np.mean(v)) for k, v in mse.items()}
    p_all = best_p(n, s); rng = np.random.default_rng(seed); boot = []
    for _ in range(200):
        b = rng.integers(0, len(n), len(n)); boot.append(best_p(n[b], s[b]))
    return dict(T4_best=min(mse, key=mse.get), T4_mse_linear=mse["linear"], T4_mse_log=mse["log"], T4_mse_power=mse["power"],
                T4_p=float(p_all), T4_p_lo=float(np.quantile(boot, 0.025)), T4_p_hi=float(np.quantile(boot, 0.975)))


# --------------------------------------------------------------------------- T5 / T6 pairs
def _pairs(N, rng, max_pairs=200_000):
    total = N * (N - 1) // 2
    if total <= max_pairs:
        return np.triu_indices(N, 1)
    i = rng.integers(0, N, int(max_pairs * 1.1)); j = rng.integers(0, N, int(max_pairs * 1.1)); k = i != j
    i, j = i[k][:max_pairs], j[k][:max_pairs]
    return np.minimum(i, j), np.maximum(i, j)


def _cv_r2(fitpred, y, seed):
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(y):
        pred[te] = fitpred(tr, te)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _iso(x, y):
    return lambda tr, te: IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[tr], y[tr]).predict(x[te])


def _digit_features(a, b, ndig):
    from scipy.sparse import csr_matrix
    cols = []
    for p in range(ndig):
        da, db = (a // 10 ** p) % 10, (b // 10 ** p) % 10
        lo, hi = np.minimum(da, db), np.maximum(da, db)
        cols.append(p * 100 + lo * 10 + hi)
    rows = np.repeat(np.arange(len(a)), ndig); c = np.stack(cols, 1).ravel()
    return csr_matrix((np.ones(len(c)), (rows, c)), shape=(len(a), ndig * 100))


def t5_t6(Z, n, seed):
    rng = np.random.default_rng(seed); i, j = _pairs(len(n), rng)
    d2 = np.concatenate([((Z[i[c:c + 20000]] - Z[j[c:c + 20000]]) ** 2).sum(1) for c in range(0, len(i), 20000)])
    a, b = n[i], n[j]
    dlin = np.abs(a - b).astype(float); dlog = np.abs(np.log(a) - np.log(b))
    out = dict(T5_r2_linear=_cv_r2(_iso(dlin, d2), d2, seed), T5_r2_log=_cv_r2(_iso(dlog, d2), d2, seed))
    ndig = len(str(n.max()))
    F = _digit_features(a, b, ndig)
    out["T6_r2_digits"] = _cv_r2(lambda tr, te: Ridge(alpha=1.0).fit(F[tr], d2[tr]).predict(F[te]), d2, seed)
    out["T6_r2_value"] = out["T5_r2_linear"]
    from scipy.sparse import hstack, csr_matrix
    q = np.quantile(dlin, np.linspace(0, 1, 21)[1:-1]); bins = np.searchsorted(q, dlin)
    B = csr_matrix((np.ones(len(bins)), (np.arange(len(bins)), bins)), shape=(len(bins), 20))
    FB = hstack([F, B]).tocsr()
    out["T6_r2_both"] = _cv_r2(lambda tr, te: Ridge(alpha=1.0).fit(FB[tr], d2[tr]).predict(FB[te]), d2, seed)
    # nearest neighbours
    _, nb = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z); nb = nb[:, 1:]
    q_, v_ = np.repeat(n, 10), n[nb].ravel(); R = n.max() - n.min()
    lead = lambda x: np.array([int(str(t)[0]) * 10 + len(str(t)) for t in x])
    out.update(T6_nn_same_last=float(np.mean(q_ % 10 == v_ % 10)), T6_nn_same_leading=float(np.mean(lead(q_) == lead(v_))),
               T6_nn_close=float(np.mean(np.abs(q_ - v_) <= 0.01 * R)))
    r1 = rng.choice(n, 50000); r2 = rng.choice(n, 50000); k = r1 != r2; r1, r2 = r1[k], r2[k]
    out.update(T6_chance_same_last=float(np.mean(r1 % 10 == r2 % 10)), T6_chance_same_leading=float(np.mean(lead(r1) == lead(r2))),
               T6_chance_close=float(np.mean(np.abs(r1 - r2) <= 0.01 * R)))
    return out


# --------------------------------------------------------------------------- T7 helix
def t7_helix(Z, n, seed, n_perm=200):
    R = n.max() - n.min(); order = np.argsort(n)
    T = np.empty_like(Z); T[order] = kernel_smooth(Z[order], n[order], 0.2 * R)
    E = Z - T; E = E - E.mean(0); tot = (E ** 2).sum(); rng = np.random.default_rng(seed); out = {}
    for p in PERIODS:
        if p > R / 3:
            out[f"T7_share_p{p}"] = np.nan; out[f"T7_pval_p{p}"] = np.nan; out[f"T7_z_p{p}"] = np.nan; continue
        A = np.c_[np.cos(2 * np.pi * n / p)] if p == 2 else np.c_[np.cos(2 * np.pi * n / p), np.sin(2 * np.pi * n / p)]
        A = A - A.mean(0); Pm = A @ np.linalg.pinv(A)
        share = lambda M: float((Pm @ M) .__pow__(2).sum() / tot)
        obs = share(E); null = np.array([share(E[rng.permutation(len(E))]) for _ in range(n_perm)])
        out[f"T7_share_p{p}"] = obs; out[f"T7_pval_p{p}"] = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        out[f"T7_z_p{p}"] = float((obs - null.mean()) / null.std())
    return out


# --------------------------------------------------------------------------- T8 topology
def t8_topology(Z, landmarks=400, rule=0.25):
    if len(Z) > landmarks:
        r = ripser(Z, maxdim=1, n_perm=landmarks); P = Z[r["idx_perm"]]
    else:
        r = ripser(Z, maxdim=1); P = Z
    med = float(np.median(pdist(P))); h1 = r["dgms"][1]
    life = (h1[:, 1] - h1[:, 0]) / med if len(h1) else np.zeros(1)
    return dict(T8_long_H1=int(np.sum(life > rule)), T8_max_H1_ratio=float(life.max()) if len(life) else 0.0)


# --------------------------------------------------------------------------- one dataset
def all_tests(Z, n, seed, full=True):
    ids = id_mle(Z)
    row = {f"T2_mle_k{k}": ids[k] for k in KS}
    row.update(T2_twonn=ids["twonn"], T2_plateau=plateau(ids),
               T2_ratio_80_5=ids[80] / ids[5] if np.isfinite(ids.get(80, np.nan)) else np.nan)
    if full:
        row.update(t3_order(Z, n, seed)); row.update(t4_spacing(Z, n, seed)); row.update(t5_t6(Z, n, seed))
        row.update(t7_helix(Z, n, seed)); row.update(t8_topology(Z))
    return row


def run_dataset(args):
    data, model, sname, d, dsid = args
    n = set_numbers(model, sname)
    X = np.load(layer_file(data, model, d))["X"].astype(np.float64)[n - 1]
    meta = dict(model=model, set=sname, depth=DEPTH[d], N=len(n), **huge_dims(X))
    Xs = standardise(X); Z = pca(Xs)
    fakes, info = make_fakes(Z, n, seed=1000 * dsid)
    meta.update(info)
    rows = []
    for c in CONDITIONS:
        M = Z if c == "real" else fakes[c]
        rows.append(dict(meta, condition=c, **all_tests(M, n, seed=1000 * dsid + CONDITIONS.index(c))))
    # robustness: real data without PCA (T2, T5, T6, T8)
    ids = id_mle(Xs); rob = {f"T2_mle_k{k}": ids[k] for k in KS}
    rob.update(T2_twonn=ids["twonn"], T2_plateau=plateau(ids))
    rob.update(t5_t6(Xs, n, 1000 * dsid + 99)); rob.update(t8_topology(Xs))
    rows.append(dict(meta, condition="real_noPCA", **rob))
    # T1 power: 5 replicates of curve, thin and fat sheet (dimension rule only)
    power = []
    for rep in range(5):
        fk, _ = make_fakes(Z, n, seed=1000 * dsid + 500 + rep, which={"curve", "sheet_thin", "sheet_fat"})
        for c, M in fk.items():
            ids = id_mle(M); power.append(dict(meta, condition=c, rep=rep, plateau=plateau(ids), mle_k20=ids[20]))
    print(f"{model} {sname} {DEPTH[d]} done", flush=True)
    return rows, power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e12_manifold")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for model in MODELS:
        for sname in SETS[model]:
            for d in range(5):
                jobs.append((a.data, model, sname, d, len(jobs) + 1))
    with Pool(a.workers) as pool:
        res = pool.map(run_dataset, jobs)
    rows = [r for rr, _ in res for r in rr]; power = [p for _, pp in res for p in pp]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame(power).to_csv(os.path.join(a.out, "power_replicates.csv"), index=False)
    print("saved", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
