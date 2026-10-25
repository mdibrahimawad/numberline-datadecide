"""Geometry audit: is the number line well described by PC1? (agent 08)"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar

REPO = "/home/user/numberline-datadecide"
D = f"{REPO}/results/beta_fine/step69369-seed-default"
OUT = sys.argv[1]
SEEDS = (45, 46, 47)
COARSE = [0, 3, 6, 9]


def fit_direct(gaps):  # identical to src/spacing_fit.fit_spacing_direct (beta only)
    gaps = np.asarray(gaps, float)
    i = np.arange(len(gaps), dtype=float)
    def loss(lb):
        b = np.exp(lb * i); s = gaps @ b / (b @ b); return float(((gaps - s * b) ** 2).sum())
    return float(np.exp(minimize_scalar(loss, bounds=(-8, 8), method="bounded").x))


def ndig(t):
    return np.array([len(str(int(x))) for x in t])


def one(S, t, g):
    S = S.astype(np.float64); u = np.log10(t); pc1 = S[:, 0]
    G = range(10)
    n = np.array([np.sum(g == k) for k in G])
    M = np.array([S[g == k].mean(0) for k in G])
    med1 = np.array([np.median(pc1[g == k]) for k in G])
    trc = np.array([np.trace(np.cov(S[g == k].T)) for k in G])          # within-group total var (5 PCs)
    W = sum(np.cov(S[g == k].T) * (n[k] - 1) for k in G) / (len(t) - 10)  # pooled within cov
    r = {}
    gc = np.diff(M[COARSE, 0])
    r["neg_coarse_gaps"] = int((gc < 0).sum())
    r["neg_fine_gaps"] = int((np.diff(M[:, 0]) < 0).sum())
    r["beta_pc1"] = fit_direct(np.abs(gc))
    r["beta_med"] = fit_direct(np.abs(np.diff(med1[COARSE])))
    def arc(dims):
        d = []
        for a, b in zip(COARSE[:-1], COARSE[1:]):
            sq = ((M[b, dims] - M[a, dims]) ** 2).sum()
            Wa = np.cov(S[g == a][:, dims].T); Wb = np.cov(S[g == b][:, dims].T)
            bias = np.trace(np.atleast_2d(Wa)) / n[a] + np.trace(np.atleast_2d(Wb)) / n[b]
            d.append(np.sqrt(max(sq - bias, 0)))
        return fit_direct(d)
    r["beta_arc5"] = arc([0, 1, 2, 3, 4]); r["beta_arc2"] = arc([0, 1])
    # between-group geometry (bias-corrected group-mean covariance)
    B = np.cov(M.T) - np.mean([np.cov(S[g == k].T) / n[k] for k in G], axis=0)
    lam, V = np.linalg.eigh(B); lam = np.clip(lam[::-1], 0, None); V = V[:, ::-1]
    r["offline_frac"] = float(1 - lam[0] / lam.sum())
    r["pc1_share_between"] = float(max(B[0, 0], 0) / np.trace(B))
    r["cos_line_pc1"] = float(abs(V[0, 0]))
    # PC2 arc: PC2 group means vs PC1 group means, linear vs quadratic
    x, y = M[:, 0], M[:, 1]
    sst = ((y - y.mean()) ** 2).sum()
    r2q = 1 - ((y - np.polyval(np.polyfit(x, y, 2), x)) ** 2).sum() / sst
    r2l = 1 - ((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2).sum() / sst
    r["pc2_amp"] = float(np.ptp(y) / np.ptp(x))          # PC2 excursion of group means / PC1 range
    r["pc2_arc_r2q_minus_lin"] = float(r2q - r2l)
    r["rho_pc2"] = abs(spearmanr(t, S[:, 1])[0])
    r["rho_max_pc2_5"] = max(abs(spearmanr(t, S[:, j])[0]) for j in range(1, 5))
    X1 = np.column_stack([np.ones_like(u), pc1]); X5 = np.column_stack([np.ones_like(u), S])
    def R2(X):
        c, *_ = np.linalg.lstsq(X, u, rcond=None); e = u - X @ c; return 1 - e @ e / ((u - u.mean()) ** 2).sum()
    r["R2u_pc1"] = R2(X1); r["R2u_pc15"] = R2(X5); r["share_u_pc1"] = r["R2u_pc1"] / r["R2u_pc15"]
    # eta^2 of PC1 on groups
    r["eta2"] = float(n @ (M[:, 0] - pc1.mean()) ** 2 / ((pc1 - pc1.mean()) ** 2).sum())
    # digit count: within-class slope vs overall slope
    dg = ndig(t)
    overall = np.polyfit(u, pc1, 1)[0]
    cls = [c for c in (2, 3, 4) if np.sum(dg == c) > 20]
    xs = np.concatenate([u[dg == c] - u[dg == c].mean() for c in cls])
    ys = np.concatenate([pc1[dg == c] - pc1[dg == c].mean() for c in cls])
    r["slope_ratio_within_digit"] = float((xs @ ys / (xs @ xs)) / overall)
    # step at digit boundaries inside straddling groups 0,3,6,9
    gap_mean = np.mean(np.abs(gc)); steps = []
    ub = np.array([u[g == k].mean() for k in G])
    for k in COARSE:
        m = g == k; lo = dg[m] == dg[m].min(); hi = ~lo
        if lo.sum() < 5 or hi.sum() < 5:
            continue
        dobs = pc1[m][hi].mean() - pc1[m][lo].mean()
        a, b = max(k - 1, 0), min(k + 1, 9)
        slope = (M[b, 0] - M[a, 0]) / (ub[b] - ub[a])
        dexp = slope * (u[m][hi].mean() - u[m][lo].mean())
        steps.append((dobs - dexp) / gap_mean)
    r["digit_step_excess"] = float(np.median(steps)) if steps else np.nan
    # outliers
    z = np.zeros_like(pc1)
    for k in G:
        m = g == k; md = np.median(pc1[m]); mad = 1.4826 * np.median(np.abs(pc1[m] - md)); z[m] = (pc1[m] - md) / mad
    r["out_frac_pc1"] = float(np.mean(np.abs(z) > 5))
    mah = np.zeros(len(t))
    for k in G:
        m = g == k; C = np.cov(S[m].T); d = S[m] - S[m].mean(0); mah[m] = np.einsum("ij,jk,ik->i", d, np.linalg.pinv(C), d)
    r["out_frac_mah5"] = float(np.mean(mah > 35.9))   # chi2(5) p ~ 1.7e-6
    r["within_sd_over_meangap"] = float(np.sqrt(W[0, 0]) / gap_mean)
    return r


rec = pd.read_csv(f"{REPO}/results/datadecide/alpha_beta.csv")[["recipe", "selected_layer"]]
rows = []
for recipe, sl in rec.itertuples(index=False):
    z = np.load(f"{D}/{recipe}.npz")
    js = json.load(open(f"{D}/{recipe}.json"))
    for tag, L in (("sel", int(sl)), ("L8", 8)):
        per = []
        for s in SEEDS:
            r = one(z[f"seed{s}_pc_scores"][L], z[f"seed{s}_targets"], z[f"seed{s}_groups"])
            ev = z[f"seed{s}_pc_evr"][L]; r["ev1"] = float(ev[0]); r["ev2"] = float(ev[1]); r["ev1_over_ev2"] = float(ev[0] / ev[1])
            r["n_prompts"] = len(z[f"seed{s}_targets"])
            per.append(r)
        df = pd.DataFrame(per)
        row = {"recipe": recipe, "which": tag, "layer": L}
        row.update(df.median().to_dict())
        row["beta_pc1_sd"] = float(df["beta_pc1"].std(ddof=1))
        row["neg_coarse_seeds"] = int((df["neg_coarse_gaps"] > 0).sum())
        if tag == "sel" and "old_layer" in js["report"]:
            row["e06_beta_coarse_oldlayer"] = js["report"]["old_layer"]["beta_coarse"]["mean"]
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("done", len(res))
