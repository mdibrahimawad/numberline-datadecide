"""E11 tier 2 analysis: number topology across pretraining checkpoints.

Implements exactly the measures pre-registered in experiments/E11_topology_hypotheses/CHECKPOINTS.md
(written before the checkpoint data arrived):

  C1  developmental order digit -> value: kNN composition of the 2,000 four-digit numbers per
      checkpoint (share_same_last, share_within_50, share_same_leading, ari_leading);
      C1a Spearman signs, C1b step fit vs line (AIC), C1c half-way step vs copy half-way step.
  C2  do rare numbers catch up: r_half ratio rarest / most frequent tenth, K = 20 subsamples of
      120 + 120, raw and scale-free, 100 within-checkpoint label permutations.
  Secondary: H1 Wasserstein velocity between checkpoints + early concentration vs a
      checkpoint-order null; 41-feature probe of training step (leave-one-checkpoint-out).

Input: <data>/<family>/<revision>/layer_LL.npz + meta.json written by src/e11_checkpoints.py.

  python -m src.e11_ckpt_analysis --data <topology_number_line>/data/e11_ckpt --out results/e11_ckpt
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import src.e10_explore as ex
from src.e10_graded import DEPTH, levels
from src.e11_tier0 import h0_deaths
from src.e11_tier1_laptop import number_subset
from src.topo_features import FEATURE_NAMES, barcode_features, diagrams

PRIMARY = 3                                   # ~84 % depth
C1_MEASURES = ["share_same_last", "share_within_50", "share_same_leading", "ari_leading"]


def step_of(rev):
    return int(rev.split("step")[1].split("-")[0])


def revisions(data, fam):
    revs = [os.path.basename(os.path.dirname(f)) for f in glob(os.path.join(data, fam, "*", "meta.json"))]
    return sorted(revs, key=step_of)


def load_rev(data, fam, rev):
    files = sorted(glob(os.path.join(data, fam, rev, "layer_*.npz")))
    Xs = [np.load(f)["X"].astype(np.float32) for f in files]
    target = np.load(files[0])["target"]
    return Xs, target, json.load(open(os.path.join(data, fam, rev, "meta.json")))


# --------------------------------------------------------------------------- #
# C1: kNN composition of the 2,000 four-digit numbers
# --------------------------------------------------------------------------- #

def knn_comp(X, nums, k=10, seed=0):
    four = nums >= 1000
    q, P = nums[four], X[four].astype(np.float64)
    sq = (P ** 2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * P @ P.T
    np.fill_diagonal(D2, np.inf)
    nb = q[np.argsort(D2, 1)[:, :k]]
    a, v = np.repeat(q, k), nb.ravel()
    lab = fcluster(linkage(P, "average"), 9, "maxclust")
    return dict(share_same_last=float(np.mean(a % 10 == v % 10)),
                share_within_50=float(np.mean(np.abs(a - v) <= 50)),
                share_same_leading=float(np.mean(a // 1000 == v // 1000)),
                ari_leading=float(adjusted_rand_score(q // 1000, lab)))


def knn_chance(nums, seed=0):
    rng = np.random.default_rng(seed)
    q = nums[nums >= 1000]
    a = rng.choice(q, 50000); v = rng.choice(q, 50000)
    keep = a != v; a, v = a[keep], v[keep]
    return dict(share_same_last=float(np.mean(a % 10 == v % 10)), share_within_50=float(np.mean(np.abs(a - v) <= 50)),
                share_same_leading=float(np.mean(a // 1000 == v // 1000)), ari_leading=0.0)


def fit_line_vs_step(x, y):
    """AIC of a straight line vs a two-segment step (piecewise constant, one breakpoint) in x."""
    n = len(x)
    A = np.c_[np.ones(n), x]
    rss_line = float(np.sum((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2))
    best = (np.inf, None)
    for j in range(1, n):                         # break between x[j-1] and x[j]
        rss = float(np.sum((y[:j] - y[:j].mean()) ** 2) + np.sum((y[j:] - y[j:].mean()) ** 2))
        if rss < best[0]:
            best = (rss, j)
    aic = lambda rss, k: n * np.log(max(rss, 1e-12) / n) + 2 * k
    return dict(aic_line=aic(rss_line, 2), aic_step=aic(best[0], 3), step_break_after=int(best[1] - 1))


def halfway(x, y, target):
    """First crossing of `target` (rising) with linear interpolation in x; nan if never crossed."""
    for j in range(len(y)):
        if y[j] >= target:
            if j == 0:
                return float(x[0])
            return float(x[j - 1] + (target - y[j - 1]) / (y[j] - y[j - 1]) * (x[j] - x[j - 1]))
    return np.nan


# --------------------------------------------------------------------------- #
# C2: rarest vs most frequent tenth, r_half ratio with label permutation
# --------------------------------------------------------------------------- #

def _r_half_stat(X, idx0, idx9, rng, K=20, m=120):
    raw, free = [], []
    for _ in range(K):
        A, B = X[rng.choice(idx0, m, replace=False)], X[rng.choice(idx9, m, replace=False)]
        ra, rb = np.median(h0_deaths(A)), np.median(h0_deaths(B))
        raw.append(np.log(ra / rb))
        free.append(np.log((ra / np.median(pdist(A))) / (rb / np.median(pdist(B)))))
    return np.mean(raw), np.mean(free)


def _c2_cell(args):
    fam, rev, d, X, lab, seed, P = args
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    idx0, idx9 = np.where(lab == 0)[0], np.where(lab == 9)[0]
    real = _r_half_stat(X, idx0, idx9, rng)
    both = np.concatenate([idx0, idx9])
    null = []
    for _ in range(P):
        perm = rng.permutation(both)
        null.append(_r_half_stat(X, perm[:len(idx0)], perm[len(idx0):], rng))
    null = np.array(null)
    out = dict(family=fam, revision=rev, step=step_of(rev), depth=DEPTH[d])
    for j, name in enumerate(("raw", "scalefree")):
        out[f"ratio_{name}"] = float(np.exp(real[j]))                            # < 1: rarest tenth more crowded
        out[f"p_{name}"] = float((np.sum(null[:, j] <= real[j]) + 1) / (P + 1))   # one-sided, crowding
        out[f"p2_{name}"] = float((np.sum(np.abs(null[:, j] - null[:, j].mean()) >= abs(real[j] - null[:, j].mean())) + 1) / (P + 1))
    return out


# --------------------------------------------------------------------------- #
# Secondary: H1 Wasserstein path over checkpoints; 41-feature step probe
# --------------------------------------------------------------------------- #

def h1_diagram(X, idx):
    P = X[idx].astype(np.float64)
    return diagrams(P, scale=float(np.median(pdist(P))))[1]    # scale-free, so pure norm growth is not "change"


def early_concentration(W, steps, order):
    """Share of the path length sum_t W(o_t, o_t+1) spent in the first third of the log-step range.
    The range is taken over the non-zero checkpoints (log 0 is undefined); the segment from step 0 counts as early."""
    seg = np.array([W[order[t], order[t + 1]] for t in range(len(order) - 1)])
    xs = np.log10(steps[steps > 0])
    cut = xs.min() + (xs.max() - xs.min()) / 3
    first = np.log10(steps[1:]) <= cut
    return float(seg[first].sum() / seg.sum())


def wasserstein_path(dgms, x, steps, P=2000, seed=0):
    from gudhi.wasserstein import wasserstein_distance
    n = len(dgms)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = wasserstein_distance(dgms[i], dgms[j], order=1, internal_p=2)
    rng = np.random.default_rng(seed)
    C = early_concentration(W, steps, np.arange(n))
    null = np.array([early_concentration(W, steps, rng.permutation(n)) for _ in range(P)])
    vel = [W[t, t + 1] / max(x[t + 1] - x[t], 1e-9) for t in range(n - 1)]
    return W, vel, C, float((np.sum(null >= C) + 1) / (P + 1)), float(null.mean())


def feature_rows(X, nums, rng, K=8, m=300):
    four = np.where(nums >= 1000)[0]
    rows = []
    for _ in range(K):
        idx = rng.choice(four, m, replace=False)
        P = X[idx].astype(np.float64)
        rows.append(barcode_features(diagrams(P, scale=float(np.median(pdist(P))))))
    return rows


def step_probe(F, groups, y):
    """Leave-one-checkpoint-out ridge regression of log step from the 41 features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(F[tr], y[tr])
        pred[te] = mdl.predict(F[te])
    return pred


# --------------------------------------------------------------------------- #

def run_family(data, fam, out, pool, P=100):
    revs = revisions(data, fam)
    if not revs:
        return None
    print(f"{fam}: {len(revs)} checkpoints: {', '.join(revs)}", flush=True)
    nums0 = number_subset()
    mt = levels(ex.load_counts(), 10).set_index("n")
    knn, meta_rows, c2_jobs, feats, dgms = [], [], [], [], {d: [] for d in range(5)}
    rng = np.random.default_rng(21)
    h1_idx = None
    for i, rev in enumerate(revs):
        Xs, nums, meta = load_rev(data, fam, rev)
        assert np.array_equal(nums, nums0), "checkpoint numbers differ from number_subset()"
        lab = mt["level"].reindex(nums).fillna(-1).astype(int).to_numpy(copy=True)
        lab[nums < 1000] = -1
        if h1_idx is None:
            h1_idx = np.sort(rng.choice(np.where(nums >= 1000)[0], 600, replace=False))
        meta_rows.append(dict(family=fam, revision=rev, step=step_of(rev), copy_logprob=meta["copy_logprob"],
                              copy_first_token_acc=meta["copy_first_token_acc"]))
        for d, X in enumerate(Xs):
            knn.append(dict(family=fam, revision=rev, step=step_of(rev), depth=DEPTH[d], **knn_comp(X, nums)))
            sel = np.where((lab == 0) | (lab == 9))[0]                 # only the two tenths go to the workers
            c2_jobs.append((fam, rev, d, X[sel].astype(np.float16), lab[sel], 1000 * i + d, P))
            dgms[d].append(h1_diagram(X, h1_idx))
            for r in feature_rows(X, nums, rng):
                feats.append(dict(family=fam, revision=rev, step=step_of(rev), depth=DEPTH[d], **r))
        print(f"  {rev}: loaded, kNN done", flush=True)
    knn, meta_df, feats = pd.DataFrame(knn), pd.DataFrame(meta_rows), pd.DataFrame(feats)
    c2 = pd.DataFrame(pool.map(_c2_cell, c2_jobs))
    print(f"{fam}: C2 permutations done", flush=True)

    steps = meta_df["step"].values
    x = np.log10(steps + 1)                       # log step; step 0 -> 0
    lp = meta_df["copy_logprob"].values
    lp_half = halfway(x, lp, lp[0] + 0.5 * (lp[-1] - lp[0]))

    c1 = []
    for d in range(5):
        k = knn[knn.depth == DEPTH[d]].sort_values("step")
        row = dict(family=fam, depth=DEPTH[d])
        for msr in C1_MEASURES:
            row[f"rho_{msr}"] = float(spearmanr(x, k[msr].values)[0])
        y = k["share_same_leading"].values
        f = fit_line_vs_step(x, y)
        row.update(aic_line=f["aic_line"], aic_step=f["aic_step"], step_fit_wins=f["aic_step"] < f["aic_line"],
                   step_break_between=f"{steps[f['step_break_after']]}-{steps[f['step_break_after'] + 1]}")
        lead_half = halfway(x, y, 0.5 * y[-1])                         # pre-registered: half of the final value
        lead_half_gain = halfway(x, y, y[0] + 0.5 * (y[-1] - y[0]))    # sensitivity: half of the gain
        row.update(lead_halfway_step=10 ** lead_half - 1, lead_halfway_gain_step=10 ** lead_half_gain - 1,
                   copy_halfway_step=10 ** lp_half - 1, C1c_value_first=bool(lead_half < lp_half),
                   C1c_value_first_gain=bool(lead_half_gain < lp_half),
                   C1a=bool(row["rho_share_same_leading"] > 0 and row["rho_share_same_last"] < 0))
        c1.append(row)
    c1 = pd.DataFrame(c1)

    c2_sum = []
    late = steps >= 0.1 * steps.max()
    for d in range(5):
        c = c2[c2.depth == DEPTH[d]].sort_values("step")
        lead_half = 10 ** halfway(x, knn[knn.depth == DEPTH[d]].sort_values("step")["share_same_leading"].values,
                                  0.5 * knn[knn.depth == DEPTH[d]].sort_values("step")["share_same_leading"].values[-1]) - 1
        row = dict(family=fam, depth=DEPTH[d])
        for name in ("raw", "scalefree"):
            r = c[f"ratio_{name}"].values; p = c[f"p_{name}"].values
            sig = steps[p < 0.05]
            rho = float(spearmanr(x[late], r[late])[0])
            toward_one = rho > 0 if r[-1] < 1 else rho < 0
            row.update({f"final_ratio_{name}": r[-1], f"final_p_{name}": p[-1], f"C2a_{name}": bool(r[-1] < 1 and p[-1] < 0.05),
                        f"rho_late_{name}": rho, f"C2b_{name}": "catch-up (toward 1)" if toward_one else "divergence (away from 1)",
                        f"first_sig_step_{name}": int(sig[0]) if len(sig) else -1,
                        f"C2c_{name}": bool(len(sig) and sig[0] >= lead_half)})
        c2_sum.append(row)
    c2_sum = pd.DataFrame(c2_sum)

    sec = []
    for d in range(5):
        W, vel, C, pC, Cnull = wasserstein_path(dgms[d], x, steps)
        fd = feats[feats.depth == DEPTH[d]]
        pred = step_probe(fd[FEATURE_NAMES].values, fd["revision"].values, np.log10(fd["step"].values + 1))
        yt = np.log10(fd["step"].values + 1)
        sec.append(dict(family=fam, depth=DEPTH[d], early_concentration=C, early_concentration_null=Cnull, p_early=pC,
                        peak_velocity_between=f"{steps[int(np.argmax(vel))]}-{steps[int(np.argmax(vel)) + 1]}",
                        probe_r2=float(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)),
                        probe_spearman=float(spearmanr(yt, pred)[0])))
        np.savetxt(os.path.join(out, f"{fam}_W_{d}.csv"), W, delimiter=",", fmt="%.5f")
    sec = pd.DataFrame(sec)

    knn = knn.merge(meta_df[["revision", "copy_logprob", "copy_first_token_acc"]], on="revision")
    knn.to_csv(os.path.join(out, f"{fam}_knn.csv"), index=False)
    c2.to_csv(os.path.join(out, f"{fam}_C2_per_checkpoint.csv"), index=False)
    feats.to_csv(os.path.join(out, f"{fam}_features.csv.gz"), index=False)
    pd.DataFrame([knn_chance(nums0)]).to_csv(os.path.join(out, "knn_chance.csv"), index=False)
    return c1, c2_sum, sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e11_ckpt")
    ap.add_argument("--families", default="pythia-1.4b,olmo2-1b")
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = []
    with Pool(a.workers) as pool:
        for fam in a.families.split(","):
            r = run_family(a.data, fam, a.out, pool, a.perms)
            if r is not None:
                res.append(r)
    for j, name in enumerate(("C1", "C2", "secondary")):
        t = pd.concat([r[j] for r in res])
        t.to_csv(os.path.join(a.out, f"{name}_summary.csv"), index=False)
        print(f"\n== {name} ==\n" + t.to_string(index=False), flush=True)
    c1 = pd.concat([r[0] for r in res]); prim = c1[c1.depth == DEPTH[PRIMARY]]
    print("\nDecision (primary ~84 % layer, all families present):",
          f"C1a {'PASS' if prim.C1a.all() else 'FAIL'}, C1b {'PASS' if prim.step_fit_wins.all() else 'FAIL'},",
          f"C1c {'PASS' if prim.C1c_value_first.all() else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
