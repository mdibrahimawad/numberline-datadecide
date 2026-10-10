"""Everything computed on one cell (model, input, checkpoint, layer)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import topology as T
from .linear import linear_cell, _m, _shuffle_within
from .mst import T3

KINDS_SAVED = ["true", "false", "mixed"]          # diagrams kept for velocity


def make_subsamples(y, topics, seed):
    """Fixed subsample index sets for one seed: true, false, mixed and N_NULL_T1 pseudo-class
    pairs (nullA_k, nullB_k). Identical for every checkpoint and layer."""
    rng = np.random.default_rng(seed)
    topic_id = np.unique(topics, return_inverse=True)[1]
    strat = topic_id * 2 + y
    allidx = np.arange(len(y))
    S = {"true": T.stratified_subsamples(topic_id, allidx[y == 1], C.B, C.M, rng),
         "false": T.stratified_subsamples(topic_id, allidx[y == 0], C.B, C.M, rng),
         "mixed": T.stratified_subsamples(strat, allidx, C.B, C.M, rng)}
    for k in range(C.N_NULL_T1):
        # random half split, each half with the same topic x truth mix (mixed vs mixed)
        a = np.zeros(len(y), bool)
        for s in np.unique(strat):
            i = rng.permutation(np.where(strat == s)[0])
            a[i[: len(i) // 2]] = True
        S[f"nullA_{k}"] = T.stratified_subsamples(topic_id, allidx[a], C.B, C.M, rng)
        S[f"nullB_{k}"] = T.stratified_subsamples(topic_id, allidx[~a], C.B, C.M, rng)
    return S


def shape_accuracy(Fa, Fb, seed=0):
    """T1: 5-fold CV accuracy of a logistic regression telling subsample summaries of A from B."""
    X = np.vstack([Fa, Fb])
    y = np.r_[np.ones(len(Fa)), np.zeros(len(Fb))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=C.LR_MAX_ITER))
    return float(cross_val_score(clf, X, y, cv=StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=seed)).mean())


def pack(dgms):
    """List of (H0, H1) -> flat arrays + offsets, for npz."""
    out = {}
    for h in (0, 1):
        arrs = [d[h] for d in dgms]
        out[f"h{h}"] = np.concatenate(arrs) if arrs else np.empty((0, 2))
        out[f"h{h}_off"] = np.cumsum([0] + [len(a) for a in arrs])
    return out


def unpack(z, prefix):
    res = []
    o0, o1 = z[f"{prefix}_h0_off"], z[f"{prefix}_h1_off"]
    h0, h1 = z[f"{prefix}_h0"], z[f"{prefix}_h1"]
    for b in range(len(o0) - 1):
        res.append([h0[o0[b]:o0[b + 1]], h1[o1[b]:o1[b + 1]]])
    return res


def topology_cell(D_norm, y, subs: dict, rng):
    """PH of every subsample, T1 accuracy vs T1-null, key-feature effect sizes."""
    dg = {k: [T.diagrams(D_norm[np.ix_(ix, ix)]) for ix in idx] for k, idx in subs.items()}
    F = {k: np.vstack([T.summary(d) for d in v]) for k, v in dg.items()}
    t1 = shape_accuracy(F["true"], F["false"])
    null = [shape_accuracy(F[f"nullA_{k}"], F[f"nullB_{k}"]) for k in range(C.N_NULL_T1)]
    rec = [_m("T1_acc", t1, null), {**_m("T1_gap", t1 - float(np.mean(null))), "null_mean": 0.0}]
    for f in C.KEY_FEATURES:
        j = T.FEATURE_NAMES.index(f)
        g, lo, hi = T.hedges_g_boot(F["true"][:, j], F["false"][:, j], C.N_BOOT, rng)
        rec.append({"metric": f"T1g_{f}", "value": g, "ci_lo": lo, "ci_hi": hi,
                    "diff": float(F["true"][:, j].mean() - F["false"][:, j].mean())})
    return rec, dg, F


def t3_cell(X, y, topics):
    # one fixed permutation sequence for every cell: permutation k is the same relabelling at every
    # layer and checkpoint, which the max-statistic correction over layers needs
    rng = np.random.default_rng(C.SEED_PERM)
    t3 = T3(X, topics)
    obs = t3.values(y)
    strata = np.unique(topics, return_inverse=True)[1]
    names = ["T3a_topic_homophily", "T3b_within_truth_homophily", "T3c_cross_truth_homophily"]
    null = np.zeros((C.N_PERM_MST, 3))
    for p in range(C.N_PERM_MST):
        v = t3.values(_shuffle_within(y, strata, rng), rng.permutation(topics))
        null[p] = [v[n] for n in names]
    recs = [_m(n, obs[n], null[:, i]) for i, n in enumerate(names)]
    recs.append(_m("T3c_n_cross_edges", obs["T3c_n_cross_edges"]))
    return recs, null


def geometry(X, scale):
    """Geometric baselines (Malhotra et al.): centroid, total variance, mean pairwise distance."""
    c = X.mean(0)
    tv = float(((X - c) ** 2).sum(1).mean())
    return {"centroid": c, "total_var": tv, "scale": scale}


def run_cell(X16, y, topics, subs_by_seed, ref_idx, seed, out_npz, linear=True):
    """Compute and save one cell. Returns the metric records (each tagged with the subsample seed)."""
    rng = np.random.default_rng(seed)
    X = X16.astype(np.float64)
    D = T.distances(X)
    scale = T.norm_scale(D, ref_idx)
    D /= scale                                   # primary: centred (PH is translation invariant) / median distance
    iu = np.triu_indices(len(D), 1)
    recs = []
    if linear:
        recs += [{**r, "seed": -1} for r in linear_cell(X, y, topics, C.N_NULL_LINEAR, rng)]
    t3recs, t3null = t3_cell(X, y, topics)
    recs += [{**r, "seed": -1} for r in t3recs]
    g = geometry(X, scale)
    recs += [{**_m("geo_total_var", g["total_var"]), "seed": -1},
             {**_m("geo_scale_median_dist", scale), "seed": -1},
             {**_m("geo_mean_pair_dist", float(D[iu].mean() * scale)), "seed": -1}]
    save = {"t3_null": t3null, "centroid": g["centroid"].astype(np.float32), "scale": scale}
    summaries = {}
    for s, subs in subs_by_seed.items():
        trec, dg, F = topology_cell(D, y, subs, rng)
        recs += [{**r, "seed": s} for r in trec]
        for k in KINDS_SAVED:
            save.update({f"s{s}_{k}_{key}": v for key, v in pack(dg[k]).items()})
        summaries.update({f"s{s}_{k}": F[k] for k in ["true", "false", "mixed", "nullA_0", "nullB_0"]})
    np.savez(out_npz, **save)
    return recs, summaries
