"""E11 tier 0 (exploratory): topology hypotheses on the existing 7-model number vectors.
See experiments/E11_topology_hypotheses/PLAN.md for the hypotheses (H#) and predictions.

Every function returns one tidy table (one row per model x depth x condition = a "cell").
Inference is per cell against a permutation of the labels, never treating overlapping
subsamples as independent.

  python -m src.e11_tier0 --data <topology_number_line>/data --out results/e11_tier0 --only all
"""
from __future__ import annotations

import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import src.e10_explore as ex
from src.e10_graded import DEPTH, MODELS, layer_files, levels
from src.topo_features import FEATURE_NAMES, barcode_features, diagrams

DIGIT_MODELS = {"crystal", "openllama-3b", "starcoderbase-3b", "qwen2.5-1.5b", "smollm2-1.7b"}
TOKENIZER = {m: ("digit" if m in DIGIT_MODELS else "chunk") for m in MODELS}
FOUR = ex.FOUR
DATA = None


def load(m, d):
    X = np.load(layer_files(DATA, m)[d])["X"].astype(np.float32)
    rs = np.random.default_rng(0).choice(FOUR, 1000, replace=False)
    return X, float(np.median(pdist(X[rs - 1])))


def cells():
    return [(m, d) for m in MODELS for d in range(5)]


def h0_deaths(P):
    """H0 death times of a Rips filtration = edge lengths of the Euclidean MST."""
    D = squareform(pdist(P.astype(np.float64)))
    return np.sort(minimum_spanning_tree(D).data)


def twonn(P):
    """TwoNN intrinsic dimension (Facco et al. 2017), maximum-likelihood form."""
    D = squareform(pdist(P.astype(np.float64))); np.fill_diagonal(D, np.inf)
    s = np.sort(D, 1)[:, :2]
    mu = s[:, 1] / s[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.log(mu).sum())


# --------------------------------------------------------------------------- #
# H1 / H2: crowding of the rarest vs most frequent tenth, cell-level permutation
# --------------------------------------------------------------------------- #

def _h12_cell(args):
    m, d, K, P, seed = args
    rng = np.random.default_rng(seed)
    X, s = load(m, d)
    mt = levels(ex.load_counts(), 10)
    n, b, lab0 = mt["n"].values, mt["block"].values, mt["level"].values
    ub = np.unique(b)
    byb = {k: np.where(b == k)[0] for k in ub}

    def stat(lab):
        raw, free = [], []
        for _ in range(K):
            blocks = rng.choice(ub, 60, replace=False)
            sets = {}
            for q in (0, 9):
                idx = np.concatenate([rng.choice(byb[k][lab[byb[k]] == q], 2, replace=False) for k in blocks])
                sets[q] = X[n[idx] - 1]
            r = {q: np.median(h0_deaths(sets[q])) for q in (0, 9)}
            f = {q: r[q] / np.median(pdist(sets[q])) for q in (0, 9)}
            raw.append(np.log(r[0] / r[9])); free.append(np.log(f[0] / f[9]))
        return np.mean(raw), np.mean(free)

    real = stat(lab0)
    null = []
    for _ in range(P):
        lab = lab0.copy()
        for k in ub:
            lab[byb[k]] = rng.permutation(lab[byb[k]])
        null.append(stat(lab))
    null = np.array(null)
    out = dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m])
    for j, name in enumerate(("raw", "scalefree")):
        out[f"log_r_half_ratio_{name}"] = real[j]                              # < 0: rarest tenth more crowded
        out[f"ratio_{name}"] = float(np.exp(real[j]))
        out[f"p_{name}"] = float((np.sum(null[:, j] <= real[j]) + 1) / (P + 1))  # one-sided
        out[f"z_{name}"] = float((real[j] - null[:, j].mean()) / null[:, j].std())
    return out


def h1_h2(pool, K=12, P=60):
    return pd.DataFrame(pool.map(_h12_cell, [(m, d, K, P, 100 + i) for i, (m, d) in enumerate(cells())]))


# --------------------------------------------------------------------------- #
# H3 / H28: distance to a spectrum-matched Gaussian floor; H4: intrinsic dimension
# --------------------------------------------------------------------------- #

def gaussian_like(P, rng):
    """Gaussian cloud with the same size and covariance spectrum as P."""
    C = P - P.mean(0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return (rng.standard_normal(U.shape) @ np.diag(S) @ Vt) / np.sqrt(len(P) - 1)


def energy(A, B):
    dab = np.linalg.norm(A[:, None] - B[None], axis=2).mean()
    daa = np.linalg.norm(A[:, None] - A[None], axis=2).mean()
    dbb = np.linalg.norm(B[:, None] - B[None], axis=2).mean()
    return float(np.sqrt(max(2 * dab - daa - dbb, 0)))


def _h34_cell(args):
    m, d, K, seed = args
    rng = np.random.default_rng(seed)
    X, s = load(m, d)
    mt = levels(ex.load_counts(), 10)
    n, b, lab = mt["n"].values, mt["block"].values, mt["level"].values
    ub = np.unique(b); byb = {k: np.where(b == k)[0] for k in ub}
    feats = {"low": [], "high": [], "gauss_low": [], "gauss_high": [], "all": [], "gauss_all": []}
    for _ in range(K):
        blocks = rng.choice(ub, 60, replace=False)
        for q, name in ((0, "low"), (9, "high")):
            idx = np.concatenate([rng.choice(byb[k][lab[byb[k]] == q], 2, replace=False) for k in blocks])
            Pq = X[n[idx] - 1]
            feats[name].append(barcode_features(diagrams(Pq, s)))
            feats["gauss_" + name].append(barcode_features(diagrams(gaussian_like(Pq, rng), s)))
        Pa = X[rng.choice(FOUR, 120, replace=False) - 1]
        feats["all"].append(barcode_features(diagrams(Pa, s)))
        feats["gauss_all"].append(barcode_features(diagrams(gaussian_like(Pa, rng), s)))
    F = {k: pd.DataFrame(v)[FEATURE_NAMES].values for k, v in feats.items()}
    mu, sd = np.vstack(list(F.values())).mean(0), np.vstack(list(F.values())).std(0) + 1e-12
    Z = {k: (v - mu) / sd for k, v in F.items()}
    # H4: TwoNN ID of each frequency level (all numbers of the level)
    ids = [twonn(X[n[lab == q] - 1]) for q in range(10)]
    perm_rho = []
    for _ in range(30):
        lp = lab.copy()
        for k in ub:
            lp[byb[k]] = rng.permutation(lp[byb[k]])
        perm_rho.append(spearmanr(range(10), [twonn(X[n[lp == q] - 1]) for q in range(10)]).correlation)
    rho = spearmanr(range(10), ids).correlation
    return dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m],
                energy_low_to_gauss=energy(Z["low"], Z["gauss_low"]), energy_high_to_gauss=energy(Z["high"], Z["gauss_high"]),
                energy_all_to_gauss=energy(Z["all"], Z["gauss_all"]),
                H1_count_real=float(F["all"][:, FEATURE_NAMES.index("H1_count")].mean()),
                H1_count_gauss=float(F["gauss_all"][:, FEATURE_NAMES.index("H1_count")].mean()),
                id_rarest=ids[0], id_most_frequent=ids[9], id_rho_with_level=rho,
                id_p=float((np.sum(np.array(perm_rho) >= rho) + 1) / 31))


def h3_h4(pool, K=10):
    return pd.DataFrame(pool.map(_h34_cell, [(m, d, K, 200 + i) for i, (m, d) in enumerate(cells())]))


# --------------------------------------------------------------------------- #
# H6 / H19 / H14 / H15: neighbourhoods of 4-digit numbers, kNN overlap across models, ID by depth
# --------------------------------------------------------------------------- #

def knn_ids(X, queries, k=10):
    F = X[FOUR - 1]; sq = (F ** 2).sum(1); out = []
    for s in range(0, len(queries), 500):
        q = X[queries[s:s + 500] - 1]
        d2 = (q ** 2).sum(1)[:, None] + sq[None, :] - 2 * q @ F.T
        d2[np.arange(len(q)), queries[s:s + 500] - 1000] = np.inf
        out.append(FOUR[np.argsort(d2, 1)[:, :k]])
    return np.vstack(out)


def _h6_cell(args):
    m, d, queries = args
    X, _ = load(m, d)
    nb = knn_ids(X, queries)
    q = np.repeat(queries, nb.shape[1]); v = nb.ravel()
    sq, sv = np.array([f"{x:04d}" for x in q]), np.array([f"{x:04d}" for x in v])
    ham = np.array([sum(a != b for a, b in zip(x, y)) for x, y in zip(sq, sv)])
    row = dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m],
               share_hamming1=float(np.mean(ham == 1)), share_within10=float(np.mean(np.abs(q - v) <= 10)),
               share_same_leading=float(np.mean(q // 1000 == v // 1000)), share_same_last=float(np.mean(q % 10 == v % 10)),
               share_same_parity=float(np.mean(q % 2 == v % 2)), share_same_mod5=float(np.mean(q % 5 == v % 5)),
               median_abs_diff=float(np.median(np.abs(q - v))), twonn_id=twonn(X[queries[:2000] - 1]))
    np.save(f"/tmp/e11_knn_{m}_{d}.npy", nb)
    return row


def h6(pool, seed=6):
    rng = np.random.default_rng(seed)
    queries = np.sort(rng.choice(FOUR, 2000, replace=False))
    t = pd.DataFrame(pool.map(_h6_cell, [(m, d, queries) for m, d in cells()]))
    # chance levels for the same queries (random neighbours among 4-digit numbers)
    v = rng.choice(FOUR, 20000); q = rng.choice(queries, 20000)
    ham = np.array([sum(a != b for a, b in zip(f"{x:04d}", f"{y:04d}")) for x, y in zip(q, v)])
    chance = dict(share_hamming1=float(np.mean(ham == 1)), share_within10=float(np.mean(np.abs(q - v) <= 10)),
                  share_same_leading=float(np.mean(q // 1000 == v // 1000)), share_same_last=float(np.mean(q % 10 == v % 10)),
                  share_same_parity=float(np.mean(q % 2 == v % 2)), share_same_mod5=float(np.mean(q % 5 == v % 5)))
    # H14: kNN Jaccard overlap between models at the same depth
    rows = []
    for d in range(5):
        nbs = {m: np.load(f"/tmp/e11_knn_{m}_{d}.npy") for m in MODELS}
        for a, b in itertools.combinations(MODELS, 2):
            jac = np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(nbs[a], nbs[b])])
            rows.append(dict(depth=DEPTH[d], a=a, b=b, same_tokenizer=TOKENIZER[a] == TOKENIZER[b], knn_jaccard=jac))
    return t, pd.DataFrame([chance]), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H8: favourite "random" numbers (x7) vs avoided (x0, repdigits) among 10-99
# --------------------------------------------------------------------------- #

def _h8_cell(args):
    m, d, seed = args
    rng = np.random.default_rng(seed)
    X, _ = load(m, d)
    nums = np.arange(10, 100); P = X[nums - 1]
    D = squareform(pdist(P.astype(np.float64))); D /= np.median(D[np.triu_indices(90, 1)])
    central = D.mean(1)                                   # low = central / typical
    iso = np.sort(D, 1)[:, 1:4].mean(1)                   # high = isolated
    groups = {"x7": nums % 10 == 7, "x0": nums % 10 == 0, "repdigit": nums % 11 == 0}
    out = dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m])
    for g, mask in groups.items():
        for name, v in (("centrality", central), ("isolation", iso)):
            obs = v[mask].mean()
            null = np.array([v[rng.choice(90, mask.sum(), replace=False)].mean() for _ in range(2000)])
            out[f"{g}_{name}_z"] = float((obs - null.mean()) / null.std())
    return out


def h8(pool):
    return pd.DataFrame(pool.map(_h8_cell, [(m, d, 300 + i) for i, (m, d) in enumerate(cells())]))


# --------------------------------------------------------------------------- #
# H9: persistent clusters of 4-digit numbers and their purity by leading digit
# --------------------------------------------------------------------------- #

def _h9_cell(args):
    m, d, seed = args
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    X, _ = load(m, d)
    nums = np.concatenate([rng.choice(np.arange(L * 1000, L * 1000 + 1000), 100, replace=False) for L in range(1, 10)])
    Z = linkage(X[nums - 1].astype(np.float64), "single")
    h = np.sort(Z[:, 2])[::-1]                            # merge heights, largest first
    n_persistent = int(np.argmax(h[:30] - h[1:31]) + 2)    # clusters that survive the largest gap among the top merges
    out = dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m], persistent_clusters=n_persistent)
    for link in ("single", "average"):
        Zl = Z if link == "single" else linkage(X[nums - 1].astype(np.float64), "average")
        lab = fcluster(Zl, 9, "maxclust")
        out[f"ari_leading_{link}"] = adjusted_rand_score(nums // 1000, lab)
        out[f"ari_last_{link}"] = adjusted_rand_score(nums % 10, fcluster(Zl, 10, "maxclust"))
    return out


def h9(pool):
    return pd.DataFrame(pool.map(_h9_cell, [(m, d, 400 + i) for i, (m, d) in enumerate(cells())]))


# --------------------------------------------------------------------------- #
# H18 / H20 / H29: loops: last-digit circle in 10-99, ring of 1-100, torus of 00-99 endings
# --------------------------------------------------------------------------- #

def _rel_loops(D, maxdim=1, top=3):
    dg = ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]
    res = {}
    for h in range(1, maxdim + 1):
        x = dg[h]
        rel = np.sort((x[:, 1] - x[:, 0]) / x[:, 1])[::-1] if len(x) else np.zeros(0)
        for i in range(top):
            res[f"H{h}_rel{i + 1}"] = float(rel[i]) if len(rel) > i else 0.0
    return res


def _loops_cell(args):
    from src.e10_topology import circle_stats
    m, d, seed = args
    rng = np.random.default_rng(seed)
    X, _ = load(m, d)
    out = dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m])
    # H18: last-digit circle among 10-99 (tens-group means removed)
    nums = np.arange(10, 100); V = X[nums - 1].copy()
    for t in range(1, 10):
        i = nums // 10 == t; V[i] -= V[i].mean(0)
    cls = nums % 10
    C = np.stack([V[cls == c].mean(0) for c in range(10)])
    rel, order, wrap = circle_stats(C)
    null = [circle_stats(np.stack([V[p == c].mean(0) for c in range(10)]))[0] for p in (rng.permutation(cls) for _ in range(300))]
    out.update(h18_loop=rel, h18_null95=float(np.quantile(null, 0.95)), h18_p=float((np.sum(np.array(null) >= rel) + 1) / 301),
               h18_digit_order=order, h18_wraps=wrap)
    # H20: ring in 1-100 vs a spectrum-matched Gaussian cloud
    P = X[np.arange(1, 101) - 1].astype(np.float64)
    D = squareform(pdist(P)); r = _rel_loops(D)
    g = [_rel_loops(squareform(pdist(gaussian_like(P, rng))))["H1_rel1"] for _ in range(20)]
    out.update(h20_loop=r["H1_rel1"], h20_gauss95=float(np.quantile(g, 0.95)))
    # H29: torus of the 100 two-digit endings (00-99) of 4-digit numbers, block means removed
    nums = FOUR[~ex.year_zone(FOUR)]; V = X[nums - 1].copy()
    for blk in np.unique(nums // 100):
        i = nums // 100 == blk; V[i] -= V[i].mean(0)
    end = nums % 100
    C = np.stack([V[end == e].mean(0) for e in range(100)])
    r = _rel_loops(squareform(pdist(C.astype(np.float64))), maxdim=2, top=2)
    pe = rng.permutation(end); Cn = np.stack([V[pe == e].mean(0) for e in range(100)])
    rn = _rel_loops(squareform(pdist(Cn.astype(np.float64))), maxdim=2, top=2)
    out.update({f"h29_{k}": v for k, v in r.items()}, **{f"h29_null_{k}": v for k, v in rn.items()})
    return out


def loops(pool):
    return pd.DataFrame(pool.map(_loops_cell, [(m, d, 500 + i) for i, (m, d) in enumerate(cells())]))


# --------------------------------------------------------------------------- #
# H10: years ("fewer, larger features") in all 7 models; H16: outlier dimensions
# --------------------------------------------------------------------------- #

def _years_cell(args):
    from src.e10_confirm import years_test
    m, d = args
    r = years_test(os.path.dirname(layer_files(DATA, m)[d]), int(os.path.basename(layer_files(DATA, m)[d])[6:8]))
    return dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m], spread_z=r["spread_z"], h0_total_scalefree_z=r["h0_total_scalefree_z"])


def years(pool):
    return pd.DataFrame(pool.map(_years_cell, cells()))


def _h16_cell(args):
    m, d = args
    X, _ = load(m, d)
    mt = ex.matched_tiers(ex.load_counts(), "block_digits_plain")
    rel = (mt["key"] - mt.groupby("block")["key"].transform("mean")).values
    n = mt["n"].values
    pick = np.random.default_rng(16).choice(len(n), 2500, replace=False)
    rel, n = rel[pick], n[pick]
    var = X[FOUR - 1].var(0); order = np.argsort(var)[::-1]
    out = dict(model=m, depth=DEPTH[d], tokenizer=TOKENIZER[m], top1_var_share=float(var[order[0]] / var.sum()),
               top10_var_share=float(var[order[:10]].sum() / var.sum()))
    for k in (0, 1, 5, 20):
        Y = X.copy(); Y[:, order[:k]] = 0
        F = Y[FOUR - 1]; sq = (F ** 2).sum(1); kd = []
        for s in range(0, len(n), 500):
            q = Y[n[s:s + 500] - 1]; d2 = (q ** 2).sum(1)[:, None] + sq[None, :] - 2 * q @ F.T
            kd.append(np.sqrt(np.maximum(np.sort(d2, 1)[:, 1:11], 0)).mean(1))
        out[f"rho_knn_drop{k}"] = spearmanr(rel, np.concatenate(kd)).correlation
        out[f"rho_norm_drop{k}"] = spearmanr(rel, np.linalg.norm(Y[n - 1], axis=1)).correlation
    return out


def h16(pool):
    return pd.DataFrame(pool.map(_h16_cell, cells()))


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e11_tier0")
    ap.add_argument("--only", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(a.out, exist_ok=True)
    todo = ["h6", "h8", "h9", "loops", "years", "h16", "h12", "h34"] if a.only == "all" else a.only.split(",")
    with Pool(a.workers, initializer=_set_data, initargs=(a.data,)) as pool:
        for key in todo:
            if key == "h6":
                t, chance, jac = h6(pool)
                t.to_csv(f"{a.out}/H6_neighbourhoods.csv", index=False); chance.to_csv(f"{a.out}/H6_chance.csv", index=False)
                jac.to_csv(f"{a.out}/H14_knn_jaccard.csv", index=False)
            else:
                fn = {"h8": h8, "h9": h9, "loops": loops, "years": years, "h16": h16, "h12": h1_h2, "h34": h3_h4}[key]
                name = {"h8": "H8_favourite_numbers", "h9": "H9_clusters", "loops": "H18_H20_H29_loops",
                        "years": "H10_years", "h16": "H16_outlier_dims", "h12": "H1_H2_crowding_cell_level",
                        "h34": "H3_H4_H28_gauss_floor_and_ID"}[key]
                fn(pool).to_csv(f"{a.out}/{name}.csv", index=False)
            print(key, "done", flush=True)


def _set_data(d):
    global DATA
    DATA = d


if __name__ == "__main__":
    main()
