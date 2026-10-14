"""E10 stage A (exploratory): does the topology of Crystal's number representations depend on
how often each number appears in the training data, with magnitude and digit form held fixed?

Data: Crystal final checkpoint, one prompt per integer 1..10000, last-token hidden states at
layers 8, 16, 24, 27, 32 (`topology_number_line/data/crystal_vectors/layer_LL.npz`, centred
float16). Frequency: SlimPajama-627B counts (`results/corpus_alpha_full/slimpajama-627b_reupload`),
Crystal's main pretraining corpus.

Everything here is exploratory (discovery data). Measures that survive are to be locked and
confirmed on new data (checkpoints), see experiments/E10_frequency_topology/README.md.

Run:  python -m src.e10_explore --vectors <dir with layer_LL.npz> --out results/e10_explore
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from src.topo_features import FEATURE_NAMES, barcode_features, diagrams

LAYERS = [8, 16, 24, 27, 32]
M = 120                     # points per subsample (years have 126, so every group can use 120)
COUNTS = "results/corpus_alpha_full/slimpajama-627b_reupload/counts_0_to_10000.csv"
FOUR = np.arange(1000, 10000)


def year_zone(n):
    return (n >= 1500) & (n <= 2099)


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #

def load_counts() -> np.ndarray:
    c = pd.read_csv(COUNTS).set_index("number")["count"].reindex(range(10001)).fillna(0)
    return c.values.astype(float)


def matched_tiers(cnt: np.ndarray, control: str = "block") -> pd.DataFrame:
    """Ordinary 4-digit numbers (no 1500-2099, no multiples of 10) split into thirds by count
    inside their own block of 100 (e.g. 5300-5399): tier 0 = rarest third, 2 = most frequent.
    Every block gives 30 numbers to each tier, so the tiers have the same magnitudes.

    control="block": rank on the raw count. The frequent third is then enriched in numbers
        ending in 1 or 5 (digit form and frequency are mixed).
    control="block_digits": rank on the count left over after an additive fit
        log10 count ~ block + tens digit + last digit (holds digit form fixed as well)."""
    n = FOUR[~year_zone(FOUR) & (FOUR % 10 != 0)]
    df = pd.DataFrame({"n": n, "block": n // 100, "logc": np.log10(cnt[n] + 1)})
    key = df["logc"]
    if control == "block_digits":
        dummies = pd.get_dummies(pd.DataFrame({"b": df["block"], "t": (n // 10) % 10, "u": n % 10}).astype(str))
        A = dummies.values.astype(float)
        beta, *_ = np.linalg.lstsq(A, df["logc"].values, rcond=None)
        key = df["logc"] - A @ beta
    elif control != "block":
        raise ValueError(control)
    df["key"] = key
    df["rank"] = df.groupby("block")["key"].rank(method="first") - 1
    df["tier"] = (df["rank"] * 3 // df.groupby("block")["n"].transform("size")).astype(int)
    return df


def naive_tiers(cnt: np.ndarray) -> pd.DataFrame:
    """Thirds of all 4-digit numbers by count, ignoring magnitude (confounded on purpose)."""
    lc = np.log10(cnt[FOUR] + 1)
    t = np.digitize(lc, np.percentile(lc, [100 / 3, 200 / 3]))
    return pd.DataFrame({"n": FOUR, "tier": t})


def special_sets() -> dict[str, np.ndarray]:
    nz = FOUR[~year_zone(FOUR)]
    return {
        "years": np.arange(1900, 2026),
        "year_twins": np.concatenate([np.arange(1900, 2026) + s for s in range(2000, 8000, 1000)]),
        "round_x00_x50": nz[nz % 50 == 0],
        "tens_other": nz[(nz % 10 == 0) & (nz % 50 != 0)],
    }


def tier_subsample(tiers: pd.DataFrame, labels: np.ndarray, rng, blocks_pool=None, per_block=2):
    """One subsample per tier, drawn from the same 60 blocks (so magnitudes are matched)."""
    blocks = np.array(sorted(tiers["block"].unique() if blocks_pool is None else blocks_pool))
    chosen = rng.choice(blocks, M // per_block, replace=False)
    out = {}
    for t in (0, 1, 2):
        idx = []
        for b in chosen:
            cand = tiers["n"].values[(tiers["block"].values == b) & (labels == t)]
            idx.extend(rng.choice(cand, per_block, replace=False))
        out[t] = np.array(idx)
    return out


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #

_X: dict[int, np.ndarray] = {}
_SCALE: dict[int, float] = {}


def _init(vec_dir):
    for L in LAYERS:
        X = np.load(os.path.join(vec_dir, f"layer_{L:02d}.npz"))["X"].astype(np.float32)
        _X[L] = X
        rs = np.random.default_rng(0).choice(FOUR, 1000, replace=False)
        _SCALE[L] = float(np.median(pdist(X[rs - 1])))      # layer scale: typical 4-digit distance


def _feat(args):
    L, idx = args
    return barcode_features(diagrams(_X[L][np.asarray(idx) - 1], _SCALE[L]))


def _dgm(args):
    L, idx = args
    return diagrams(_X[L][np.asarray(idx) - 1], _SCALE[L])


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #

def profiles(pool, cnt, K=50, seed=1):
    """A: mean of every feature for every group at every layer."""
    rng = np.random.default_rng(seed)
    mt, md, nt, sp = matched_tiers(cnt), matched_tiers(cnt, "block_digits"), naive_tiers(cnt), special_sets()
    jobs, meta = [], []
    for k in range(K):
        sub = tier_subsample(mt, mt["tier"].values, rng)
        groups = {f"block_{['low', 'mid', 'high'][t]}": sub[t] for t in (0, 1, 2)}
        sub = tier_subsample(md, md["tier"].values, rng)
        groups.update({f"blockdigit_{['low', 'mid', 'high'][t]}": sub[t] for t in (0, 1, 2)})
        for t in (0, 1, 2):
            groups[f"naive_{['low', 'mid', 'high'][t]}"] = rng.choice(nt["n"].values[nt["tier"].values == t], M, replace=False)
        for name, s in sp.items():
            groups[name] = rng.choice(s, M, replace=False)
        for L in LAYERS:
            for g, idx in groups.items():
                jobs.append((L, idx)); meta.append(dict(layer=L, group=g, rep=k, median_count=float(np.median(cnt[idx])),
                                                        mean_value=float(np.mean(idx))))
    feats = pool.map(_feat, jobs, chunksize=20)
    return pd.DataFrame([{**m, **f} for m, f in zip(meta, feats)])


def tier_test(pool, cnt, control, K=20, n_null=200, seed=2):
    """B: high minus low (matched tiers) for every feature, against shuffling the tier labels
    inside each block (which keeps magnitudes but destroys the frequency ordering)."""
    rng = np.random.default_rng(seed)
    mt = matched_tiers(cnt, control)
    blocks = mt["block"].values

    def shuffled():
        lab = mt["tier"].values.copy()
        for b in np.unique(blocks):
            i = np.where(blocks == b)[0]; lab[i] = rng.permutation(lab[i])
        return lab

    label_sets = [mt["tier"].values] + [shuffled() for _ in range(n_null)]
    jobs, meta = [], []
    for j, lab in enumerate(label_sets):
        for k in range(K):
            sub = tier_subsample(mt, lab, rng)
            for L in LAYERS:
                for t in (0, 2):
                    jobs.append((L, sub[t])); meta.append((j, L, t))
    feats = pool.map(_feat, jobs, chunksize=50)
    df = pd.DataFrame([dict(perm=j, layer=L, tier=t, **f) for (j, L, t), f in zip(meta, feats)])
    m = df.groupby(["perm", "layer", "tier"])[FEATURE_NAMES].mean()
    diff = (m.xs(2, level="tier") - m.xs(0, level="tier")).reset_index()
    rows = []
    for L in LAYERS:
        d = diff[diff.layer == L].set_index("perm")[FEATURE_NAMES]
        real, null = d.loc[0], d.drop(index=0)
        for f in FEATURE_NAMES:
            sd = null[f].std()
            p = (np.sum(np.abs(null[f] - null[f].mean()) >= abs(real[f] - null[f].mean())) + 1) / (len(null) + 1)
            rows.append(dict(layer=L, feature=f, high_minus_low=real[f], null_sd=sd,
                             z=(real[f] - null[f].mean()) / sd if sd > 0 else 0.0, p=p))
    out = pd.DataFrame(rows)
    out["q_bh"] = _bh(out["p"].values)      # Benjamini-Hochberg over all layer x feature tests
    return out


def _bh(p):
    n = len(p); order = np.argsort(p); q = np.empty(n)
    ranked = p[order] * n / (np.arange(n) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1)


def classify(pool, cnt, control, K=48, n_null=30, seed=3):
    """C: Fay-style. Logistic regression on the 41 features, high vs low tier. Blocks are split
    in two halves; train on subsamples from one half, test on the other (no shared numbers)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    mt = matched_tiers(cnt, control)
    ub = np.array(sorted(mt["block"].unique())); rng.shuffle(ub)
    halves = [ub[: len(ub) // 2], ub[len(ub) // 2:]]

    def build(lab):
        jobs, meta = [], []
        for h, pool_b in enumerate(halves):
            for k in range(K):
                sub = tier_subsample(mt, lab, rng, blocks_pool=pool_b, per_block=3)
                for L in LAYERS:
                    for t in (0, 2):
                        jobs.append((L, sub[t])); meta.append((h, L, t))
        feats = pool.map(_feat, jobs, chunksize=50)
        return pd.DataFrame([dict(half=h, layer=L, y=int(t == 2), **f) for (h, L, t), f in zip(meta, feats)])

    def aucs(df):
        res = {}
        for L in LAYERS:
            d = df[df.layer == L]; s = []
            for tr, te in ((0, 1), (1, 0)):
                A, B = d[d.half == tr], d[d.half == te]
                sc = StandardScaler().fit(A[FEATURE_NAMES])
                clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(A[FEATURE_NAMES]), A.y)
                s.append(roc_auc_score(B.y, clf.predict_proba(sc.transform(B[FEATURE_NAMES]))[:, 1]))
            res[L] = float(np.mean(s))
        return res

    real = aucs(build(mt["tier"].values))
    blocks = mt["block"].values
    null = []
    for _ in range(n_null):
        lab = mt["tier"].values.copy()
        for b in np.unique(blocks):
            i = np.where(blocks == b)[0]; lab[i] = rng.permutation(lab[i])
        null.append(aucs(build(lab)))
    null = pd.DataFrame(null)
    return pd.DataFrame([dict(layer=L, auc=real[L], null_mean=null[L].mean(), null_95=null[L].quantile(0.95),
                              p=(np.sum(null[L] >= real[L]) + 1) / (len(null) + 1)) for L in LAYERS])


def per_number(cnt, vec_dir, control):
    """D: every ordinary 4-digit number on its own. Local geometry vs frequency relative to its
    block (log count minus block mean), so magnitude is removed."""
    mt = matched_tiers(cnt, control)
    mt["rel_logc"] = mt["key"] - mt.groupby("block")["key"].transform("mean")
    rows = []
    for L in LAYERS:
        X = np.load(os.path.join(vec_dir, f"layer_{L:02d}.npz"))["X"].astype(np.float32)
        F = X[FOUR - 1]; sc = float(np.median(pdist(F[np.random.default_rng(0).choice(len(F), 1000, replace=False)])))
        sq = (F ** 2).sum(1)
        n = mt["n"].values
        step = (np.linalg.norm(X[n] - X[n - 1], axis=1) + np.linalg.norm(X[n - 1] - X[n - 2], axis=1)) / 2 / sc
        knn, nrm = [], np.linalg.norm(X[n - 1], axis=1) / sc
        for start in range(0, len(n), 500):
            q = X[n[start:start + 500] - 1]
            d2 = (q ** 2).sum(1)[:, None] + sq[None, :] - 2 * q @ F.T
            d = np.sqrt(np.maximum(np.sort(d2, axis=1)[:, 1:11], 0))          # 10 nearest, self excluded
            knn.append(d)
        knn = np.vstack(knn) / sc
        twonn = knn[:, 1] / knn[:, 0]
        for name, v in {"step_to_neighbours": step, "knn1_dist": knn[:, 0], "knn10_dist": knn[:, 9],
                        "twonn_ratio": twonn, "norm_from_centre": nrm}.items():
            r, p = spearmanr(mt["rel_logc"], v)
            rows.append(dict(layer=L, measure=name, spearman_with_rel_freq=r, p=p, n=len(n)))
    return pd.DataFrame(rows)


def layer_change(pool, cnt, K=30, seed=4):
    """E: how much each group's topology changes from one layer to the next (Wasserstein-1
    between diagrams of the same numbers at consecutive layers)."""
    from gudhi.wasserstein import wasserstein_distance
    rng = np.random.default_rng(seed)
    mt, sp = matched_tiers(cnt), special_sets()
    rows = []
    for k in range(K):
        sub = tier_subsample(mt, mt["tier"].values, rng)
        groups = {"block_low": sub[0], "block_mid": sub[1], "block_high": sub[2],
                  "years": rng.choice(sp["years"], M, replace=False),
                  "year_twins": rng.choice(sp["year_twins"], M, replace=False)}
        for g, idx in groups.items():
            dg = pool.map(_dgm, [(L, idx) for L in LAYERS])
            for a in range(len(LAYERS) - 1):
                for h in (0, 1):
                    A = dg[a][h][np.isfinite(dg[a][h][:, 1])]; B = dg[a + 1][h][np.isfinite(dg[a + 1][h][:, 1])]
                    rows.append(dict(group=g, rep=k, step=f"L{LAYERS[a]}->L{LAYERS[a + 1]}", dim=f"H{h}",
                                     w1=wasserstein_distance(A, B, order=1, internal_p=2)))
    return pd.DataFrame(rows)


def special_isolation(cnt, vec_dir):
    """F: how much room special numbers get. For each number, mean distance to its 10 nearest
    4-digit numbers, as a z-score against ordinary numbers of the same block of 100 (nearest
    block with ordinary numbers for 1500-2099). > 0 means more isolated than its neighbours."""
    s = pd.Series(FOUR.astype(str), index=FOUR)
    yz = year_zone(FOUR)
    cats = {
        "thousands x000": FOUR[FOUR % 1000 == 0],
        "hundreds x00": FOUR[(FOUR % 100 == 0) & (FOUR % 1000 != 0) & ~yz],
        "fifties x50": FOUR[(FOUR % 100 == 50) & ~yz],
        "years 1950-2025": np.arange(1950, 2026),
        "years 1800-1899": np.arange(1800, 1900),
        "repdigits 1111-9999": np.array([int(str(d) * 4) for d in range(1, 10)]),
        "powers of 2": np.array([1024, 2048, 4096, 8192]),
        "digit runs 1234, 4321, ...": np.array([1234, 2345, 3456, 4567, 5678, 6789, 4321, 5432, 6543, 7654, 8765, 9876]),
    }
    ordinary = FOUR[(FOUR % 10 != 0) & ~yz & s.map(lambda x: len(set(x)) > 2).values]
    rows = []
    for L in LAYERS:
        X = np.load(os.path.join(vec_dir, f"layer_{L:02d}.npz"))["X"].astype(np.float32)
        F = X[FOUR - 1]; sq = (F ** 2).sum(1)

        def knn(nums):
            q = X[nums - 1]; d2 = (q ** 2).sum(1)[:, None] + sq[None, :] - 2 * q @ F.T
            return np.sqrt(np.maximum(np.sort(d2, 1)[:, 1:11], 0)).mean(1)

        ko = pd.Series(knn(ordinary), index=ordinary).groupby(ordinary // 100)
        med, sd = ko.median(), ko.std()
        nearest = lambda b: med.index[np.abs(med.index - b).argmin()]
        for name, nums in cats.items():
            b = [nearest(x) for x in nums // 100]
            z = (knn(nums) - med.loc[b].values) / sd.loc[b].values
            rows.append(dict(layer=L, category=name, n=len(nums), median_count=float(np.median(cnt[nums])),
                             mean_z=float(z.mean()), share_above=float((z > 0).mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", default="results/e10_explore")
    ap.add_argument("--only", default="ABCDE")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cnt = load_counts()
    pd.concat([matched_tiers(cnt, c).assign(control=c) for c in ("block", "block_digits")]).to_csv(
        os.path.join(a.out, "matched_tiers.csv"), index=False)
    with Pool(a.workers, initializer=_init, initargs=(a.vectors,)) as pool:
        if "A" in a.only:
            profiles(pool, cnt).to_csv(os.path.join(a.out, "A_profiles.csv"), index=False); print("A done", flush=True)
        for ctl in ("block", "block_digits"):
            if "B" in a.only:
                tier_test(pool, cnt, ctl).to_csv(os.path.join(a.out, f"B_tier_test_{ctl}.csv"), index=False); print("B", ctl, "done", flush=True)
            if "C" in a.only:
                classify(pool, cnt, ctl).to_csv(os.path.join(a.out, f"C_classifier_{ctl}.csv"), index=False); print("C", ctl, "done", flush=True)
        if "E" in a.only:
            layer_change(pool, cnt).to_csv(os.path.join(a.out, "E_layer_change.csv"), index=False); print("E done", flush=True)
    if "D" in a.only:
        pd.concat([per_number(cnt, a.vectors, ctl).assign(control=ctl) for ctl in ("block", "block_digits")]).to_csv(
            os.path.join(a.out, "D_per_number.csv"), index=False); print("D done", flush=True)
    if "F" in a.only:
        special_isolation(cnt, a.vectors).to_csv(os.path.join(a.out, "F_special.csv"), index=False); print("F done", flush=True)
    json.dump(dict(layers=LAYERS, m=M, counts=COUNTS, features=FEATURE_NAMES), open(os.path.join(a.out, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
