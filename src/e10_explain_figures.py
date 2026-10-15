"""Explanatory figures for E10 stage A (Crystal): what "frequent numbers are further apart" means,
where z comes from, special numbers, and years. Recomputes from the vectors (same procedure as
src/e10_explore.py; the null for z is re-drawn, so z differs slightly from B_tier_test_*.csv).

  python -m src.e10_explain_figures --vectors <topology_number_line>/data/crystal_vectors --out results/e10_explore/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

import src.e10_explore as ex
from src.e10_graded_analysis import GRID, INK, INK2, SURF, _style
from src.topo_features import diagrams

RARE, FREQ, NULL = "#eb6834", "#2a78d6", "#b9b8b2"


def mst_edges(P):
    """H0 death times of the Rips filtration = edge lengths of the Euclidean minimum spanning tree."""
    return np.sort(minimum_spanning_tree(squareform(pdist(P.astype(np.float64)))).data)


def layer(vec, L):
    X = np.load(os.path.join(vec, f"layer_{L:02d}.npz"))["X"].astype(np.float32)
    rs = np.random.default_rng(0).choice(ex.FOUR, 1000, replace=False)
    return X, float(np.median(pdist(X[rs - 1])))


def tier_null(X, scale, mt, K=20, n_null=200, seed=2):
    """Frequent minus rare third, mean H0 death, real labels and n_null within-block shuffles."""
    rng = np.random.default_rng(seed)
    b = mt["block"].values
    out, pooled = [], {0: [], 2: []}
    for j in range(n_null + 1):
        lab = mt["tier"].values.copy()
        if j:
            for blk in np.unique(b):
                i = np.where(b == blk)[0]; lab[i] = rng.permutation(lab[i])
        diffs = []
        for _ in range(K):
            sub = ex.tier_subsample(mt, lab, rng)
            e = {t: mst_edges(X[sub[t] - 1]) / scale for t in (0, 2)}
            diffs.append(e[2].mean() - e[0].mean())
            if j == 0:
                for t in (0, 2):
                    pooled[t].append(e[t])
        out.append(np.mean(diffs))
    return out[0], np.array(out[1:]), {t: np.concatenate(v) for t, v in pooled.items()}


def fig_z(vec, outdir, cnt):
    mt = ex.matched_tiers(cnt, "block_digits")
    res = {}
    for L in (8, 32):
        X, s = layer(vec, L)
        res[L] = tier_null(X, s, mt)
    # 1. what is compared: distribution of MST edge lengths (gaps to the nearest group member)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, L in zip(axes, (8, 32)):
        pooled = res[L][2]
        bins = np.linspace(min(map(np.min, pooled.values())), np.percentile(np.concatenate(list(pooled.values())), 99.5), 45)
        for t, col, lab in ((0, RARE, "rarest third"), (2, FREQ, "most frequent third")):
            ax.hist(pooled[t], bins=bins, histtype="step", color=col, lw=2, label=f"{lab}: mean {pooled[t].mean():.3f}")
            ax.axvline(pooled[t].mean(), color=col, lw=1.2, ls="--")
        ax.set_title(f"layer {L}: gaps between nearest numbers (H0 deaths)", fontsize=9, loc="left")
        ax.set_xlabel("gap length (Euclidean, in units of the layer's typical distance)")
        ax.set_ylabel("count (20 sets × 119 gaps)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A1_gap_histograms.png"), dpi=160); plt.close(fig)
    # 2. where z comes from: real difference vs 200 label shuffles
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, L in zip(axes, (8, 32)):
        real, null, _ = res[L]
        z = (real - null.mean()) / null.std()
        ax.hist(null, bins=25, color=NULL, edgecolor=SURF, label="200 shuffles of rare/frequent labels inside each block")
        ax.axvline(real, color=FREQ, lw=2.5, label=f"real labels: {real:+.4f}")
        ax.set_title(f"layer {L}: z = (real − shuffle mean) / shuffle SD = {z:+.1f}", fontsize=9, loc="left")
        ax.set_xlabel("mean gap (frequent third) − mean gap (rarest third)")
        ax.set_ylabel("number of shuffles")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A2_where_z_comes_from.png"), dpi=160); plt.close(fig)
    return {L: ((r - n.mean()) / n.std()) for L, (r, n, _) in res.items()}


def fig_profiles(outdir):
    A = pd.read_csv("results/e10_explore/A_profiles.csv")
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.0))
    for ax, L in zip(axes, ex.LAYERS):
        for k, (g, lab) in enumerate((("low", "rare"), ("mid", "middle"), ("high", "frequent"))):
            v = A[(A.layer == L) & (A.group == f"blockdigit_{g}")]["H0_death_mean"]
            ci = 1.96 * v.std() / np.sqrt(len(v))
            col = [RARE, INK2, FREQ][k]
            ax.errorbar(k, v.mean(), yerr=ci, fmt="o", color=col, ms=8, capsize=3, mec=SURF, mew=1)
        ax.set_xticks([0, 1, 2], ["rare", "mid", "freq."]); ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"layer {L}", fontsize=9)
    axes[0].set_ylabel("mean gap (H0 mean death)")
    fig.suptitle("Mean gap per frequency third (block + digit matched; mean of 50 sets, 95% interval). Each panel has its own y-axis.",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A3_profiles_by_layer.png"), dpi=160); plt.close(fig)


def fig_special(vec, outdir):
    X, _ = layer(vec, 8)
    F = X[ex.FOUR - 1]; sq = (F ** 2).sum(1)
    nums = np.arange(1100, 1200)
    q = X[nums - 1]; d2 = (q ** 2).sum(1)[:, None] + sq[None, :] - 2 * q @ F.T
    knn = np.sqrt(np.maximum(np.sort(d2, 1)[:, 1:11], 0)).mean(1)
    S = pd.read_csv("results/e10_explore/F_special.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    special = {1111: "1111 (repdigit)", 1100: "1100 (x00)", 1150: "1150 (x50)", 1110: "1110 (two digits only)", 1101: "1101 (two digits only)"}
    ordinary = np.array([n not in special and n % 10 != 0 for n in nums])
    ax.scatter(nums[ordinary], knn[ordinary], s=14, color=NULL, label="ordinary numbers in 1100–1199")
    ax.scatter(nums[nums % 10 == 0], knn[nums % 10 == 0], s=14, color=INK2, marker="s", label="multiples of 10")
    for n, lab in special.items():
        i = n - 1100
        ax.scatter([n], [knn[i]], s=60, color=FREQ, zorder=3, edgecolors=SURF, linewidths=1)
        ax.annotate(lab, (n, knn[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(np.median(knn[ordinary]), color=INK2, lw=1, ls="--")
    ax.set_xlabel("number"); ax.set_ylabel("mean distance to its 10 nearest 4-digit numbers")
    ax.set_title("layer 8, block 1100–1199: how much room each number has", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax = axes[1]
    cats = S[S.layer == 8].sort_values("mean_z", ascending=False)["category"].tolist()
    M = S.pivot(index="category", columns="layer", values="mean_z").loc[cats]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])
    im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=7, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.1f}", ha="center", va="center", fontsize=8, color=SURF if M.values[i, j] > 3.5 else INK)
    med = S[S.layer == 8].set_index("category").loc[cats, "median_count"]
    ax.set_yticks(range(len(cats)), [f"{c}  (count {v / 1e3:,.0f}k)" for c, v in zip(cats, med)], fontsize=8)
    ax.set_xticks(range(5), [f"L{L}" for L in M.columns]); ax.grid(False)
    ax.set_title("isolation z vs ordinary numbers of the same block (> 0 = more room)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A5_special_numbers.png"), dpi=160); plt.close(fig)


def year_feats(X, idx):
    P = X[idx - 1].astype(np.float64); D = squareform(pdist(P)); med = np.median(D[np.triu_indices(len(idx), 1)])
    dg = ripser(D / med, distance_matrix=True, maxdim=1)["dgms"]
    h0 = dg[0][np.isfinite(dg[0][:, 1]), 1]; l1 = dg[1][:, 1] - dg[1][:, 0]
    return dict(spread=med, H0_total=h0.sum() / len(idx), H1_n=len(l1))


def fig_years(vec, outdir):
    X, s = layer(vec, 8)
    yrs = np.arange(1900, 2026); win = np.arange(5900, 6026)
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    Dy, Dw = squareform(pdist(X[yrs - 1])) / s, squareform(pdist(X[win - 1])) / s
    vmax = np.percentile(np.concatenate([Dy.ravel(), Dw.ravel()]), 99)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#1c5cab", "#5598e7", "#cde2fb", "#f4f3ef"])
    for k, (D, nums, title) in enumerate(((Dy, yrs, "years 1900–2025"), (Dw, win, "ordinary 5900–6025"))):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(D, cmap=cmap, vmin=0, vmax=vmax)
        ticks = np.arange(0, len(nums), 20)
        ax.set_xticks(ticks, nums[ticks], fontsize=7); ax.set_yticks(ticks, nums[ticks], fontsize=7); ax.grid(False)
        ax.set_title(f"layer 8 distances, {title}", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="distance (dark = close)")
    # decade clustering: within-decade / between-decade distance ratio
    def ratio(idx):
        D = squareform(pdist(X[idx - 1])); dec = idx // 10; same = dec[:, None] == dec[None, :]; off = ~np.eye(len(idx), dtype=bool)
        return D[same & off].mean() / D[~same].mean()
    starts = [st for st in range(2100, 10000 - 100, 50) if st % 10 == 0]
    rnull = np.array([ratio(np.arange(st, st + 100)) for st in starts]); ry = ratio(np.arange(1900, 2000))
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(rnull, bins=20, color=NULL, edgecolor=SURF, label=f"{len(rnull)} windows of 100 ordinary numbers")
    ax.axvline(ry, color=FREQ, lw=2.5, label=f"years 1900–1999: {ry:.2f}")
    ax.set_xlabel("within-decade ÷ between-decade distance\n(lower = tighter decades)"); ax.set_ylabel("windows")
    ax.set_title("decades cluster", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7.5)
    # barcode summaries vs same-length windows
    starts = list(range(2100, 10000 - 126, 50))
    null = pd.DataFrame([year_feats(X, np.arange(st, st + 126)) for st in starts]); y = year_feats(X, yrs)
    null["spread"] /= s; y["spread"] /= s
    labels = {"spread": "spread (median distance)", "H0_total": "scale-free H0 total (sum of gaps)", "H1_n": "number of H1 loops"}
    for k, (f, lab) in enumerate(labels.items()):
        ax = fig.add_subplot(gs[1, k])
        z = (y[f] - null[f].mean()) / null[f].std()
        ax.hist(null[f], bins=20, color=NULL, edgecolor=SURF, label=f"{len(null)} windows of 126 ordinary numbers")
        ax.axvline(y[f], color=FREQ, lw=2.5, label=f"years 1900–2025 (z = {z:+.1f})")
        ax.set_xlabel(lab); ax.set_ylabel("windows"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A6_years.png"), dpi=160); plt.close(fig)


def fig_barcodes(vec, outdir, cnt, L=8, seed=5):
    """One rare and one frequent set of 120 numbers (2 per block, the same 60 blocks): their H0 and H1 barcodes."""
    mt = ex.matched_tiers(cnt, "block_digits")
    sub = ex.tier_subsample(mt, mt["tier"].values, np.random.default_rng(seed))
    X, s = layer(vec, L)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), gridspec_kw=dict(height_ratios=[1.6, 1]))
    for j, (t, col, lab) in enumerate(((0, RARE, "rare set"), (2, FREQ, "frequent set"))):
        dg = diagrams(X[sub[t] - 1], s)
        h0 = np.sort(dg[0][np.isfinite(dg[0][:, 1]), 1])
        ax = axes[0, j]
        ax.hlines(np.arange(len(h0)), 0, h0, color=col, lw=1.4)
        ax.axvline(h0.mean(), color=INK, lw=1.2, ls="--")
        ax.text(h0.mean(), len(h0) + 2, f" mean H0 death = {h0.mean():.3f}", fontsize=8, va="bottom")
        ax.set_xlim(0, 1.15); ax.set_ylim(-2, len(h0) + 12); ax.set_yticks([])
        ax.set_title(f"layer {L}, {lab}: 119 H0 bars (one per merge), sorted", fontsize=9, loc="left")
        ax.set_xlabel("radius (Euclidean distance ÷ layer's typical distance)")
        h1 = dg[1][np.argsort(dg[1][:, 0])]
        ax = axes[1, j]
        ax.hlines(np.arange(len(h1)), h1[:, 0], h1[:, 1], color=col, lw=1.4)
        ax.set_xlim(0, 1.15); ax.set_yticks([])
        ax.set_title(f"{lab}: {len(h1)} H1 bars (loops), birth → death", fontsize=9, loc="left")
        ax.set_xlabel("radius (Euclidean distance ÷ layer's typical distance)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A0_barcodes.png"), dpi=160); plt.close(fig)


def fig_densities(outdir):
    """Fay-style grid: distribution over 50 sets of four barcode features, rare vs frequent third, per layer."""
    from scipy.stats import gaussian_kde
    A = pd.read_csv("results/e10_explore/A_profiles.csv")
    feats = [("H0_death_mean", "mean H0 death (mean gap)"), ("H1_birth_mean", "mean births of 1-bars"),
             ("H1_life_mean", "mean persistence of 1-bars"), ("H1_count", "number of 1-bars")]
    fig, axes = plt.subplots(len(feats), 5, figsize=(12, 8))
    for i, (f, lab) in enumerate(feats):
        for j, L in enumerate(ex.LAYERS):
            ax = axes[i, j]
            vals = {t: A[(A.layer == L) & (A.group == f"blockdigit_{t}")][f].values for t in ("low", "high")}
            lo = min(v.min() for v in vals.values()); hi = max(v.max() for v in vals.values()); pad = 0.15 * (hi - lo)
            grid = np.linspace(lo - pad, hi + pad, 200)
            for t, col, name in (("low", RARE, "rare third"), ("high", FREQ, "frequent third")):
                y = gaussian_kde(vals[t])(grid)
                ax.fill_between(grid, y, color=col, alpha=0.35, lw=0); ax.plot(grid, y, color=col, lw=1.5, label=name)
            ax.set_yticks([]); ax.tick_params(labelsize=7)
            if i == 0: ax.set_title(f"layer {L}", fontsize=10)
            if j == 0: ax.set_ylabel(lab, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Each curve = spread of a barcode feature over 50 sets of 120 numbers (block + digit matched)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_A4_feature_densities.png"), dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", default="results/e10_explore/figures")
    ap.add_argument("--only", default="profiles,special,years,z,barcodes,densities")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _style(plt)
    cnt = ex.load_counts()
    only = a.only.split(",")
    if "barcodes" in only: fig_barcodes(a.vectors, a.out, cnt)
    if "densities" in only: fig_densities(a.out)
    if "profiles" in only: fig_profiles(a.out)
    if "special" in only: fig_special(a.vectors, a.out)
    if "years" in only: fig_years(a.vectors, a.out)
    if "z" in only: print("z (re-drawn null):", fig_z(a.vectors, a.out, cnt), flush=True)


if __name__ == "__main__":
    main()
