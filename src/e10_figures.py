"""E10 figures that need the vectors themselves (exploratory, descriptive):

  fig_betti_curves.png     Betti-0 / Betti-1 curves of the rarest vs most frequent tenth (one
                           depth per model), mean over sets
  fig_diagrams.png         H1 persistence diagram, rarest vs most frequent tenth (Crystal, ~25 %)
  fig_feature_map.png      41-feature vectors of every set, PCA to 2-D, coloured by frequency level
  fig_depth_profile.png    H1 loop count and H0 spacing across depth, all 7 models
  fig_rtd_heatmap.png      model × model RTD / shuffled RTD at each depth (from e10_convergence)

  python -m src.e10_figures --data <topology_number_line>/data --out results/e10_graded/figures
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import src.e10_explore as ex
from src.e10_graded import DEPTH, MODELS, layer_files, levels, subsample
from src.e10_graded_analysis import DEPTH_COLORS, GRID, INK, INK2, SURF, VARYING, _style
from src.topo_features import barcode_features, diagrams


def betti_curve(dgm, grid):
    if len(dgm) == 0:
        return np.zeros_like(grid)
    b, d = dgm[:, 0][:, None], np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], np.inf)[:, None]
    return ((b <= grid[None, :]) & (d > grid[None, :])).sum(0)


def load_layer(data, m, d):
    X = np.load(layer_files(data, m)[d])["X"].astype(np.float32)
    rs = np.random.default_rng(0).choice(ex.FOUR, 1000, replace=False)
    return X, float(np.median(pdist(X[rs - 1])))


def fig_betti(data, path, K=15, seed=21):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    cnt = ex.load_counts(); mt = levels(cnt, 10); rng = np.random.default_rng(seed)
    subs = [subsample(mt, mt["level"].values, 10, rng) for _ in range(K)]
    grid = np.linspace(0, 1.4, 200)
    fig, axes = plt.subplots(2, 7, figsize=(18, 5.2), sharex=True)
    for j, m in enumerate(MODELS):
        X, s = load_layer(data, m, 0)
        for q, col, lab in ((0, "#86b6ef", "rarest tenth"), (9, "#104281", "most frequent tenth")):
            b0 = np.mean([betti_curve(diagrams(X[sb[q] - 1], s)[0], grid) for sb in subs], 0)
            b1 = np.mean([betti_curve(diagrams(X[sb[q] - 1], s)[1], grid) for sb in subs], 0)
            axes[0, j].plot(grid, b0, color=col, label=lab); axes[1, j].plot(grid, b1, color=col, label=lab)
        axes[0, j].set_title(f"{m} (~25 % depth)", fontsize=9)
    axes[0, 0].set_ylabel("Betti-0 (components)"); axes[1, 0].set_ylabel("Betti-1 (loops)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("filtration radius (in units of the layer's typical 4-digit distance)")
    fig.suptitle("Betti curves of 120 equally large numbers: rarest vs most frequent tenth (mean of 15 sets)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_diagrams(data, path, seed=22):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    cnt = ex.load_counts(); mt = levels(cnt, 10); rng = np.random.default_rng(seed)
    X, s = load_layer(data, "crystal", 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for k in range(5):
        sb = subsample(mt, mt["level"].values, 10, rng)
        for q, col, mk, lab in ((0, "#86b6ef", "o", "rarest tenth"), (9, "#104281", "s", "most frequent tenth")):
            dg = diagrams(X[sb[q] - 1], s)
            h0 = dg[0][np.isfinite(dg[0][:, 1]), 1]
            axes[0].hist(h0, bins=np.linspace(0.3, 1.3, 40), histtype="step", color=col, lw=1.5, label=lab if k == 0 else None)
            axes[1].scatter(dg[1][:, 0], dg[1][:, 1], s=10, color=col, marker=mk, alpha=0.6, label=lab if k == 0 else None,
                            edgecolors=SURF, linewidths=0.4)
    lim = axes[1].get_xlim()[1]; axes[1].plot([0, 1.3], [0, 1.3], color=INK2, lw=0.8)
    axes[0].set_xlabel("H0 death (when a point joins the rest)"); axes[0].set_ylabel("count (5 sets)")
    axes[1].set_xlabel("H1 birth"); axes[1].set_ylabel("H1 death")
    axes[0].legend(frameon=False); axes[1].legend(frameon=False)
    fig.suptitle("Crystal, ~25 % depth: H0 deaths and H1 diagram, rarest vs most frequent tenth (5 sets of 120)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_feature_map(features_csv, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _style(plt)
    df = pd.read_csv(features_csv); df = df[(df.Q == 10) & (~df.shuffled) & (df.depth == "~25%")]
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, m in zip(axes.flat, MODELS):
        d = df[df.model == m]
        Z = PCA(2).fit_transform(StandardScaler().fit_transform(d[VARYING]))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=d.level + 1, cmap=cmap, s=12, edgecolors=SURF, linewidths=0.3)
        ax.set_title(m, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axes.flat[-1].axis("off")
    cb = fig.colorbar(sc, ax=axes.flat[-1], fraction=0.5); cb.set_label("frequency level (1 = rarest tenth)")
    fig.suptitle("Topology fingerprints: 41 barcode features of every set of 120 numbers, PCA to 2-D (~25 % depth)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_depth(data, path, K=12, m_pts=300, seed=23):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    rng = np.random.default_rng(seed); pool = ex.FOUR[~ex.year_zone(ex.FOUR)]
    rows = []
    for m in MODELS:
        for d in range(5):
            X, _ = load_layer(data, m, d)
            for _ in range(K):
                P = X[rng.choice(pool, m_pts, replace=False) - 1]
                f = barcode_features(diagrams(P, float(np.median(pdist(P)))))
                rows.append(dict(model=m, depth=d, H1_count=f["H1_count"], H0_death_mean=f["H0_death_mean"]))
    t = pd.DataFrame(rows).groupby(["model", "depth"]).mean().reset_index()
    t.to_csv(path.replace(".png", ".csv"), index=False)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for m, c in zip(MODELS, cols):
        r = t[t.model == m]
        axes[0].plot(r.depth, r.H1_count, color=c, marker="o", ms=4, label=m)
        axes[1].plot(r.depth, r.H0_death_mean, color=c, marker="o", ms=4, label=m)
    for ax, lab in zip(axes, ("H1 loop count (300 numbers)", "H0 mean death (scale-free)")):
        ax.set_xticks(range(5)); ax.set_xticklabels(DEPTH); ax.set_ylabel(lab); ax.set_xlabel("layer depth")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Number topology simplifies with depth in all 7 models: fewer loops, tighter clusters")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_rtd(conv_csv, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    df = pd.read_csv(conv_csv)
    g = df.groupby(["a", "b", "depth"])[["rtd", "rtd_shuffled"]].mean(); g["r"] = g.rtd / g.rtd_shuffled; g = g.reset_index()
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    for ax, dep in zip(axes, DEPTH):
        M = np.full((7, 7), np.nan)
        for _, r in g[g.depth == dep].iterrows():
            i, j = MODELS.index(r.a), MODELS.index(r.b); M[i, j] = M[j, i] = r.r
        im = ax.imshow(M, cmap="Blues_r", vmin=0.2, vmax=1.0)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(MODELS, rotation=90, fontsize=7); ax.set_yticklabels(MODELS if dep == DEPTH[0] else [], fontsize=7)
        ax.set_title(dep); ax.grid(False)
        for i in range(7):
            for j in range(7):
                if i != j and np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK if M[i, j] > 0.55 else SURF)
    cb = fig.colorbar(im, ax=axes, shrink=0.8); cb.set_label("RTD / shuffled RTD (lower = same topology)")
    fig.suptitle("Do models give the same numbers the same topology? (darker = more similar)")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e10_graded/figures")
    ap.add_argument("--only", default="betti,diagrams,map,depth,rtd")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if "betti" in a.only: fig_betti(a.data, os.path.join(a.out, "fig_betti_curves.png")); print("betti done", flush=True)
    if "diagrams" in a.only: fig_diagrams(a.data, os.path.join(a.out, "fig_diagrams.png")); print("diagrams done", flush=True)
    if "map" in a.only: fig_feature_map("results/e10_graded/features.csv.gz", os.path.join(a.out, "fig_feature_map.png")); print("map done", flush=True)
    if "depth" in a.only: fig_depth(a.data, os.path.join(a.out, "fig_depth_profile.png")); print("depth done", flush=True)
    if "rtd" in a.only: fig_rtd("results/e10_convergence/rtd_pairs.csv", os.path.join(a.out, "fig_rtd_heatmap.png")); print("rtd done", flush=True)


if __name__ == "__main__":
    main()
