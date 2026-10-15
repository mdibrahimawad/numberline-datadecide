"""E10 stage E analysis: dose-response of the 41 barcode features over 3/5/10 frequency levels,
shape of the response (smooth vs threshold), classifiers with held-out blocks, linear SHAP, and
figures. Input: results/e10_graded/{features,clf_features}.csv.gz (src/e10_graded.py).

  python -m src.e10_graded_analysis --out results/e10_graded
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.topo_features import FEATURE_NAMES

MODELS = ["crystal", "openllama-3b", "starcoderbase-3b", "qwen2.5-1.5b", "smollm2-1.7b", "pythia-1.4b", "olmo2-1b"]
DEPTH = ["~25%", "~50%", "~75%", "~84%", "last"]
FAMILY = {  # groups of features that measure the same thing
    "H0 spacing (deaths)": [f for f in FEATURE_NAMES if f.startswith("H0_death") or f == "H0_total"],
    "H0 entropy": ["H0_entropy"],
    "H1 loop count": ["H1_count"],
    "H1 birth/death scale": [f for f in FEATURE_NAMES if f.startswith(("H1_birth", "H1_death", "H1_mid"))],
    "H1 lifetimes": [f for f in FEATURE_NAMES if f.startswith("H1_life")] + ["H1_total", "H1_entropy"],
    "H1 birth/death ratio": [f for f in FEATURE_NAMES if f.startswith("H1_ratio")],
}
VARYING = [f for f in FEATURE_NAMES if f not in ("H0_count", "H0_death_min")]   # constant / degenerate


# --------------------------------------------------------------------------- #
# 1. dose-response: per subsample, slope of feature over level; real vs shuffled levels
# --------------------------------------------------------------------------- #

def slopes(df):
    """Per (Q, shuffled, rep, model, depth, feature): Spearman of level vs feature over the Q
    levels of one subsample (all levels share the same blocks)."""
    rows = []
    for key, g in df.groupby(["Q", "shuffled", "rep", "model", "depth"]):
        g = g.sort_values("level")
        for f in VARYING:
            v = g[f].values
            rho = spearmanr(g["level"], v).correlation if np.ptp(v) > 0 else 0.0
            rows.append(dict(zip(["Q", "shuffled", "rep", "model", "depth"], key), feature=f, rho=rho))
    return pd.DataFrame(rows)


def dose_response(sl):
    """z of mean per-subsample rho, real vs shuffled levels (Welch)."""
    g = sl.groupby(["Q", "model", "depth", "feature", "shuffled"])["rho"].agg(["mean", "var", "count"]).unstack("shuffled")
    z = (g[("mean", False)] - g[("mean", True)]) / np.sqrt(g[("var", False)] / g[("count", False)] + g[("var", True)] / g[("count", True)])
    out = pd.DataFrame({"rho_real": g[("mean", False)], "rho_shuffled": g[("mean", True)], "z": z}).reset_index()
    return out


# --------------------------------------------------------------------------- #
# 2. shape of the response over 10 levels: linear vs one step
# --------------------------------------------------------------------------- #

def shape_fit(df, feature="H0_death_mean"):
    rows = []
    d10 = df[df.Q == 10]
    for (m, dep), g in d10.groupby(["model", "depth"]):
        real = g[~g.shuffled].groupby("level")[feature].mean()
        sh = g[g.shuffled]
        y = (real - sh[feature].mean()) / sh.groupby("rep")[feature].mean().std()   # in null-sd units
        x = np.arange(10)
        lin = np.polyfit(x, y.values, 1); r_lin = y.values - np.polyval(lin, x)
        best = None
        for b in range(1, 10):
            fit = np.where(x < b, y.values[:b].mean(), y.values[b:].mean()); r = y.values - fit
            if best is None or (r ** 2).sum() < best[1]:
                best = (b, (r ** 2).sum())
        tss = ((y.values - y.values.mean()) ** 2).sum()
        rows.append(dict(model=m, depth=dep, slope_per_level=lin[0], r2_linear=1 - (r_lin ** 2).sum() / tss,
                         r2_step=1 - best[1] / tss, step_after_level=best[0], spearman=spearmanr(x, y.values).correlation,
                         **{f"L{i + 1}": y.values[i] for i in range(10)}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. classifiers (held-out halves of the blocks) and linear SHAP
# --------------------------------------------------------------------------- #

def classify(cf, n_null=20, seed=0):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    rows, shap_rows = [], []
    for (m, dep), g in cf.groupby(["model", "depth"]):
        for Q in (3, 10):
            d = g[g.Q == Q]
            res = {"model": m, "depth": dep, "Q": Q}
            accs, nulls, rhos, aucs, null_aucs = [], [], [], [], []
            for tr, te in ((0, 1), (1, 0)):
                A, B = d[d.half == tr], d[d.half == te]
                sc = StandardScaler().fit(A[VARYING]); XA, XB = sc.transform(A[VARYING]), sc.transform(B[VARYING])
                if Q == 3:
                    clf = LogisticRegression(max_iter=3000, C=0.1).fit(XA, A.level)
                    accs.append((clf.predict(XB) == B.level).mean())
                    nulls += [(LogisticRegression(max_iter=3000, C=0.1).fit(XA, rng.permutation(A.level.values)).predict(XB) == B.level).mean() for _ in range(n_null)]
                    # binary low vs high + linear SHAP (exact for a linear model, features taken independent)
                    a2, b2 = A.level.isin([0, 2]).values, B.level.isin([0, 2]).values
                    lb = LogisticRegression(max_iter=3000, C=0.1).fit(XA[a2], (A.level.values[a2] == 2))
                    aucs.append(roc_auc_score(B.level.values[b2] == 2, lb.decision_function(XB[b2])))
                    null_aucs += [roc_auc_score(B.level.values[b2] == 2, LogisticRegression(max_iter=3000, C=0.1).fit(
                        XA[a2], rng.permutation(A.level.values[a2] == 2)).decision_function(XB[b2])) for _ in range(n_null)]
                    contrib = np.abs(lb.coef_[0][None, :] * XB[b2]).mean(0)
                    for f, c, w in zip(VARYING, contrib, lb.coef_[0]):
                        shap_rows.append(dict(model=m, depth=dep, fold=tr, feature=f, mean_abs_shap=c, coef=w))
                else:
                    reg = Ridge(alpha=10.0).fit(XA, A.level)
                    rhos.append(spearmanr(reg.predict(XB), B.level).correlation)
                    nulls += [spearmanr(Ridge(alpha=10.0).fit(XA, rng.permutation(A.level.values)).predict(XB), B.level).correlation for _ in range(n_null)]
            if Q == 3:
                res.update(acc_3way=np.mean(accs), acc_null_mean=np.mean(nulls), acc_null_95=np.quantile(nulls, 0.95),
                           auc_low_high=np.mean(aucs), auc_null_95=np.quantile(null_aucs, 0.95))
            else:
                res.update(ordinal_spearman=np.mean(rhos), ordinal_null_95=np.quantile(nulls, 0.95))
            rows.append(res)
    out = pd.DataFrame(rows)
    return out.groupby(["model", "depth"]).first().reset_index(), pd.DataFrame(shap_rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DEPTH_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]     # ordinal, validated


def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2})


def fig_dose(shape, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MODELS):
        s = shape[shape.model == m]
        ax.axhspan(-2, 2, color="#f0efec", zorder=0)
        for dep, col in zip(DEPTH, DEPTH_COLORS):
            r = s[s.depth == dep]
            if len(r):
                ax.plot(range(1, 11), [r[f"L{i}"].values[0] for i in range(1, 11)], color=col, marker="o", ms=3, label=dep)
        ax.set_title(m, fontsize=10); ax.axhline(0, color=INK2, lw=0.8)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, title="layer depth", loc="center", frameon=False)
    fig.supxlabel("frequency level (1 = rarest tenth, 10 = most frequent tenth; same magnitude and digits)")
    fig.supylabel("H0 mean death (spacing), in null-sd units from shuffled levels")
    fig.suptitle("More frequent numbers sit further apart: H0 spacing over 10 frequency levels (grey band = ±2 null sd)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_heat(dr, path, Q=10):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style(plt)
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#c23a39"])
    d = dr[dr.Q == Q].copy(); d["col"] = d.model + " " + d.depth
    piv = d.pivot_table(index="feature", columns="col", values="z")
    piv = piv[[f"{m} {dep}" for m in MODELS for dep in DEPTH if f"{m} {dep}" in piv.columns]]
    piv = piv.loc[[f for f in VARYING if f in piv.index]]
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(piv.clip(-8, 8).values, cmap=cmap, vmin=-8, vmax=8, aspect="auto")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90, fontsize=7)
    ax.grid(False)
    for i in range(1, len(MODELS)):
        ax.axvline(i * 5 - 0.5, color=SURF, lw=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("z: trend with frequency level vs shuffled (red = rises with frequency)")
    ax.set_title(f"All 41 barcode features × 7 models × 5 depths ({Q} frequency levels)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_shap(shap, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _style(plt)
    fam = {f: k for k, v in FAMILY.items() for f in v}
    s = shap.copy(); s["family"] = s.feature.map(fam)
    tot = s.groupby(["model", "depth", "fold"])["mean_abs_shap"].transform("sum")
    s["share"] = s["mean_abs_shap"] / tot
    t = s.groupby(["model", "family"])["share"].sum().unstack() / (5 * 2)
    t = t[list(FAMILY)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    left = np.zeros(len(t))
    for c, fname in zip(cols, t.columns):
        ax.barh(t.index, t[fname], left=left, color=c, label=fname, edgecolor=SURF, linewidth=2)
        left += t[fname].values
    ax.set_xlabel("share of the classifier's |SHAP| (low vs high frequency third; mean over depths and folds)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("Which topological features tell frequent from rare numbers"); ax.grid(axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e10_graded")
    a = ap.parse_args()
    figs = os.path.join(a.out, "figures"); os.makedirs(figs, exist_ok=True)
    df = pd.read_csv(os.path.join(a.out, "features.csv.gz"))
    sl = slopes(df)
    dr = dose_response(sl); dr.to_csv(os.path.join(a.out, "dose_response.csv"), index=False)
    sh = shape_fit(df); sh.to_csv(os.path.join(a.out, "shape_H0_death_mean.csv"), index=False)
    fig_dose(sh, os.path.join(figs, "fig_dose_response_H0.png"))
    for Q in (3, 5, 10):
        fig_heat(dr, os.path.join(figs, f"fig_heatmap_Q{Q}.png"), Q)
    summary = {}
    for Q in (3, 5, 10):
        d = dr[dr.Q == Q]
        summary[f"Q{Q}_cells_abs_z_gt_3"] = int((d.z.abs() > 3).sum()); summary[f"Q{Q}_cells"] = len(d)
    if os.path.exists(os.path.join(a.out, "clf_features.csv.gz")):
        cf = pd.read_csv(os.path.join(a.out, "clf_features.csv.gz"))
        res, shap = classify(cf)
        res.to_csv(os.path.join(a.out, "classifiers.csv"), index=False); shap.to_csv(os.path.join(a.out, "shap.csv"), index=False)
        fig_shap(shap, os.path.join(figs, "fig_shap_families.png"))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
