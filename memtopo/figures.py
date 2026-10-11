"""Figures 0-5 of the brief (section 9). One y-axis per panel, small multiples, CIs as error
bars or bands, palette in the brief's order (validated: CVD-safe; the green and amber are below
3:1 contrast on white, so every series also carries a legend entry or a direct label)."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from memtopo import stats  # noqa: E402
from memtopo.analysis import RUNG_LABEL  # noqa: E402
from memtopo.config import paths, read_gates  # noqa: E402

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def _style(ax, title=None):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def _runs(root: Path):
    for f in sorted(glob.glob(str(root / "results" / "*" / "S*" / "ladder.csv"))):
        f = Path(f)
        yield f.parent.parent.name, int(f.parent.name[1:]), f.parent


def fig1_ladder(cfg):
    p = paths(cfg)
    runs = list(_runs(p.root))
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), squeeze=False, sharey=True)
    for ax, (model, sf, d) in zip(axes[0], runs):
        lad = pd.read_csv(d / "ladder.csv", dtype={"rung": str})
        lad = lad[(lad["subset"] == "hard") & (lad["clf"] == "lr")]
        order = list(RUNG_LABEL)
        for k, seed in enumerate(sorted(lad["seed"].unique())):
            s = lad[lad["seed"] == seed].set_index("rung").reindex(order).dropna(subset=["AUROC"])
            x = np.array([order.index(r) for r in s.index]) + (k - 0.5) * 0.25
            ax.errorbar(x, s["AUROC"], yerr=[s["AUROC"] - s["AUROC_lo"], s["AUROC_hi"] - s["AUROC"]],
                        fmt="o", ms=5, lw=1.2, color=PAL[k], label=f"negatives seed {seed}")
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(len(order)), [RUNG_LABEL[r] for r in order], rotation=60, ha="right", fontsize=7)
        _style(ax, f"pythia-{model}-deduped, S_f = {sf}")
    axes[0][0].set_ylabel("AUROC, hard subset, out-of-time test", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(p.figures / "fig1_ladder.png", dpi=160)
    plt.close(fig)


def fig2_pr(cfg):
    p = paths(cfg)
    runs = list(_runs(p.root))
    if not runs:
        return
    g = read_gates(p)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4), squeeze=False)
    for ax, (model, sf, d) in zip(axes[0], runs):
        pr = pd.read_parquet(d / "preds.parquet")
        pr = pr[(pr["subset"] == "full") & (pr["clf"] == "lr") & (pr["seed"] == 0)]
        for k, rung in enumerate(["5", "6", "7"]):
            s = pr[pr["rung"] == rung]
            if s.empty:
                continue
            prec, rec = stats.pr_curve(s["label"].to_numpy(), s["score"].to_numpy(), s["weight"].to_numpy())
            ax.plot(rec, prec, color=PAL[k], lw=2, label=RUNG_LABEL[rung])
        b0 = g.get(f"labels_{model}", {}).get("per_Sf", {}).get(str(sf), {}).get("B0_vs_mem_T")
        if b0:
            ax.plot([b0["recall"]], [b0["precision"]], "D", ms=8, color=INK, label="B0 (Biderman: memorized at S_f)")
        ax.set_xlabel("recall", fontsize=9)
        ax.set_ylim(0, 1.02)
        _style(ax, f"{model}, S_f = {sf} (natural prevalence)")
    axes[0][0].set_ylabel("precision", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(p.figures / "fig2_pr_curves.png", dpi=160)
    plt.close(fig)


def fig3_residual(cfg):
    p = paths(cfg)
    runs = [(m, s, d) for m, s, d in _runs(p.root) if (d / "residual.csv").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.6), squeeze=False, sharey=True)
    for ax, (model, sf, d) in zip(axes[0], runs):
        r = pd.read_csv(d / "residual.csv")
        r = r[(r["clf"] == "lr") & (r["seed"] == 0)]
        ax.fill_between(r["decile"] + 1, r["lo"], r["hi"], color=PAL[0], alpha=0.2, lw=0)
        ax.plot(r["decile"] + 1, r["AUROC"], "o-", color=PAL[0], lw=2, ms=5)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.text(10.2, 0.5, "chance", fontsize=7, color=MUTED, va="center")
        ax.set_xlabel("decile of continuation cross-entropy at S_f", fontsize=9)
        _style(ax, f"{model}, S_f = {sf}: topology-only AUROC within loss decile")
    axes[0][0].set_ylabel("AUROC (hard subset, test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(p.figures / "fig3_residual.png", dpi=160)
    plt.close(fig)


def fig4_effects(cfg):
    p = paths(cfg)
    runs = [(m, s, d) for m, s, d in _runs(p.root) if (d / "effects.csv").exists()]
    if not runs:
        return
    feats = ["h1_n_bars", "h1_pers_mean", "h1_entropy", "h0_death_mean"]
    fig, axes = plt.subplots(len(runs), len(feats), figsize=(3.2 * len(feats), 2.8 * len(runs)),
                             squeeze=False, sharey=True)
    for i, (model, sf, d) in enumerate(runs):
        e = pd.read_csv(d / "effects.csv")
        for j, f in enumerate(feats):
            ax = axes[i][j]
            s = e[e["feature"] == f].sort_values("layer")
            if len(s):
                ax.errorbar(s["layer"], s["hedges_g"], yerr=[s["hedges_g"] - s["lo"], s["hi"] - s["hedges_g"]],
                            fmt="o", color=PAL[0], ms=5, lw=1.2)
            else:
                ax.text(0.5, 0.5, "pruned (|r| > 0.5)", transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
            ax.axhline(0, color=MUTED, lw=0.8)
            sign = {"h1_n_bars": "-", "h1_pers_mean": "+", "h1_entropy": "-", "h0_death_mean": "+"}[f]
            _style(ax, f"{f} (H2 expects {sign})" + (f"\n{model}, S_f={sf}" if j == 0 else ""))
            ax.set_xlabel("layer", fontsize=8)
        axes[i][0].set_ylabel("Hedges' g (future-memorized − not)", fontsize=8)
    fig.tight_layout()
    fig.savefig(p.figures / "fig4_effects.png", dpi=160)
    plt.close(fig)


def fig5_velocity(cfg):
    p = paths(cfg)
    runs = list(_runs(p.root))
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 2, figsize=(8, 3 * len(runs)), squeeze=False)
    for i, (model, sf, _) in enumerate(runs):
        f = p.features(model, sf)
        t2 = pd.read_parquet(f / "topo_T2.parquet").set_index("index")
        lab = pd.read_parquet(f / "loss.parquet").set_index("index")
        cc = pd.read_parquet(p.data / f"cc_{model}.parquet")
        s = cc[(cc["target"] == "mem_T") & (cc["seed"] == 0)].set_index("index")
        s = s[s.index.isin(t2.index)]
        s = s[lab.loc[s.index, "mem_Sf"].to_numpy() == 0]
        for j, kind in enumerate(["prepost", "postsf"]):
            ax = axes[i][j]
            v = t2.loc[s.index, [c for c in t2.columns if c.startswith(f"v_{kind}_h1")]].mean(1)
            bins = np.linspace(np.nanpercentile(v, 1), np.nanpercentile(v, 99), 40)
            for k, (lbl, name) in enumerate([(1, "future-memorized"), (0, "not memorized")]):
                ax.hist(v[s["label"] == lbl], bins=bins, density=True, histtype="step", lw=2, color=PAL[k], label=name)
            _style(ax, f"{model}, S_f={sf}: H1 velocity {'pre→post exposure' if kind == 'prepost' else 'post→S_f'}")
            ax.set_xlabel("mean over layers of W2(H1 diagrams)", fontsize=8)
        axes[i][0].set_ylabel("density (hard subset)", fontsize=8)
        axes[i][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(p.figures / "fig5_velocity.png", dpi=160)
    plt.close(fig)


def fig0_checks(cfg):
    p = paths(cfg)
    g = read_gates(p)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    al = [(k.split("_", 1)[1], v) for k, v in g.items() if k.startswith("alignment_")]
    ax = axes[0]
    if al:
        names = [a for a, _ in al]
        ax.bar(names, [v["n_match"] for _, v in al], color=PAL[0], width=0.5)
        ax.bar(names, [v["n_checked"] - v["n_match"] for _, v in al], bottom=[v["n_match"] for _, v in al],
               color=PAL[1], width=0.5)
        for x, (_, v) in enumerate(al):
            ax.text(x, v["n_checked"], f"{v['n_match']}/{v['n_checked']}", ha="center", va="bottom", fontsize=8, color=INK)
    _style(ax, "alignment: shard tokens == memorized-evals tokens")
    ax = axes[1]
    gfs = sorted(glob.glob(str(p.root / "results" / "*" / "S*" / "gaussian_floor.csv")))
    if gfs:
        d = pd.read_csv(gfs[0])
        d = d[d["feature"] == "h1_n_bars"]
        ax.plot(d["layer"], d["real_mean"], "o-", color=PAL[0], lw=2, label="real clouds (mean)")
        ax.plot(d["layer"], d["gauss_mean"], "s--", color=PAL[1], lw=2, label="matched Gaussian (mean)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("layer", fontsize=8)
        _style(ax, f"H1 bar count vs Gaussian floor ({Path(gfs[0]).parent.parent.name}, {Path(gfs[0]).parent.name})")
    ax = axes[2]
    runs = list(_runs(p.root))
    if runs:
        lad = pd.read_csv(runs[0][2] / "ladder.csv", dtype={"rung": str})
        lad = lad[(lad["subset"] == "hard") & (lad["clf"] == "lr") & (lad["seed"] == 0)]
        order = [r for r in RUNG_LABEL if r in set(lad["rung"])]
        s = lad.set_index("rung").loc[order]
        ax.errorbar(range(len(order)), s["shuffled_AUROC"],
                    yerr=[s["shuffled_AUROC"] - s["shuffled_lo"], s["shuffled_hi"] - s["shuffled_AUROC"]],
                    fmt="o", color=PAL[0], ms=5)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(len(order)), [RUNG_LABEL[r] for r in order], rotation=60, ha="right", fontsize=7)
        _style(ax, f"shuffled-label AUROC ({runs[0][0]}, S_f={runs[0][1]})")
    fig.tight_layout()
    fig.savefig(p.figures / "fig0_checks.png", dpi=160)
    plt.close(fig)


def all_figures(cfg):
    for f in (fig0_checks, fig1_ladder, fig2_pr, fig3_residual, fig4_effects, fig5_velocity):
        f(cfg)
