"""Figures (brief section 9): log-x steps with step 0 at an offset labelled "init", one y-axis
per panel, grey null band (mean +- 2 sd), direct labels, colour-blind-safe palette.

    python -m tt.figures --model pythia-1.4b-deduped --input P1
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

PAL = C.PALETTE
INIT_X = 100          # where step 0 is drawn on the log axis
GREY = "#9a9a9a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#eee", "grid.linewidth": .6})


def _x(steps):
    s = np.asarray(steps, float)
    return np.where(s == 0, INIT_X, s)


def _logx(ax, steps):
    ax.set_xscale("log")
    ticks = [INIT_X, 1e3, 1e4, 1e5]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["init", "1k", "10k", "100k"])
    ax.set_xlim(INIT_X * .7, max(steps) * 1.4)


def _line(ax, d, color, label, null=True):
    d = d.sort_values("step")
    x = _x(d.step)
    if null and "null_mean" in d and d.null_mean.notna().any():
        ax.fill_between(x, d.null_mean - 2 * d.null_sd, d.null_mean + 2 * d.null_sd, color=GREY, alpha=.25, lw=0)
    ax.plot(x, d.value, color=color, lw=2, marker="o", ms=4)
    ax.annotate(label, (x.iloc[-1] if hasattr(x, "iloc") else x[-1], d.value.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", color="#333", va="center", fontsize=8)


def _sel(m, metric, layer, seed=None):
    d = m[(m.metric == metric) & (m.layer == layer)]
    if seed is not None:
        d = d[d.seed == seed]
    return d


def fig1(m, vel, ref, model, out):
    panels = [("B1_loglik_auc", -1, None, "B1  log-likelihood AUC (behaviour)"),
              ("L1_auc", ref, None, "L1  logistic probe AUC"),
              ("T1_gap", ref, 0, "T1 gap  shape accuracy − mixed-vs-mixed"),
              ("T3b_within_truth_homophily", ref, None, "T3b  within-topic truth homophily"),
              (None, ref, 0, "T2  H1 velocity (normalised; mixed clouds)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.2, 2.0 * len(panels)), sharex=True)
    steps = sorted(m.step.unique())
    for ax, (met, lay, seed, title) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=9)
        if met is None:
            v = vel[(vel.layer == lay) & (vel.seed == 0) & (vel.kind == "mixed") & (vel.hom == 1) & (vel.prep == "norm")]
            ax.plot(np.sqrt(_x(v.step_from) * _x(v.step_to)), v.velocity, color=PAL[0], lw=2, marker="o", ms=4)
            ax.set_ylabel("W2 per transition")
        else:
            d = _sel(m, met, lay, seed)
            if d.empty:
                ax.text(.5, .5, "not available", transform=ax.transAxes, ha="center", color=GREY)
                continue
            _line(ax, d, PAL[0], "")
        _logx(ax, steps)
    axes[-1].set_xlabel("training step (log)")
    fig.suptitle(f"{model} — reference layer {ref}", x=.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2(m, vel, model, out):
    items = [("L1_auc", "value", None, "L1 AUC"), ("L3_auc", "value", None, "L3 cross-topic AUC"),
             ("T1_gap", "value", 0, "T1 gap"), ("T3b_within_truth_homophily", "z", None, "T3b z"),
             ("T3c_cross_truth_homophily", "z", None, "T3c z"), ("velocity", None, 0, "T2 H1 velocity")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (met, col, seed, title) in zip(axes.flat, items):
        if met == "velocity":
            v = vel[(vel.seed == 0) & (vel.kind == "mixed") & (vel.hom == 1) & (vel.prep == "norm")]
            piv = v.pivot(index="layer", columns="step_to", values="velocity")
            xl = [f"→{int(s/1000)}k" if s >= 1000 else f"→{s}" for s in piv.columns]
        else:
            d = m[(m.metric == met) & (m.layer >= 0)]
            if seed is not None:
                d = d[d.seed == seed]
            if d.empty:
                ax.set_axis_off(); continue
            piv = d.pivot_table(index="layer", columns="step", values=col)
            xl = ["init" if s == 0 else (f"{int(s/1000)}k" if s >= 1000 else str(s)) for s in piv.columns]
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=90, fontsize=7)
        ax.set_ylabel("layer"); ax.set_title(title, loc="left", fontsize=9); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    fig.suptitle(f"{model}: layer × checkpoint", x=.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3(m, ref, model, out):
    panels = [("T3a_topic_homophily", "z", "T3a topic homophily (z)"),
              ("T3b_within_truth_homophily", "z", "T3b within-topic truth homophily (z)"),
              ("T3c_cross_truth_homophily", "z", "T3c cross-topic truth homophily (z)"),
              ("L3_auc", "value", "L3 linear cross-topic AUC")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.2, 2.0 * len(panels)), sharex=True)
    steps = sorted(m.step.unique())
    for ax, (met, col, title) in zip(axes, panels):
        d = _sel(m, met, ref).sort_values("step")
        ax.set_title(title, loc="left", fontsize=9)
        if col == "z":
            ax.axhspan(-2, 2, color=GREY, alpha=.25, lw=0)
            ax.plot(_x(d.step), d.z, color=PAL[0], lw=2, marker="o", ms=4)
        else:
            _line(ax, d, PAL[0], "")
        _logx(ax, steps)
    axes[-1].set_xlabel("training step (log)")
    fig.suptitle(f"{model}: global vs local truth, layer {ref}", x=.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


RAVFOGEL = pd.DataFrame({  # Pythia-6.9B, App. E.4 values quoted in the brief
    "step": [0, 1000, 3000, 10000, 20000, 60000, 143000],
    "probe_auc": [.383, .467, .587, .667, .754, .802, .831],
})
RAVFOGEL_MEM = pd.DataFrame({"step": [0, 3000, 10000, 143000], "mem": [0, .242, .547, .875]})
RAVFOGEL_DH = pd.DataFrame({"step": [0, 1000, 3000, 143000], "dh": [0, 0, .219, .518]})


def fig4(runs: dict, out):
    """runs: label -> (metrics dataframe, reference layer) for P2 runs."""
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.5), sharex=True)
    specs = [("L1_auc", "probe AUC (context truth)", RAVFOGEL.rename(columns={"probe_auc": "v"})),
             ("B2_memorization_truectx", "memorization (top-1)", RAVFOGEL_MEM.rename(columns={"mem": "v"})),
             ("B2_deltaH", "ΔH = H(false ctx) − H(true ctx), nats", RAVFOGEL_DH.rename(columns={"dh": "v"}))]
    for ax, (met, title, ref_df) in zip(axes, specs):
        ax.set_title(title, loc="left", fontsize=9)
        ax.plot(_x(ref_df.step), ref_df.v, color=GREY, lw=2, ls="--", marker="s", ms=4)
        ax.annotate("Ravfogel et al. 6.9B", (_x(ref_df.step)[-1], ref_df.v.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color="#333", va="center")
        for i, (lab, (m, ref)) in enumerate(runs.items()):
            lay = -1 if met.startswith("B2") else ref
            d = _sel(m, met, lay).sort_values("step")
            if not d.empty:
                _line(ax, d, PAL[i % 4], lab, null=False)
        _logx(ax, [143000])
    axes[-1].set_xlabel("training step (log)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig0(report, summaries):
    """Validation: per synthetic scenario, the key measures; plus the abrupt-change velocity curve."""
    tests = report["tests"]
    names = [n for n in tests if n[0] in "1234"]
    cols = [("L1", "L1 AUC", .5), ("L3", "L3 AUC", .5), ("T1_gap", "T1 gap", 0),
            ("T3b_z", "T3b z", 0), ("T3c_z", "T3c z", 0)]
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(15, 3.2))
    for ax, (k, title, ref) in zip(axes, cols):
        vals = [tests[n]["values"][k] for n in names]
        ax.barh(range(len(names)), np.array(vals) - ref, left=ref, color=PAL[0], height=.5)
        ax.axvline(ref, color=GREY, lw=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.split("_", 1)[1].replace("_", " ") for n in names] if ax is axes[0] else [])
        ax.set_title(title, loc="left", fontsize=9)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    v = tests["5_abrupt_change"]["values"]
    ax = axes[-1]
    ax.plot(range(1, len(v["velocity_h1"]) + 1), v["velocity_h1"], color=PAL[0], lw=2, marker="o", ms=4)
    ax.axvline(v["change_transition"] + 1, color=GREY, ls="--", lw=1)
    ax.set_title("5 abrupt change: H1 velocity", loc="left", fontsize=9)
    ax.set_xlabel("transition")
    status = "  ".join(f"{n.split('_')[0]}:{'pass' if t['pass'] else 'FAIL'}" for n, t in tests.items())
    fig.suptitle(f"Synthetic validation (n={report['n']}, d={report['d']})   {status}", x=.01, ha="left", fontsize=10)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "fig0_validation.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b-deduped")
    ap.add_argument("--input", default="P1")
    a = ap.parse_args()
    res = C.RESULTS / a.model / a.input
    m = pd.read_csv(res / "metrics.csv")
    ref = json.load(open(res / "reference_layer.json"))["reference_layer"]
    vel = pd.read_csv(res / "velocity.csv") if (res / "velocity.csv").exists() else pd.DataFrame(
        columns=["layer", "seed", "kind", "hom", "prep", "step_from", "step_to", "velocity"])
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}" + ("" if a.input == "P1" else f"_{a.input}")
    fig1(m, vel, ref, a.model, C.FIGURES / f"fig1_timeline_{tag}.png")
    fig2(m, vel, a.model, C.FIGURES / f"fig2_heatmaps_{tag}.png")
    if a.input == "P1":
        fig3(m, ref, a.model, C.FIGURES / f"fig3_global_local_{tag}.png")
    p2 = {mod: (pd.read_csv(p / "metrics.csv"), json.load(open(p / "reference_layer.json"))["reference_layer"])
          for mod in C.MODELS if (p := C.RESULTS / mod / "P2").joinpath("metrics.csv").exists()}
    if p2:
        fig4(p2, C.FIGURES / "fig4_replication.png")


if __name__ == "__main__":
    main()
