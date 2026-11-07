"""E12 report: verdicts (PREREG decision rules), summary tables and the dimension-vs-k figure.

  python -m src.e12_report --out results/e12_manifold
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e10_graded_analysis import INK2, _style
from src.e12_manifold import DEPTH, KS, PERIODS

COLORS = {"real": "#0b0b0b", "curve": "#2a78d6", "log_curve": "#1baf7a", "helix": "#6250d6", "sheet_thin": "#eda100",
          "sheet_fat": "#eb6834", "shuffled": "#b9b8b2", "matched_smooth": "#e87ba4"}
SET_ORDER = [("qwen2.5-1.5b", "Q2"), ("qwen2.5-1.5b", "Q3"), ("qwen2.5-1.5b", "Q4"),
             ("pythia-1.4b", "P2"), ("pythia-1.4b", "P3"), ("pythia-1.4b", "P1")]


def power_table(P):
    g = P.groupby(["model", "set", "depth", "condition"]).plateau.sum().unstack()
    g["power_fat"] = (g["curve"] >= 4) & (g["sheet_fat"] <= 1)
    g["power_thin"] = (g["curve"] >= 4) & (g["sheet_thin"] <= 1)
    return g.rename(columns={"curve": "curve_pass_of5", "sheet_thin": "thin_pass_of5", "sheet_fat": "fat_pass_of5"})


def verdict_rows(T, PW):
    rows = []
    for (m, s, d), g in T.groupby(["model", "set", "depth"]):
        g = g.set_index("condition"); r = g.loc["real"]; ms = g.loc["matched_smooth"]; pw = PW.loc[(m, s, d)]
        h5 = (abs(r.T2_mle_k20 - ms.T2_mle_k20) <= 0.2 * ms.T2_mle_k20) and (abs(r.T3_random_rho - ms.T3_random_rho) <= 0.05)
        small = all(0.8 <= r[f"T2_mle_k{k}"] <= 1.5 for k in (5, 10, 20))
        per = any((r.get(f"T7_pval_p{p}", np.nan) < 0.01) and (r.get(f"T7_share_p{p}", 0) >= 0.02) for p in PERIODS)
        h1 = bool(pw.power_fat and r.T2_plateau and r.T3_random_rho >= 0.9 and r.T3_block_rho >= 0.8 and not h5)
        h4 = bool(pw.power_fat and r.T2_ratio_80_5 <= 0.5 and r.T3_random_rho >= 0.8 and (r.T3_fine_rho < 0.5 if np.isfinite(r.T3_fine_rho) else False))
        rows.append(dict(model=m, set=s, depth=d, power_fat=bool(pw.power_fat), power_thin=bool(pw.power_thin),
                         H1=h1, H2=bool(per and not small),
                         H3=bool((r.T6_r2_digits - r.T6_r2_value >= 0.05) and (r.T6_nn_same_last >= 3 * r.T6_chance_same_last)),
                         H4=h4, H5=bool(h5)))
    return pd.DataFrame(rows)


def over_depths(V):
    out = []
    for (m, s), g in V.groupby(["model", "set"]):
        row = dict(model=m, set=s, depths_with_power=int(g.power_fat.sum()))
        for h in ("H1", "H2", "H3", "H4", "H5"):
            k = int(g[h].sum())
            if h in ("H1", "H4") and row["depths_with_power"] < 3 and k < 4:
                row[h] = "cannot tell"
            else:
                row[h] = "supported" if k >= 4 else ("not supported" if k <= 1 else "mixed")
            row[h] += f" ({k}/5)"
        out.append(row)
    return pd.DataFrame(out)


def figure(T, out):
    _style(plt)
    fig, axes = plt.subplots(6, 5, figsize=(17, 19), sharex=True)
    for i, (m, s) in enumerate(SET_ORDER):
        for j, d in enumerate(DEPTH):
            ax = axes[i, j]; g = T[(T.model == m) & (T.set == s) & (T.depth == d)]
            ax.axhspan(0.8, 1.5, color="#e4e3df", lw=0)
            for c, col in COLORS.items():
                r = g[g.condition == c]
                if len(r):
                    y = [r[f"T2_mle_k{k}"].values[0] for k in KS]
                    ax.plot(KS, y, "o-", color=col, lw=2.4 if c == "real" else 1.3, ms=3, label=c.replace("_", " "))
            ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_xticks(KS, [str(k) for k in KS]); ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]); ax.set_ylim(0.7, 45)
            if i == 0: ax.set_title(d, fontsize=10)
            if j == 0: ax.set_ylabel(f"{m.split('-')[0]} {s}\nMLE dimension", fontsize=9)
            if i == 5: ax.set_xlabel("k neighbours")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Levina–Bickel dimension vs k: real (black) vs fakes. Grey band = 1-D plateau zone [0.8, 1.5].", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(os.path.join(out, "fig_dimension_vs_k.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/e12_manifold"); a = ap.parse_args()
    T = pd.read_csv(os.path.join(a.out, "tests.csv")); P = pd.read_csv(os.path.join(a.out, "power_replicates.csv"))
    PW = power_table(P); PW.to_csv(os.path.join(a.out, "T1_power.csv"))
    V = verdict_rows(T, PW); V.to_csv(os.path.join(a.out, "verdicts_by_depth.csv"), index=False)
    O = over_depths(V); O.to_csv(os.path.join(a.out, "verdicts.csv"), index=False)
    cols = ["model", "set", "depth", "condition", "T2_mle_k5", "T2_mle_k20", "T2_mle_k80", "T2_twonn", "T2_plateau",
            "T3_random_rho", "T3_block_rho", "T3_extrap_rho", "T3_fine_rho", "T4_best", "T4_p", "T4_p_lo", "T4_p_hi",
            "T5_r2_linear", "T5_r2_log", "T6_r2_digits", "T6_r2_both", "T6_nn_same_last", "T6_nn_same_leading", "T6_nn_close",
            "T7_share_p2", "T7_pval_p2", "T7_share_p5", "T7_pval_p5", "T7_share_p10", "T7_pval_p10", "T7_share_p100", "T7_pval_p100",
            "T8_long_H1", "T8_max_H1_ratio"]
    T[[c for c in cols if c in T]].to_csv(os.path.join(a.out, "summary_table.csv"), index=False)
    figure(T, a.out)
    print(PW.to_string()); print(O.to_string(index=False))


if __name__ == "__main__":
    main()
