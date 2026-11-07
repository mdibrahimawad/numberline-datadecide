"""E12b report: variance shares, power by noise level, verdicts (PREREG_B rules) and a dimension-vs-k figure.

  python -m src.e12b_report --out results/e12b_smooth_digits
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e10_graded_analysis import _style
from src.e12_manifold import DEPTH, KS
from src.e12_report import SET_ORDER

COL = {"f": "#0b0b0b", "Z_minus_digits": "#e34948", "curve": "#2a78d6", "sheet_fat": "#eb6834"}


def over5(k):
    return "supported" if k >= 4 else ("not supported" if k <= 1 else "mixed")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/e12b_smooth_digits"); a = ap.parse_args()
    T = pd.read_csv(os.path.join(a.out, "tests.csv")); P = pd.read_csv(os.path.join(a.out, "power_replicates.csv"))
    M = T[T.variant == "main"]
    # power per dataset and noise level
    pw = P.groupby(["model", "set", "depth", "noise", "condition"]).plateau.sum().unstack()
    pw["power_fat"] = (pw["curve"] >= 4) & (pw["sheet_fat"] <= 1); pw["power_thin"] = (pw["curve"] >= 4) & (pw["sheet_thin"] <= 1)
    pw.to_csv(os.path.join(a.out, "power.csv"))
    # verdicts per dataset
    rows = []
    for (m, s, d), g in M.groupby(["model", "set", "depth"]):
        f = g[g.condition == "f"].iloc[0]; zd = g[g.condition == "Z_minus_digits"].iloc[0]
        p1 = bool(pw.loc[(m, s, d, 1.0)].power_fat)
        lowest = [c for c in (0.1, 0.3, 1.0) if pw.loc[(m, s, d, c)].power_fat]
        rows.append(dict(model=m, set=s, depth=d, share_f=f.share_f, share_digit=f.share_digit, share_resid=f.share_resid,
                         h_frac=f.h_frac, f_is_1D=bool(f.T2_plateau),
                         f_recovers_order=bool(f.T3_random_rho >= 0.9 and f.T3_block_rho >= 0.8),
                         power_at_1x=p1, lowest_noise_with_power=min(lowest) if lowest else np.nan,
                         Zd_plateau=bool(zd.T2_plateau),
                         Zd_is_1D=("cannot tell" if not p1 else bool(zd.T2_plateau)),
                         f_mle_k20=f.T2_mle_k20, Zd_mle_k20=zd.T2_mle_k20, f_T3_random=f.T3_random_rho, f_T3_block=f.T3_block_rho,
                         Zd_T3_random=zd.T3_random_rho, Zd_T3_block=zd.T3_block_rho,
                         f_spacing=f.T4x_best, f_p=f.T4x_p, f_p_lo=f.T4x_p_lo, f_p_hi=f.T4x_p_hi))
    V = pd.DataFrame(rows); V.to_csv(os.path.join(a.out, "verdicts_by_depth.csv"), index=False)
    O = []
    for (m, s), g in V.groupby(["model", "set"]):
        zd = g.Zd_is_1D.astype(str)
        O.append(dict(model=m, set=s, share_f=f"{g.share_f.min():.2f}–{g.share_f.max():.2f}",
                      share_digit=f"{g.share_digit.min():.2f}–{g.share_digit.max():.2f}",
                      f_is_1D=f"{over5(int(g.f_is_1D.sum()))} ({int(g.f_is_1D.sum())}/5)",
                      f_recovers_order=f"{over5(int(g.f_recovers_order.sum()))} ({int(g.f_recovers_order.sum())}/5)",
                      power_at_1x=f"{int(g.power_at_1x.sum())}/5", power_at_0p3x=f"{int((g.lowest_noise_with_power <= 0.3).sum())}/5",
                      power_at_0p1x=f"{int((g.lowest_noise_with_power <= 0.1).sum())}/5",
                      Zd_is_1D=("cannot tell" if (zd == "cannot tell").sum() >= 3 else f"{over5(int((zd == 'True').sum()))} ({int((zd == 'True').sum())}/5)")))
    O = pd.DataFrame(O); O.to_csv(os.path.join(a.out, "verdicts.csv"), index=False)
    # units-only secondary
    U = T[(T.variant == "units_only") & (T.condition.isin(["f", "Z_minus_digits"]))]
    U[["model", "set", "depth", "condition", "share_f", "share_digit", "T2_mle_k20", "T2_plateau", "T3_random_rho", "T3_block_rho"]].to_csv(
        os.path.join(a.out, "secondary_units_only.csv"), index=False)
    # figure
    _style(plt)
    fig, axes = plt.subplots(6, 5, figsize=(17, 19), sharex=True)
    for i, (m, s) in enumerate(SET_ORDER):
        for j, d in enumerate(DEPTH):
            ax = axes[i, j]; g = M[(M.model == m) & (M.set == s) & (M.depth == d)]; ax.axhspan(0.8, 1.5, color="#e4e3df", lw=0)
            for cond, col in COL.items():
                for c, ls in ((np.nan, "-"), (0.1, ":"), (0.3, "--"), (1.0, "-")):
                    r = g[(g.condition == cond) & ((g.noise.isna()) if np.isnan(c) else (g.noise == c))]
                    if len(r):
                        lab = cond.replace("_", " ") + ("" if np.isnan(c) else f" {c}x")
                        ax.plot(KS, [r[f"T2_mle_k{k}"].values[0] for k in KS], ls, marker="o", ms=3, color=col, lw=2.2 if cond == "f" else 1.2, label=lab)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_xticks(KS, [str(k) for k in KS])
            ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]); ax.set_ylim(0.5, 45)
            if i == 0: ax.set_title(d, fontsize=10)
            if j == 0: ax.set_ylabel(f"{m.split('-')[0]} {s}\nMLE dimension", fontsize=9)
            if i == 5: ax.set_xlabel("k neighbours")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    fig.suptitle("E12b: dimension vs k of the smooth part f (black), digit-removed data (red), fake curve (blue) and fat sheet (orange) at noise 0.1x (dotted), 0.3x (dashed), 1x (solid)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(os.path.join(a.out, "fig_dimension_vs_k.png"), dpi=130); plt.close(fig)
    print(O.to_string(index=False))


if __name__ == "__main__":
    main()
