"""Figure for E11 checkpoints: neighbourhood composition, copy accuracy and rare/frequent crowding over training
(~84 % depth). python -m src.e11_ckpt_figure -> results/e11_ckpt/fig_trajectories.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e10_graded_analysis import INK2, _style

COL = {"share_within_50": "#2a78d6", "share_same_leading": "#1baf7a", "share_same_last": "#eb6834", "copy": "#6250d6"}


def main(depth="~84%"):
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", gridspec_kw=dict(height_ratios=[1.4, 1]))
    for j, fam in enumerate(["pythia-1.4b", "olmo2-1b"]):
        k = pd.read_csv(f"results/e11_ckpt/{fam}_knn.csv"); c = pd.read_csv(f"results/e11_ckpt/{fam}_C2_per_checkpoint.csv")
        k = k[k.depth == depth].sort_values("step"); c = c[c.depth == depth].sort_values("step")
        x = np.log10(k.step.values + 1)
        ax = axes[0, j]
        for col, lab in (("share_within_50", "neighbours within ±50 in value"), ("share_same_leading", "neighbours with the same leading digit"),
                         ("share_same_last", "neighbours with the same last digit")):
            ax.plot(x, k[col], "o-", color=COL[col], ms=4, label=lab)
        ax.plot(x, k.copy_first_token_acc, "s--", color=COL["copy"], ms=4, lw=1.5, label="copy task: first-token accuracy")
        ax.axhline(0.11, color=INK2, lw=0.8, ls=":"); ax.text(x[-1], 0.12, "chance ≈ 0.11", fontsize=7, ha="right", color=INK2)
        ax.set_ylim(-0.02, 1.05); ax.set_title(f"{fam}, {depth} depth", fontsize=10, loc="left")
        if j == 0: ax.set_ylabel("share of the 10 nearest neighbours"); ax.legend(frameon=False, fontsize=7.5, loc="center left")
        ax = axes[1, j]
        ax.plot(x, c.ratio_raw, "o-", color="#2a78d6", ms=4)
        sig = c.p_raw.values < 0.05
        ax.plot(x[sig], c.ratio_raw.values[sig], "o", color="#e34948", ms=8, mfc="none", mew=1.5, label="p < 0.05")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(x, [f"{s:,}" for s in k.step], rotation=60, fontsize=7); ax.set_xlabel("training step (log scale)")
        if j == 0: ax.set_ylabel("rare ÷ frequent typical gap\n(< 1: rare more crowded)"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig("results/e11_ckpt/fig_trajectories.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
