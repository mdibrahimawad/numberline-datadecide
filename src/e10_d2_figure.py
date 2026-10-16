"""E10 stage D/D2 figure: 9 x 9 matrices of RTD / shuffled RTD (lower = more similar number topology), one per
relative depth, models ordered by how they tokenize numbers.

  python -m src.e10_d2_figure   -> results/e10_convergence_d2/fig_rtd_by_depth.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.e10_graded_analysis import INK, SURF, _style

ORDER = ["crystal", "openllama-3b", "starcoderbase-3b", "qwen2.5-1.5b", "smollm2-1.7b", "pythia-1.4b", "redpajama-3b", "olmo2-1b", "llama3.2-1b"]
SHORT = ["Crystal", "OpenLLaMA", "StarCoder", "Qwen2.5", "SmolLM2", "Pythia", "RedPajama", "OLMo-2", "Llama-3.2"]
GROUPS = {"digit": range(0, 5), "neox": range(5, 7), "chunk3": range(7, 9)}
DEPTHS = ["~25%", "~50%", "~75%", "~84%", "last"]


def matrices(d):
    out = {}
    for dep in DEPTHS:
        M = np.full((9, 9), np.nan)
        for _, r in d.iterrows():
            i, j = ORDER.index(r.a), ORDER.index(r.b); M[i, j] = M[j, i] = r[dep]
        out[dep] = M
    return out


def main():
    _style(plt)
    d = pd.read_csv("results/e10_convergence_d2/ratio_by_pair.csv")
    Ms = matrices(d)
    cmap = LinearSegmentedColormap.from_list("seq", ["#0d366b", "#1c5cab", "#5598e7", "#cde2fb", "#f4f3ef"])
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    for ax, dep in zip(axes, DEPTHS):
        M = Ms[dep]
        im = ax.imshow(M, cmap=cmap, vmin=0.15, vmax=1.0)
        for i in range(9):
            for j in range(9):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=SURF if M[i, j] < 0.5 else INK)
        for b in (4.5, 6.5):
            ax.axhline(b, color=INK, lw=1.2); ax.axvline(b, color=INK, lw=1.2)
        ax.set_xticks(range(9), SHORT, rotation=90, fontsize=7.5); ax.set_yticks(range(9), SHORT if dep == "~25%" else [""] * 9, fontsize=7.5)
        ax.grid(False); ax.set_title(f"depth {dep}", fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("RTD / shuffled RTD (dark = same topology, 1 = chance)")
    fig.suptitle("Number topology similarity between models. Lines separate tokenizer groups: digit-by-digit | GPT-NeoX chunks | ≤3-digit chunks",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig("results/e10_convergence_d2/fig_rtd_by_depth.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    for k, dep in enumerate(DEPTHS):                      # one larger picture per depth
        M = Ms[dep]; fig, ax = plt.subplots(figsize=(7.2, 6.2))
        im = ax.imshow(M, cmap=cmap, vmin=0.15, vmax=1.0)
        for i in range(9):
            for j in range(9):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color=SURF if M[i, j] < 0.5 else INK)
        for b in (4.5, 6.5):
            ax.axhline(b, color=INK, lw=1.5); ax.axvline(b, color=INK, lw=1.5)
        ax.set_xticks(range(9), SHORT, rotation=60, ha="right", fontsize=9); ax.set_yticks(range(9), SHORT, fontsize=9); ax.grid(False)
        ax.set_title(f"Depth {dep}: similarity of number topology (dark = same shape, 1 = chance)", fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout(); fig.savefig(f"results/e10_convergence_d2/fig_rtd_depth{k + 1}_{dep.strip('~%')}.png", dpi=150); plt.close(fig)
    rows = []
    for dep, M in Ms.items():
        blk = lambda A, B: np.nanmean(M[np.ix_(list(A), list(B))][~np.eye(9, dtype=bool)[np.ix_(list(A), list(B))]])
        g = GROUPS
        rows.append(dict(depth=dep, digit_digit=blk(g["digit"], g["digit"]), neox_pair=M[5, 6], chunk3_pair=M[7, 8],
                         digit_vs_neox=blk(g["digit"], g["neox"]), digit_vs_chunk3=blk(g["digit"], g["chunk3"]),
                         neox_vs_chunk3=blk(g["neox"], g["chunk3"])))
    t = pd.DataFrame(rows).round(2); t.to_csv("results/e10_convergence_d2/group_means_by_depth.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
