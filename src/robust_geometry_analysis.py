"""Plot ordinary versus robust PCA across model layers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="OLMo-3-32B")
    args = parser.parse_args()

    with args.csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    layer = np.asarray([int(row["layer"]) for row in rows])
    pca_ev = np.asarray([float(row["pca_ev"]) for row in rows])
    rpca_ev = np.asarray([float(row["rpca_ev"]) for row in rows])
    pca_rho = np.asarray([float(row["pca_rho"]) for row in rows])
    rpca_rho = np.asarray([float(row["rpca_rho"]) for row in rows])
    pca_best = int(layer[np.nanargmax(pca_ev)])
    rpca_best = int(layer[np.nanargmax(rpca_ev)])

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), sharex=True)
    colors = {"pca": "#222222", "rpca": "#0072B2"}

    axes[0].plot(layer, pca_ev, color=colors["pca"], lw=1.25, label="PCA")
    axes[0].plot(layer, rpca_ev, color=colors["rpca"], lw=1.25, ls="--", label="Robust PCA")
    axes[0].scatter([pca_best], [pca_ev[pca_best - 1]], color=colors["pca"], s=24, zorder=3)
    axes[0].scatter([rpca_best], [rpca_ev[rpca_best - 1]], color=colors["rpca"], s=24, zorder=3)
    axes[0].axvline(rpca_best, color="#777777", lw=0.8, ls=":")
    axes[0].annotate(
        f"PCA/RPCA best: L={rpca_best}",
        xy=(rpca_best, rpca_ev[rpca_best - 1]),
        xytext=(7, -22),
        textcoords="offset points",
        ha="left",
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7},
    )
    axes[0].set_ylabel("explained variance")
    axes[0].set_title("(a) Variance criterion")
    axes[0].legend(loc="upper right")

    axes[1].plot(layer, pca_rho, color=colors["pca"], lw=1.25, label="PCA")
    axes[1].plot(layer, rpca_rho, color=colors["rpca"], lw=1.25, ls="--", label="Robust PCA")
    axes[1].axvline(rpca_best, color="#777777", lw=0.8, ls=":")
    axes[1].set_ylabel(r"$|\rho_s|$")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) Numerical monotonicity")

    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_xlim(1, 64)
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        ax.grid(True, color="#D8D8D8", lw=0.45, alpha=0.7)
    fig.suptitle(args.title, y=1.01, fontsize=11)
    fig.tight_layout(w_pad=2.1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
