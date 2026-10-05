"""Plot a joint-selected layer profile from saved seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
    })


def _plot(payload: dict, selected_layer: int, criterion: str, output: Path, title: str) -> None:
    layers = np.asarray(sorted(map(int, payload["aggregate_layers"])))
    runs = payload["runs"]
    metrics = {
        key: np.asarray([[run["layers"][str(layer)][key] for layer in layers] for run in runs])
        for key in ("ev", "rho", "beta")
    }
    colors = {"ev": "#0072B2", "rho": "#D55E00", "beta": "#009E73"}
    panels = (
        ("ev", "explained variance", False),
        ("rho", r"$|\rho_s|$  (PC1 vs target)", False),
        ("beta", r"direct-fit scaling rate $\beta$", True),
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.5), sharex=True)
    for ax, (key, ylabel, log_scale) in zip(axes, panels):
        values = metrics[key]
        mean = np.asarray([np.nanmean(column) if np.isfinite(column).any() else np.nan
                           for column in values.T])
        std = np.asarray([np.nanstd(column) if np.isfinite(column).any() else np.nan
                          for column in values.T])
        mask = np.isfinite(mean)
        ax.plot(layers[mask], mean[mask], color=colors[key], lw=1.3)
        ax.fill_between(layers[mask], mean[mask] - std[mask], mean[mask] + std[mask],
                        color=colors[key], alpha=0.18, lw=0)
        ax.axvline(selected_layer, color="#222222", lw=0.9, ls="--")
        ax.set(xlabel="layer", ylabel=ylabel, xlim=(layers.min(), layers.max()))
        ax.grid(True, color="#D8D8D8", lw=0.45, alpha=0.7)
        if log_scale:
            ax.axhline(1, color="#888888", lw=0.7, ls=":")
            ax.set_yscale("log")
        ax.text(0.97, 0.94, rf"$L^*={selected_layer}$", transform=ax.transAxes,
                ha="right", va="top")

    fig.suptitle(f"{title} - {criterion}", y=0.98, fontsize=11)
    fig.subplots_adjust(wspace=0.30, top=0.82)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("per_seed", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="OpenLLaMA-7B")
    args = parser.parse_args()

    payload = json.loads(args.per_seed.read_text())
    selection = json.loads(args.selection.read_text())
    joint_layer = int(selection["selected_layer"])

    _style()
    _plot(payload, joint_layer, r"joint $Q_l$ selection", args.output, args.title)


if __name__ == "__main__":
    main()
