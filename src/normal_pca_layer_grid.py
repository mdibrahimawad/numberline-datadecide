"""Create a multi-model ordinary-PCA layer-profile figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "LLM360/Amber": "Amber",
    "LLM360/Crystal": "Crystal",
    "LLM360/K2": "K2",
    "cerebras/btlm-3b-8k-base": "BTLM-3B",
    "llm-jp/llm-jp-3-1.8b": "LLM-JP-1.8B",
    "llm-jp/llm-jp-3-3.7b": "LLM-JP-3.7B",
    "llm-jp/llm-jp-3-7.2b": "LLM-JP-7.2B",
    "llm-jp/llm-jp-3-13b": "LLM-JP-13B",
    "llm-jp/llm-jp-3-172b": "LLM-JP-172B",
    "bigcode/starcoderbase": "StarCoderBase-15.5B",
    "Qwen/Qwen2.5-32B": "Qwen2.5-32B",
    "Qwen/Qwen2.5-72B": "Qwen2.5-72B",
    "01-ai/Yi-1.5-34B": "Yi-1.5-34B",
    "deepseek-ai/deepseek-llm-67b-base": "DeepSeek-67B",
    "tiiuae/falcon-40b": "Falcon-40B",
    "mistralai/Mixtral-8x7B-v0.1": "Mixtral-8x7B",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        model = data["config"]["model_name"]
        if model == "allenai/Olmo-3-1125-32B":
            continue
        records.append((DISPLAY_NAMES.get(model, model), data))
    records.sort(key=lambda item: list(DISPLAY_NAMES.values()).index(item[0]))

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 8.0), sharey=True)
    for ax, (name, data) in zip(axes.flat, records):
        pca = data["pca"]
        layers = np.asarray(sorted(map(int, pca)))
        ev = np.asarray([pca[str(layer)]["ev_mean"] for layer in layers])
        ev_std = np.asarray([pca[str(layer)]["ev_std"] for layer in layers])
        rho = np.asarray([pca[str(layer)]["rho_mean"] for layer in layers])
        rho_std = np.asarray([pca[str(layer)]["rho_std"] for layer in layers])
        best = int(data["best_layer_pca"])

        ax.plot(layers, ev, color="#222222", lw=1.15)
        ax.fill_between(layers, ev - ev_std, ev + ev_std, color="#222222", alpha=0.10, lw=0)
        ax.plot(layers, rho, color="#0072B2", lw=1.15, ls="--")
        ax.fill_between(layers, rho - rho_std, rho + rho_std, color="#0072B2", alpha=0.10, lw=0)
        ax.axvline(best, color="#888888", lw=0.75, ls=":")
        ax.set_title(f"{name}  ($L^*={best}$)", pad=3)
        ax.set_xlim(layers.min(), layers.max())
        ax.set_ylim(0, 1.05)
        ax.grid(True, color="#D8D8D8", lw=0.4, alpha=0.65)

    for row in axes:
        row[0].set_ylabel("score")
    for ax in axes[-1]:
        ax.set_xlabel("layer")
    handles = [
        plt.Line2D([], [], color="#222222", lw=1.3, label="explained variance"),
        plt.Line2D([], [], color="#0072B2", lw=1.3, ls="--", label=r"$|\rho_s|$"),
        plt.Line2D([], [], color="#888888", lw=0.8, ls=":", label=r"best-EV layer $L^*$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.1, w_pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
