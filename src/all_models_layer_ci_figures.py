"""Plot per-layer metrics for all saved three-seed geometry runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/geometry/all_models_layer_ci"
T95_DF2 = 4.302652729911275

MODELS = (
    ("Falcon-RW-1B", "joint_recomputed/falcon_rw_1b/selection_runs.json", 0),
    ("Falcon-RW-7B", "joint_recomputed/falcon_rw_7b/selection_runs.json", 0),
    ("INCITE-3B", "joint_recomputed/redpajama_3b/selection_runs.json", 0),
    ("INCITE-7B", "joint_recomputed/redpajama_7b/selection_runs.json", 0),
    ("OLMo-7B-2T", "joint_recomputed/olmo_7b_2t/selection_runs.json", 0),
    ("OLMo-7B-Twin-2T", "joint_recomputed/olmo_7b_twin_2t/selection_runs.json", 0),
    ("StarCoderBase-1B", "joint_recomputed/starcoderbase_1b/selection_runs.json", 0),
    ("StarCoderBase-3B", "joint_recomputed/starcoderbase_3b/selection_runs.json", 0),
    ("StarCoderBase-7B", "joint_recomputed/starcoderbase_7b/selection_runs.json", 0),
    ("StarCoderBase-15.5B", "joint_recomputed_extended/starcoderbase_15_5b/selection_runs.json", 0),
    ("OpenLLaMA-3B", "per_seed/open_llama_3b/results.json", 0),
    ("OpenLLaMA-7B", "per_seed/open_llama_7b/results.json", 0),
    ("OpenLLaMA-13B", "joint_recomputed_extended/open_llama_13b/selection_runs.json", 0),
    ("LLM360/Amber", "joint_recomputed_extended/amber/selection_runs.json", 1),
    ("LLM360/Crystal", "joint_recomputed_extended/crystal/selection_runs.json", 1),
    ("LLM360/K2", "joint_recomputed_extended/k2/selection_runs.json", 1),
    ("OLMo-3-32B", "per_seed/olmo_3_32b/results.json", 1),
    ("BTLM-3B-8K", "joint_recomputed_extended/btlm_3b/selection_runs.json", 1),
    ("LLM-JP-3-1.8B", "joint_recomputed_extended/llm_jp_1_8b/selection_runs.json", 1),
    ("LLM-JP-3-3.7B", "joint_recomputed_extended/llm_jp_3_7b/selection_runs.json", 1),
    ("LLM-JP-3-7.2B", "joint_recomputed_extended/llm_jp_7_2b/selection_runs.json", 1),
    ("LLM-JP-3-13B", "joint_recomputed_extended/llm_jp_13b/selection_runs.json", 1),
    ("LLM-JP-3-172B", "joint_recomputed_extended/llm_jp_172b/selection_runs.json", 1),
)

METRICS = {
    "beta": (r"Scaling factor $\beta$", True),
    "ev": (r"Explained variance $\sigma^2$", False),
    "rho": (r"Monotonicity $|\rho_s|$", False),
}


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load(relative: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads((ROOT / "results/geometry" / relative).read_text())
    runs = payload["runs"]
    if len(runs) != 3:
        raise ValueError(f"{relative}: expected three seeds, found {len(runs)}")
    layers = np.array(sorted(set.intersection(*(
        set(map(int, run["layers"])) for run in runs
    ))))
    layers = layers[layers > 0]  # hidden_states[0] is the embedding output
    values = np.array([
        [abs(run["layers"][str(layer)][metric]) if metric == "rho"
         else run["layers"][str(layer)][metric] for layer in layers]
        for run in runs
    ], dtype=float)
    depth = layers / max(layers.max(), 1)
    valid = np.all(np.isfinite(values), axis=0)
    depth, values = depth[valid], values[:, valid]
    if metric == "beta":
        logged = np.log(values)
        center = np.mean(logged, axis=0)
        half_width = T95_DF2 * np.std(logged, axis=0, ddof=1) / np.sqrt(3)
        return depth, np.exp(center), np.exp(center - half_width), np.exp(center + half_width)
    mean = np.mean(values, axis=0)
    half_width = T95_DF2 * np.std(values, axis=0, ddof=1) / np.sqrt(3)
    return depth, mean, np.clip(mean - half_width, 0, 1), np.clip(mean + half_width, 0, 1)


def _model_colors() -> list[np.ndarray]:
    panel_counts = [sum(model[2] == panel for model in MODELS) for panel in (0, 1)]
    panel_colors = (
        mpl.colormaps["viridis"](np.linspace(0.08, 0.92, panel_counts[0])),
        mpl.colormaps["plasma"](np.linspace(0.08, 0.92, panel_counts[1])),
    )
    panel_indices = [0, 0]
    colors = []
    for *_, panel in MODELS:
        colors.append(panel_colors[panel][panel_indices[panel]])
        panel_indices[panel] += 1
    return colors


def _plot(metric: str) -> None:
    ylabel, log_y = METRICS[metric]
    colors = _model_colors()
    styles = ("-", "--", "-.", ":")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharex=True, sharey=True)
    panel_titles = ("Corpus-matched model families", "Additional base models")
    handles = []

    for index, (label, relative, panel) in enumerate(MODELS):
        x, mean, low, high = _load(relative, metric)
        mask = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
        color = colors[index]
        line, = axes[panel].plot(
            x[mask], mean[mask], color=color, ls=styles[index % len(styles)],
            lw=1.25, label=label,
        )
        axes[panel].fill_between(x[mask], low[mask], high[mask], color=color, alpha=0.07, lw=0)
        handles.append(line)

    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, pad=7)
        ax.set_xlabel(r"Normalized transformer depth $\ell/L$")
        ax.set_xlim(0, 1)
        ax.grid(True, which="major", color="#D9D9D9", lw=0.45, alpha=0.75)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(0.1, 30)
            ax.axhline(1, color="#777777", lw=0.7, ls=":", zorder=0)
        else:
            ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(ylabel)

    fig.legend(
        handles=handles, labels=[model[0] for model in MODELS],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6,
        frameon=False, handlelength=2.1, columnspacing=1.25,
    )
    fig.suptitle(f"Layer-wise {ylabel} across models", y=0.985, fontsize=12)
    note = r"Pointwise 95\% $t$ CI ($n=3$)"
    if metric == "beta":
        note += "\n" + r"$\log\beta$ space; extreme tails clipped"
    axes[1].text(
        0.985, 0.965, note, transform=axes[1].transAxes,
        ha="right", va="top", fontsize=7.5, color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.25, wspace=0.08)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"all_models_layer_{metric}_ci95"
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_all_metrics() -> None:
    colors = _model_colors()
    styles = ("-", "--", "-.", ":")
    panels = (
        ("rho", r"(a) Monotonicity $|\rho_s|$"),
        ("ev", r"(b) Explained variance $\sigma^2$"),
        ("beta", r"(c) Scaling factor $\beta$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharex=True)
    handles = []

    for index, (label, relative, _) in enumerate(MODELS):
        for ax, (metric, _) in zip(axes, panels):
            x, mean, low, high = _load(relative, metric)
            mask = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
            line, = ax.plot(
                x[mask], mean[mask], color=colors[index],
                ls=styles[index % len(styles)], lw=1.05, label=label,
            )
            ax.fill_between(
                x[mask], low[mask], high[mask],
                color=colors[index], alpha=0.045, lw=0,
            )
            if metric == "rho":
                handles.append(line)

    for ax, (metric, title) in zip(axes, panels):
        ax.set_title(title, pad=7)
        ax.set_xlabel(r"Normalized transformer depth $\ell/L$")
        ax.set_xlim(0, 1)
        ax.grid(True, which="major", color="#D9D9D9", lw=0.45, alpha=0.75)
        if metric == "beta":
            ax.set_yscale("log")
            ax.set_ylim(0.1, 30)
            ax.axhline(1, color="#777777", lw=0.7, ls=":", zorder=0)
            ax.text(
                0.98, 0.965,
                r"95\% $t$ CI in $\log\beta$ space" + "\n" + "extreme tails clipped",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.2, color="#555555",
            )
        else:
            ax.set_ylim(0, 1.02)
            ax.text(
                0.98, 0.965, r"Pointwise 95\% $t$ CI ($n=3$)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.2, color="#555555",
            )
    axes[0].set_ylabel(r"Monotonicity $|\rho_s|$")
    axes[1].set_ylabel(r"Explained variance $\sigma^2$")
    axes[2].set_ylabel(r"Scaling factor $\beta$")

    fig.legend(
        handles=handles, labels=[model[0] for model in MODELS],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6,
        frameon=False, handlelength=2.1, columnspacing=1.25,
    )
    fig.suptitle("Layer-wise numerical geometry across all 23 models", y=0.985, fontsize=12)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.87, bottom=0.25, wspace=0.22)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "all_23_models_layer_rho_ev_beta_ci95"
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    assert len(MODELS) == 23 and {panel for *_, panel in MODELS} == {0, 1}
    _style()
    for metric in METRICS:
        _plot(metric)
    _plot_all_metrics()


if __name__ == "__main__":
    main()
