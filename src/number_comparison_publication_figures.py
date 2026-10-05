from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ANALYSIS_DIR = Path(
    "results/number_comparison/fixed_gap_fewshot_exemplar_bootstrap_analysis"
)

MODEL_ORDER = [
    "Falcon-RW-1B",
    "Falcon-RW-7B",
    "RedPajama-3B",
    "RedPajama-7B",
    "OLMo-7B-2T",
    "OLMo-7B-Twin-2T",
    "StarCoderBase-1B",
    "StarCoderBase-3B",
    "StarCoderBase-7B",
]

LABEL_TEXT = {
    "Falcon-RW-1B": "Falcon-1B",
    "Falcon-RW-7B": "Falcon-7B",
    "RedPajama-3B": "RPJ-3B",
    "RedPajama-7B": "RPJ-7B",
    "OLMo-7B-2T": "OLMo-2T",
    "OLMo-7B-Twin-2T": "OLMo-Twin",
    "StarCoderBase-1B": "SCB-1B",
    "StarCoderBase-3B": "SCB-3B",
    "StarCoderBase-7B": "SCB-7B",
}

FAMILY = {
    "Falcon-RW-1B": "Falcon",
    "Falcon-RW-7B": "Falcon",
    "RedPajama-3B": "RedPajama",
    "RedPajama-7B": "RedPajama",
    "OLMo-7B-2T": "OLMo",
    "OLMo-7B-Twin-2T": "OLMo",
    "StarCoderBase-1B": "StarCoderBase",
    "StarCoderBase-3B": "StarCoderBase",
    "StarCoderBase-7B": "StarCoderBase",
}

FAMILY_STYLE = {
    "Falcon": {"color": "#2b6f8a", "marker": "o"},
    "RedPajama": {"color": "#b66a3c", "marker": "s"},
    "OLMo": {"color": "#5f8d4e", "marker": "^"},
    "StarCoderBase": {"color": "#7157a4", "marker": "D"},
}

ACCURACY_COLOR = "#7c4d6b"
FIT_COLOR = "#3f756f"
TREND_COLOR = "#8b5a44"

LABEL_OFFSETS = {
    "Falcon-RW-1B": (4, -11),
    "Falcon-RW-7B": (4, 5),
    "RedPajama-3B": (4, -10),
    "RedPajama-7B": (4, 5),
    "OLMo-7B-2T": (4, 6),
    "OLMo-7B-Twin-2T": (4, -10),
    "StarCoderBase-1B": (4, -12),
    "StarCoderBase-3B": (4, 4),
    "StarCoderBase-7B": (4, 4),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.85,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _model_rows_by_shot(rows: list[dict[str, str]], shots: int) -> list[dict[str, str]]:
    selected = [row for row in rows if int(row["shots"]) == shots]
    return sorted(selected, key=lambda row: MODEL_ORDER.index(row["model_label"]))


def _summary_by_shot(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["shots"]): row for row in rows}


def _draw_beta_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    summary: dict[str, str],
    *,
    shots: int,
) -> None:
    xs = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
    ys = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
    lows = np.asarray([float(row["accuracy_ci_low"]) for row in rows], dtype=float)
    highs = np.asarray([float(row["accuracy_ci_high"]) for row in rows], dtype=float)
    xx = np.linspace(0.45, 11.1, 200)

    ax.errorbar(
        xs,
        ys,
        yerr=np.vstack([ys - lows, highs - ys]),
        fmt="none",
        ecolor="#b7b7b7",
        elinewidth=1.0,
        capsize=2.0,
        zorder=1,
    )
    for row in rows:
        family = FAMILY[row["model_label"]]
        style = FAMILY_STYLE[family]
        x = float(row["pca_beta"])
        y = float(row["accuracy_mean"])
        ax.scatter(
            x,
            y,
            s=34,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS[row["model_label"]]
        ax.annotate(
            LABEL_TEXT[row["model_label"]],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.8,
            ha="left",
            va="center",
            color="#222222",
            zorder=4,
        )

    coef = np.polyfit(xs, ys, 1)
    ax.plot(xx, coef[0] * xx + coef[1], color=TREND_COLOR, linewidth=1.25, zorder=2)
    ax.set_xlim(0.35, 11.35)
    ax.set_ylim(0.48, 1.02)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, color="#d5d5d5", linewidth=0.45, alpha=0.65)
    ax.tick_params(direction="in", top=True, right=True, length=3.5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.minorticks_on()
    ax.set_title(
        rf"{shots}-shot: $\rho_s={float(summary['rho_s_mean']):.2f}$, "
        rf"$R^2={float(summary['r2_mean']):.2f}$",
        pad=5,
    )
    ax.set_xlabel(r"PCA $\beta$")


def plot_beta_accuracy_panels(analysis_dir: Path) -> None:
    model_rows = _read_csv(analysis_dir / "fewshot_model_accuracy_by_shot.csv")
    summary = _summary_by_shot(_read_csv(analysis_dir / "fewshot_summary_mean_std_ci.csv"))

    fig, axes = plt.subplots(1, 5, figsize=(13.8, 2.85), sharey=True)
    for shots, ax in zip(range(5), axes):
        _draw_beta_panel(
            ax,
            _model_rows_by_shot(model_rows, shots),
            summary[shots],
            shots=shots,
        )
        if shots != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("accuracy")

    handles = []
    for family, style in FAMILY_STYLE.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="",
                color=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=family,
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.22, top=0.78, wspace=0.18)
    fig.savefig(analysis_dir / "fewshot_beta_accuracy_by_shot_publication.pdf", bbox_inches="tight")
    fig.savefig(analysis_dir / "fewshot_beta_accuracy_by_shot_publication.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_goodness_curve(analysis_dir: Path) -> None:
    model_rows = _read_csv(analysis_dir / "fewshot_model_accuracy_by_shot.csv")
    summary = _summary_by_shot(_read_csv(analysis_dir / "fewshot_summary_mean_std_ci.csv"))

    fig, ax = plt.subplots(figsize=(6.9, 4.25))
    shots = np.asarray(sorted(summary), dtype=float)
    for model in MODEL_ORDER:
        rows = sorted(
            [row for row in model_rows if row["model_label"] == model],
            key=lambda row: int(row["shots"]),
        )
        ys = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
        family = FAMILY[model]
        style = FAMILY_STYLE[family]
        ax.plot(
            shots,
            ys,
            marker=style["marker"],
            markersize=3.5,
            linewidth=0.85,
            color=style["color"],
            alpha=0.38,
            label=LABEL_TEXT[model],
        )

    mean_acc = np.asarray([float(summary[int(s)]["mean_accuracy_mean"]) for s in shots])
    mean_low = np.asarray([float(summary[int(s)]["mean_accuracy_ci_low"]) for s in shots])
    mean_high = np.asarray([float(summary[int(s)]["mean_accuracy_ci_high"]) for s in shots])
    ax.fill_between(shots, mean_low, mean_high, color=ACCURACY_COLOR, alpha=0.12, linewidth=0)
    ax.plot(
        shots,
        mean_acc,
        color=ACCURACY_COLOR,
        marker="o",
        markersize=4.8,
        linewidth=1.55,
        label="Average performance",
        zorder=4,
    )
    ax.set_xlabel("Few-shot examples")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(0.48, 1.02)
    ax.grid(True, color="#d5d5d5", linewidth=0.45, alpha=0.65)
    ax.tick_params(direction="in", top=True, right=False, length=3.8)
    ax.minorticks_on()

    ax2 = ax.twinx()
    rho = np.asarray([float(summary[int(s)]["rho_s_mean"]) for s in shots])
    r2 = np.asarray([float(summary[int(s)]["r2_mean"]) for s in shots])
    ax2.plot(
        shots,
        rho,
        color=FIT_COLOR,
        marker="s",
        markersize=4.8,
        linewidth=1.35,
        label=r"Rank fit $\rho_s$",
        zorder=5,
    )
    ax2.plot(
        shots,
        r2,
        color=FIT_COLOR,
        marker="^",
        markersize=4.8,
        linewidth=1.1,
        linestyle="--",
        label=r"Goodness-of-fit $R^2$",
        zorder=5,
    )
    ax2.set_ylabel(r"Goodness-of-fit for PCA $\beta$", color=FIT_COLOR)
    ax2.tick_params(axis="y", colors=FIT_COLOR, direction="in", length=3.8)
    ax2.spines["right"].set_color(FIT_COLOR)
    ax.spines["left"].set_color(ACCURACY_COLOR)
    ax.tick_params(axis="y", colors=ACCURACY_COLOR)
    ax2.set_ylim(0.15, 0.75)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    key_handles = handles_left[-1:] + handles_right
    key_labels = labels_left[-1:] + labels_right
    model_handles = handles_left[:-1]
    model_labels = labels_left[:-1]
    legend1 = ax.legend(
        key_handles,
        key_labels,
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        model_handles,
        model_labels,
        loc="upper left",
        frameon=False,
        fontsize=6.7,
        ncol=3,
        columnspacing=0.7,
        handlelength=1.5,
    )
    ax.set_title("Accuracy benchmark", pad=6)
    fig.savefig(analysis_dir / "fewshot_accuracy_goodness_curve_publication.pdf", bbox_inches="tight")
    fig.savefig(analysis_dir / "fewshot_accuracy_goodness_curve_publication.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    _setup_style()
    plot_beta_accuracy_panels(args.analysis_dir)
    plot_accuracy_goodness_curve(args.analysis_dir)


if __name__ == "__main__":
    main()
