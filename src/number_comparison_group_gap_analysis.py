from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.number_comparison import DENSE_FIXED_GAP_WINDOWS
from src.number_comparison_fewshot_bootstrap import (
    DEFAULT_BETA_CSV,
    LABEL_TEXT,
    MODEL_LABELS,
    condition_paths,
)
from src.number_comparison import load_beta_table


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

LABEL_OFFSETS = {
    "Falcon-RW-1B": (4, -10),
    "Falcon-RW-7B": (4, 5),
    "RedPajama-3B": (4, -9),
    "RedPajama-7B": (4, 5),
    "OLMo-7B-2T": (4, 6),
    "OLMo-7B-Twin-2T": (4, -10),
    "StarCoderBase-1B": (4, -10),
    "StarCoderBase-3B": (4, 5),
    "StarCoderBase-7B": (4, 5),
}


GAP_REGIMES = {
    "SG": range(1, 6),
    "MG": range(6, 11),
    "LG": range(11, 21),
}


def _read_rows(path: Path, shots: int, exemplar_set_id: int) -> list[dict[str, str]]:
    rows_path = path / "number_comparison_rows.csv"
    with open(rows_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row.setdefault("n_shots", str(shots))
        row.setdefault("exemplar_set_id", str(exemplar_set_id))
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _corr(beta: np.ndarray, acc: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(beta) & np.isfinite(acc)
    if int(ok.sum()) < 3:
        return {"rho_s": float("nan"), "r": float("nan"), "r2": float("nan")}
    rho_s = float(spearmanr(beta[ok], acc[ok]).statistic)
    r = float(pearsonr(beta[ok], acc[ok]).statistic)
    return {"rho_s": rho_s, "r": r, "r2": r * r}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


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


def collect(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    beta_by_model = load_beta_table(args.beta_csv)
    model_labels = [label for label in MODEL_LABELS if label in beta_by_model]
    model_set = set(model_labels)

    set_cell: dict[tuple[int, int, int, int, str], list[float]] = defaultdict(list)
    conditions = condition_paths(
        args.sweep_dir,
        args.zero_shot_dir,
        args.two_shot_set0_dir,
        args.four_shot_set0_dir,
    )
    for condition in conditions:
        for row in _read_rows(condition.path, condition.shots, condition.exemplar_set_id):
            label = row["model_label"]
            if label not in model_set:
                continue
            key = (
                condition.shots,
                condition.exemplar_set_id,
                int(row["group"]),
                int(row["gap"]),
                label,
            )
            set_cell[key].append(float(row["correct"]))

    # First average within each exemplar set.
    set_rows: list[dict[str, object]] = []
    for (shots, set_id, group, gap, label), values in sorted(set_cell.items()):
        set_rows.append(
            {
                "shots": shots,
                "exemplar_set_id": set_id,
                "group": group,
                "gap": gap,
                "model_label": label,
                "pca_beta": beta_by_model[label],
                "accuracy": _mean(values),
                "n_tasks": len(values),
            }
        )

    # Then average across exemplar sets for model-level cells.
    grouped: dict[tuple[int, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in set_rows:
        grouped[(int(row["shots"]), int(row["group"]), int(row["gap"]), str(row["model_label"]))].append(row)

    model_cell_rows: list[dict[str, object]] = []
    for (shots, group, gap, label), rows in sorted(grouped.items()):
        accs = [float(row["accuracy"]) for row in rows]
        n_tasks = sum(int(row["n_tasks"]) for row in rows)
        model_cell_rows.append(
            {
                "shots": shots,
                "group": group,
                "gap": gap,
                "model_label": label,
                "pca_beta": beta_by_model[label],
                "accuracy_mean": _mean(accs),
                "accuracy_std_across_sets": _std(accs),
                "n_exemplar_sets": len(rows),
                "n_tasks_total": n_tasks,
            }
        )

    corr_rows: list[dict[str, object]] = []
    by_cell: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    by_group: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_gap: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in model_cell_rows:
        shots = int(row["shots"])
        group = int(row["group"])
        gap = int(row["gap"])
        label = str(row["model_label"])
        by_cell[(shots, group, gap)].append(row)
        by_group[(shots, group)][label].append(float(row["accuracy_mean"]))
        by_gap[(shots, gap)][label].append(float(row["accuracy_mean"]))

    for (shots, group, gap), rows in sorted(by_cell.items()):
        beta = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
        acc = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
        point = _corr(beta, acc)
        corr_rows.append(
            {
                "level": "group_gap",
                "shots": shots,
                "group": group,
                "gap": gap,
                "rho_s": point["rho_s"],
                "r": point["r"],
                "r2": point["r2"],
                "mean_accuracy": float(np.mean(acc)),
                "n_models": len(rows),
            }
        )

    for (shots, group), by_model in sorted(by_group.items()):
        labels = [label for label in model_labels if label in by_model]
        beta = np.asarray([beta_by_model[label] for label in labels], dtype=float)
        acc = np.asarray([_mean(by_model[label]) for label in labels], dtype=float)
        point = _corr(beta, acc)
        corr_rows.append(
            {
                "level": "group",
                "shots": shots,
                "group": group,
                "gap": "",
                "rho_s": point["rho_s"],
                "r": point["r"],
                "r2": point["r2"],
                "mean_accuracy": float(np.mean(acc)),
                "n_models": len(labels),
            }
        )

    for (shots, gap), by_model in sorted(by_gap.items()):
        labels = [label for label in model_labels if label in by_model]
        beta = np.asarray([beta_by_model[label] for label in labels], dtype=float)
        acc = np.asarray([_mean(by_model[label]) for label in labels], dtype=float)
        point = _corr(beta, acc)
        corr_rows.append(
            {
                "level": "gap",
                "shots": shots,
                "group": "",
                "gap": gap,
                "rho_s": point["rho_s"],
                "r": point["r"],
                "r2": point["r2"],
                "mean_accuracy": float(np.mean(acc)),
                "n_models": len(labels),
            }
        )

    return set_rows, model_cell_rows, corr_rows


def _matrix(rows: list[dict[str, object]], shots: int, value_key: str) -> np.ndarray:
    groups = sorted(DENSE_FIXED_GAP_WINDOWS)
    gaps = list(range(1, 21))
    arr = np.full((len(groups), len(gaps)), np.nan, dtype=float)
    for row in rows:
        if int(row["shots"]) != shots:
            continue
        group = row["group"]
        gap = row["gap"]
        if group == "" or gap == "":
            continue
        group_i = groups.index(int(group))
        gap_i = gaps.index(int(gap))
        arr[group_i, gap_i] = float(row[value_key])
    return arr


def plot_heatmaps(outdir: Path, corr_rows: list[dict[str, object]]) -> None:
    _setup_style()
    group_gap_rows = [row for row in corr_rows if row["level"] == "group_gap"]

    for value_key, label, filename, vmin, vmax, cmap in [
        ("mean_accuracy", "mean accuracy", "group_gap_mean_accuracy", 0.45, 0.95, "viridis"),
        ("rho_s", r"Spearman $\rho_s(\beta,\mathrm{accuracy})$", "group_gap_beta_spearman", -0.6, 1.0, "BrBG"),
    ]:
        fig, axes = plt.subplots(1, 5, figsize=(13.8, 2.8), sharey=True)
        im = None
        for ax, shots in zip(axes, range(5)):
            mat = _matrix(group_gap_rows, shots, value_key)
            im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(f"{shots}-shot")
            ax.set_xlabel("gap")
            ax.set_xticks([0, 4, 9, 14, 19])
            ax.set_xticklabels(["1", "5", "10", "15", "20"])
            ax.tick_params(direction="in", top=True, right=True)
            if ax is axes[0]:
                ax.set_ylabel("number group")
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels([r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"])
        assert im is not None
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
        cbar.set_label(label)
        fig.savefig(outdir / f"{filename}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"{filename}.png", dpi=320, bbox_inches="tight")
        plt.close(fig)


def plot_group_gap_lines(outdir: Path, corr_rows: list[dict[str, object]]) -> None:
    _setup_style()
    group_gap_rows = [row for row in corr_rows if row["level"] == "group_gap"]
    colors = {1: "#2b6f8a", 2: "#b66a3c", 3: "#5f8d4e", 4: "#7157a4"}
    fig, axes = plt.subplots(1, 5, figsize=(13.8, 2.8), sharey=True)
    for ax, shots in zip(axes, range(5)):
        for group in sorted(DENSE_FIXED_GAP_WINDOWS):
            rows = sorted(
                [
                    row
                    for row in group_gap_rows
                    if int(row["shots"]) == shots and int(row["group"]) == group
                ],
                key=lambda row: int(row["gap"]),
            )
            xs = np.asarray([int(row["gap"]) for row in rows], dtype=float)
            ys = np.asarray([float(row["mean_accuracy"]) for row in rows], dtype=float)
            ax.plot(xs, ys, marker="o", markersize=2.4, linewidth=0.85, color=colors[group], label=rf"$10^{group}$")
        ax.set_title(f"{shots}-shot")
        ax.set_xlabel("gap")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel("mean accuracy")
    axes[-1].legend(frameon=False, loc="lower right", fontsize=7)
    fig.savefig(outdir / "group_accuracy_by_gap_lines.pdf", bbox_inches="tight")
    fig.savefig(outdir / "group_accuracy_by_gap_lines.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _format_p(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value < 0.001:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _anchor_label(group: int) -> str:
    return {
        1: r"$10^1$ group",
        2: r"$10^2$ group",
        3: r"$10^3$ group",
        4: r"$10^4$ group",
    }[group]


def _gap_regime_rows(
    model_cell_rows: list[dict[str, object]],
    *,
    shots: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in sorted(DENSE_FIXED_GAP_WINDOWS):
        for regime, gaps in GAP_REGIMES.items():
            by_model: dict[str, dict[str, object]] = {}
            for row in model_cell_rows:
                if int(row["group"]) != group:
                    continue
                if int(row["gap"]) not in gaps:
                    continue
                if shots is not None and int(row["shots"]) != shots:
                    continue
                label = str(row["model_label"])
                entry = by_model.setdefault(
                    label,
                    {
                        "pca_beta": float(row["pca_beta"]),
                        "acc": [],
                        "weight": [],
                    },
                )
                entry["acc"].append(float(row["accuracy_mean"]))
                entry["weight"].append(float(row["n_tasks_total"]))

            labels = sorted(by_model)
            beta = np.asarray([float(by_model[label]["pca_beta"]) for label in labels], dtype=float)
            acc = np.asarray(
                [
                    float(np.average(by_model[label]["acc"], weights=by_model[label]["weight"]))
                    for label in labels
                ],
                dtype=float,
            )
            if len(labels) >= 3:
                spearman = spearmanr(beta, acc)
                pearson = pearsonr(beta, acc)
                rho_s = float(spearman.statistic)
                p_value = float(spearman.pvalue)
                pearson_r = float(pearson.statistic)
                pearson_p = float(pearson.pvalue)
            else:
                rho_s = float("nan")
                p_value = float("nan")
                pearson_r = float("nan")
                pearson_p = float("nan")
            rows.append(
                {
                    "shots": "all" if shots is None else shots,
                    "anchor_category": _anchor_label(group),
                    "group": group,
                    "gap_regime": regime,
                    "gap_range": f"{min(gaps)}-{max(gaps)}",
                    "spearman_rho_s": rho_s,
                    "spearman_p_value": p_value,
                    "pearson_r": pearson_r,
                    "pearson_p_value": pearson_p,
                    "mean_accuracy": float(np.mean(acc)) if len(acc) else float("nan"),
                    "n_models": len(labels),
                }
            )
    return rows


def write_gap_regime_table(outdir: Path, rows: list[dict[str, object]], *, filename: str, caption_scope: str) -> None:
    _write_csv(outdir / f"{filename}.csv", rows)
    with open(outdir / f"{filename}.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\begin{tabular}{@{}llcc@{}}\n")
        fh.write("        \\toprule\n")
        fh.write("        Anchor category & Gap regime & Spearman $\\rho_s$ & $p$-value \\\\\n")
        fh.write("        \\midrule\n")
        previous_group = None
        for row in rows:
            group = int(row["group"])
            if previous_group is not None and group != previous_group:
                fh.write("        \\midrule\n")
            previous_group = group
            fh.write(
                f"        {row['anchor_category']} & {row['gap_regime']} & "
                f"{float(row['spearman_rho_s']):.3f} & {_format_p(float(row['spearman_p_value']))} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}\n")
        fh.write(
            "    \\caption{Spearman rank correlation between PCA $\\beta$ and "
            f"scenario-level number-comparison accuracy ({caption_scope}). "
            "Each row computes the correlation across the nine evaluated models "
            "for one number-group/gap-regime scenario. SG = gaps 1--5, "
            "MG = gaps 6--10, and LG = gaps 11--20.}\n"
        )
        fh.write(f"    \\label{{tab:{filename.replace('_', '-')}}}\n")
        fh.write("\\end{table}\n")


def write_gap_regime_outputs(outdir: Path, model_cell_rows: list[dict[str, object]]) -> None:
    all_shot_rows: list[dict[str, object]] = []
    for shots in range(5):
        all_shot_rows.extend(_gap_regime_rows(model_cell_rows, shots=shots))
    _write_csv(outdir / "gap_regime_spearman_by_shot.csv", all_shot_rows)

    write_gap_regime_table(
        outdir,
        _gap_regime_rows(model_cell_rows, shots=None),
        filename="gap_regime_spearman_all_shots",
        caption_scope="task-weighted across all shot conditions",
    )
    write_gap_regime_table(
        outdir,
        _gap_regime_rows(model_cell_rows, shots=4),
        filename="gap_regime_spearman_4shot",
        caption_scope="4-shot condition",
    )


def _best_gap_by_group(model_cell_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    for group in sorted(DENSE_FIXED_GAP_WINDOWS):
        candidates: list[dict[str, object]] = []
        for gap in range(1, 21):
            by_model: dict[str, dict[str, object]] = {}
            for row in model_cell_rows:
                if int(row["group"]) != group or int(row["gap"]) != gap:
                    continue
                label = str(row["model_label"])
                entry = by_model.setdefault(
                    label,
                    {"pca_beta": float(row["pca_beta"]), "acc": [], "weight": []},
                )
                entry["acc"].append(float(row["accuracy_mean"]))
                entry["weight"].append(float(row["n_tasks_total"]))
            labels = sorted(by_model)
            beta = np.asarray([float(by_model[label]["pca_beta"]) for label in labels], dtype=float)
            acc = np.asarray(
                [
                    float(np.average(by_model[label]["acc"], weights=by_model[label]["weight"]))
                    for label in labels
                ],
                dtype=float,
            )
            if len(labels) < 3:
                continue
            result = pearsonr(beta, acc)
            candidates.append(
                {
                    "group": group,
                    "gap": gap,
                    "pooled_pearson_r": float(result.statistic),
                    "pooled_pearson_p": float(result.pvalue),
                    "mean_accuracy": float(np.mean(acc)),
                    "n_models": len(labels),
                }
            )
        candidates.sort(key=lambda row: (float(row["pooled_pearson_r"]), -int(row["gap"])), reverse=True)
        best_rows.append(candidates[0])
    return best_rows


def plot_best_gap_beta_accuracy(
    outdir: Path,
    model_cell_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
) -> None:
    _setup_style()
    selected = {(int(row["group"]), int(row["gap"])) for row in best_rows}
    selected_label = ", ".join(
        rf"$10^{int(row['group'])}$: gap {int(row['gap'])}" for row in best_rows
    )

    plot_rows: list[dict[str, object]] = []
    for shots in range(5):
        for label in MODEL_LABELS:
            cells = [
                row
                for row in model_cell_rows
                if int(row["shots"]) == shots
                and str(row["model_label"]) == label
                and (int(row["group"]), int(row["gap"])) in selected
            ]
            if len(cells) != len(selected):
                continue
            plot_rows.append(
                {
                    "shots": shots,
                    "model_label": label,
                    "pca_beta": float(cells[0]["pca_beta"]),
                    "accuracy": float(np.mean([float(row["accuracy_mean"]) for row in cells])),
                }
            )

    _write_csv(outdir / "best_pearson_gap_model_accuracy_by_shot.csv", plot_rows)

    fig, axes = plt.subplots(1, 5, figsize=(13.8, 2.85), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in plot_rows if int(row["shots"]) == shots]
        xs = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
        ys = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
        rho = float(spearmanr(xs, ys).statistic)
        r = float(pearsonr(xs, ys).statistic)
        for row in rows:
            label = str(row["model_label"])
            style = FAMILY_STYLE[FAMILY[label]]
            x = float(row["pca_beta"])
            y = float(row["accuracy"])
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
            dx, dy = LABEL_OFFSETS[label]
            ha = "left"
            if x > 9.5:
                dx = -4
                ha = "right"
            if y < 0.54:
                dy = 8
            elif y > 0.94:
                dy = -8
            ax.annotate(
                LABEL_TEXT[label],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.8,
                ha=ha,
                va="center",
                color="#222222",
                zorder=4,
            )
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(0.45, 11.1, 200)
        ax.plot(xx, coef[0] * xx + coef[1], color="#8b5a44", linewidth=1.25, zorder=2)
        ax.set_xlim(0.35, 11.35)
        ax.set_ylim(0.48, 1.02)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True, color="#d5d5d5", linewidth=0.45, alpha=0.65)
        ax.tick_params(direction="in", top=True, right=True, length=3.5)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
        ax.minorticks_on()
        ax.set_title(rf"{shots}-shot: $\rho_s={rho:.2f}$, $r={r:.2f}$", pad=5)
        ax.set_xlabel(r"PCA $\beta$")
        if shots == 0:
            ax.set_ylabel("accuracy")
        else:
            ax.set_yticklabels([])

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
    fig.text(
        0.5,
        0.01,
        f"Selected by pooled Pearson r across all shot conditions: {selected_label}",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.78, wspace=0.18)
    fig.savefig(outdir / "best_pearson_gap_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "best_pearson_gap_beta_accuracy_by_shot.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def write_best_gap_outputs(outdir: Path, model_cell_rows: list[dict[str, object]]) -> None:
    best_rows = _best_gap_by_group(model_cell_rows)
    _write_csv(outdir / "best_pearson_gap_selection.csv", best_rows)
    plot_best_gap_beta_accuracy(outdir, model_cell_rows, best_rows)


def _best_regime_by_group(model_cell_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    for group in sorted(DENSE_FIXED_GAP_WINDOWS):
        candidates: list[dict[str, object]] = []
        for regime, gaps in GAP_REGIMES.items():
            by_model: dict[str, dict[str, object]] = {}
            for row in model_cell_rows:
                if int(row["group"]) != group or int(row["gap"]) not in gaps:
                    continue
                label = str(row["model_label"])
                entry = by_model.setdefault(
                    label,
                    {"pca_beta": float(row["pca_beta"]), "acc": [], "weight": []},
                )
                entry["acc"].append(float(row["accuracy_mean"]))
                entry["weight"].append(float(row["n_tasks_total"]))
            labels = sorted(by_model)
            beta = np.asarray([float(by_model[label]["pca_beta"]) for label in labels], dtype=float)
            acc = np.asarray(
                [
                    float(np.average(by_model[label]["acc"], weights=by_model[label]["weight"]))
                    for label in labels
                ],
                dtype=float,
            )
            if len(labels) < 3:
                continue
            pearson = pearsonr(beta, acc)
            spearman = spearmanr(beta, acc)
            candidates.append(
                {
                    "group": group,
                    "gap_regime": regime,
                    "gap_range": f"{min(gaps)}-{max(gaps)}",
                    "pooled_pearson_r": float(pearson.statistic),
                    "pooled_pearson_p": float(pearson.pvalue),
                    "pooled_spearman_rho_s": float(spearman.statistic),
                    "pooled_spearman_p": float(spearman.pvalue),
                    "mean_accuracy": float(np.mean(acc)),
                    "n_models": len(labels),
                }
            )
        candidates.sort(key=lambda row: float(row["pooled_pearson_r"]), reverse=True)
        best_rows.append(candidates[0])
    return best_rows


def _regime_accuracy_rows(
    model_cell_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected = {
        int(row["group"]): set(GAP_REGIMES[str(row["gap_regime"])])
        for row in selection_rows
    }
    selected_regime = {
        int(row["group"]): str(row["gap_regime"])
        for row in selection_rows
    }
    plot_rows: list[dict[str, object]] = []
    for shots in range(5):
        for label in MODEL_LABELS:
            by_group: list[float] = []
            for group, gaps in selected.items():
                cells = [
                    row
                    for row in model_cell_rows
                    if int(row["shots"]) == shots
                    and str(row["model_label"]) == label
                    and int(row["group"]) == group
                    and int(row["gap"]) in gaps
                ]
                if not cells:
                    continue
                by_group.append(
                    float(
                        np.average(
                            [float(row["accuracy_mean"]) for row in cells],
                            weights=[float(row["n_tasks_total"]) for row in cells],
                        )
                    )
                )
            if len(by_group) != len(selected):
                continue
            model_cells = [
                row
                for row in model_cell_rows
                if int(row["shots"]) == shots and str(row["model_label"]) == label
            ]
            plot_rows.append(
                {
                    "shots": shots,
                    "model_label": label,
                    "pca_beta": float(model_cells[0]["pca_beta"]),
                    "accuracy": float(np.mean(by_group)),
                    "selected_regimes": "; ".join(
                        f"10^{group}:{selected_regime[group]}" for group in sorted(selected)
                    ),
                }
            )
    return plot_rows


def plot_best_regime_beta_accuracy(
    outdir: Path,
    model_cell_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
) -> None:
    _setup_style()
    plot_rows = _regime_accuracy_rows(model_cell_rows, best_rows)
    _write_csv(outdir / "best_pearson_gap_regime_model_accuracy_by_shot.csv", plot_rows)

    selected_label = ", ".join(
        rf"$10^{int(row['group'])}$: {row['gap_regime']}" for row in best_rows
    )
    fig, axes = plt.subplots(1, 5, figsize=(13.8, 2.85), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in plot_rows if int(row["shots"]) == shots]
        xs = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
        ys = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
        rho = float(spearmanr(xs, ys).statistic)
        r = float(pearsonr(xs, ys).statistic)
        for row in rows:
            label = str(row["model_label"])
            style = FAMILY_STYLE[FAMILY[label]]
            x = float(row["pca_beta"])
            y = float(row["accuracy"])
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
            dx, dy = LABEL_OFFSETS[label]
            ha = "left"
            if x > 9.5:
                dx = -4
                ha = "right"
            if y < 0.54:
                dy = 8
            elif y > 0.94:
                dy = -8
            ax.annotate(
                LABEL_TEXT[label],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.8,
                ha=ha,
                va="center",
                color="#222222",
                zorder=4,
            )
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(0.45, 11.1, 200)
        ax.plot(xx, coef[0] * xx + coef[1], color="#8b5a44", linewidth=1.25, zorder=2)
        ax.set_xlim(0.35, 11.35)
        ax.set_ylim(0.48, 1.02)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True, color="#d5d5d5", linewidth=0.45, alpha=0.65)
        ax.tick_params(direction="in", top=True, right=True, length=3.5)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
        ax.minorticks_on()
        ax.set_title(rf"{shots}-shot: $\rho_s={rho:.2f}$, $r={r:.2f}$", pad=5)
        ax.set_xlabel(r"PCA $\beta$")
        if shots == 0:
            ax.set_ylabel("accuracy")
        else:
            ax.set_yticklabels([])

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
    fig.text(
        0.5,
        0.01,
        f"Selected by pooled Pearson r across all shot conditions: {selected_label}",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.78, wspace=0.18)
    fig.savefig(outdir / "best_pearson_gap_regime_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "best_pearson_gap_regime_beta_accuracy_by_shot.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def write_best_regime_outputs(outdir: Path, model_cell_rows: list[dict[str, object]]) -> None:
    regime_rows = []
    for shots in range(5):
        regime_rows.extend(_gap_regime_rows(model_cell_rows, shots=shots))
    _write_csv(outdir / "gap_regime_correlations_by_shot.csv", regime_rows)

    best_rows = _best_regime_by_group(model_cell_rows)
    _write_csv(outdir / "best_pearson_gap_regime_selection.csv", best_rows)
    plot_best_regime_beta_accuracy(outdir, model_cell_rows, best_rows)


def _fixed_sg_mg_accuracy_rows(
    model_cell_rows: list[dict[str, object]],
    groups: list[int] | None = None,
    selected_label: str = "all groups: SG+MG",
) -> list[dict[str, object]]:
    plot_rows: list[dict[str, object]] = []
    selected_regimes = ("SG", "MG")
    selected_groups = sorted(DENSE_FIXED_GAP_WINDOWS) if groups is None else groups
    expected_scenarios = len(selected_groups) * len(selected_regimes)
    for shots in range(5):
        for label in MODEL_LABELS:
            scenario_accs: list[float] = []
            for group in selected_groups:
                for regime in selected_regimes:
                    gaps = GAP_REGIMES[regime]
                    cells = [
                        row
                        for row in model_cell_rows
                        if int(row["shots"]) == shots
                        and str(row["model_label"]) == label
                        and int(row["group"]) == group
                        and int(row["gap"]) in gaps
                    ]
                    if not cells:
                        continue
                    scenario_accs.append(
                        float(
                            np.average(
                                [float(row["accuracy_mean"]) for row in cells],
                                weights=[float(row["n_tasks_total"]) for row in cells],
                            )
                        )
                    )
            if len(scenario_accs) != expected_scenarios:
                continue
            model_cells = [
                row
                for row in model_cell_rows
                if int(row["shots"]) == shots and str(row["model_label"]) == label
            ]
            plot_rows.append(
                {
                    "shots": shots,
                    "model_label": label,
                    "pca_beta": float(model_cells[0]["pca_beta"]),
                    "accuracy": float(np.mean(scenario_accs)),
                    "n_scenarios": len(scenario_accs),
                    "selected_regimes": selected_label,
                }
            )
    return plot_rows


def plot_fixed_sg_mg_beta_accuracy(
    outdir: Path,
    model_cell_rows: list[dict[str, object]],
    output_prefix: str = "sg_mg",
    groups: list[int] | None = None,
    selected_label: str = "all groups: SG+MG",
) -> None:
    _setup_style()
    selected_groups = sorted(DENSE_FIXED_GAP_WINDOWS) if groups is None else groups
    expected_scenarios = len(selected_groups) * 2
    plot_rows = _fixed_sg_mg_accuracy_rows(model_cell_rows, selected_groups, selected_label)
    _write_csv(outdir / f"{output_prefix}_model_accuracy_by_shot.csv", plot_rows)

    summary_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 5, figsize=(13.8, 2.85), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in plot_rows if int(row["shots"]) == shots]
        xs = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
        ys = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
        spearman = spearmanr(xs, ys)
        pearson = pearsonr(xs, ys)
        rho = float(spearman.statistic)
        r = float(pearson.statistic)
        summary_rows.append(
            {
                "shots": shots,
                "spearman_rho_s": rho,
                "spearman_p_value": float(spearman.pvalue),
                "pearson_r": r,
                "pearson_p_value": float(pearson.pvalue),
                "mean_accuracy": float(np.mean(ys)),
                "n_models": len(rows),
                "n_scenarios_per_model": expected_scenarios,
            }
        )
        for row in rows:
            label = str(row["model_label"])
            style = FAMILY_STYLE[FAMILY[label]]
            x = float(row["pca_beta"])
            y = float(row["accuracy"])
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
            dx, dy = LABEL_OFFSETS[label]
            ha = "left"
            if x > 9.5:
                dx = -4
                ha = "right"
            if y < 0.54:
                dy = 8
            elif y > 0.94:
                dy = -8
            ax.annotate(
                LABEL_TEXT[label],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.8,
                ha=ha,
                va="center",
                color="#222222",
                zorder=4,
            )
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(0.45, 11.1, 200)
        ax.plot(xx, coef[0] * xx + coef[1], color="#8b5a44", linewidth=1.25, zorder=2)
        ax.set_xlim(0.35, 11.35)
        ax.set_ylim(0.48, 1.02)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True, color="#d5d5d5", linewidth=0.45, alpha=0.65)
        ax.tick_params(direction="in", top=True, right=True, length=3.5)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
        ax.minorticks_on()
        ax.set_title(rf"{shots}-shot: $\rho_s={rho:.2f}$, $r={r:.2f}$", pad=5)
        ax.set_xlabel(r"PCA $\beta$")
        if shots == 0:
            ax.set_ylabel("accuracy")
        else:
            ax.set_yticklabels([])

    _write_csv(outdir / f"{output_prefix}_beta_accuracy_by_shot_summary.csv", summary_rows)
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
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.2, top=0.78, wspace=0.18)
    fig.savefig(outdir / f"{output_prefix}_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{output_prefix}_beta_accuracy_by_shot.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=Path("results/number_comparison/fixed_gap_dense_fewshot_exemplar_sweep"))
    parser.add_argument("--zero-shot-dir", type=Path, default=Path("results/number_comparison/fixed_gap_dense_number_answer"))
    parser.add_argument("--two-shot-set0-dir", type=Path, default=Path("results/number_comparison/fixed_gap_dense_number_answer_2shot"))
    parser.add_argument("--four-shot-set0-dir", type=Path, default=Path("results/number_comparison/fixed_gap_dense_number_answer_4shot"))
    parser.add_argument("--beta-csv", type=Path, default=DEFAULT_BETA_CSV)
    parser.add_argument("--outdir", type=Path, default=Path("results/number_comparison/fixed_gap_fewshot_exemplar_bootstrap_analysis/group_gap_analysis"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_rows, model_cell_rows, corr_rows = collect(args)
    _write_csv(args.outdir / "group_gap_accuracy_by_set.csv", set_rows)
    _write_csv(args.outdir / "group_gap_model_accuracy_by_shot.csv", model_cell_rows)
    _write_csv(args.outdir / "group_gap_correlation_by_shot.csv", corr_rows)
    write_gap_regime_outputs(args.outdir, model_cell_rows)
    write_best_gap_outputs(args.outdir, model_cell_rows)
    write_best_regime_outputs(args.outdir, model_cell_rows)
    plot_fixed_sg_mg_beta_accuracy(args.outdir, model_cell_rows)
    plot_fixed_sg_mg_beta_accuracy(
        args.outdir,
        model_cell_rows,
        output_prefix="sg_mg_no_10e1",
        groups=[2, 3, 4],
        selected_label="groups 10^2-10^4: SG+MG",
    )
    plot_heatmaps(args.outdir, corr_rows)
    plot_group_gap_lines(args.outdir, corr_rows)
    print(f"wrote {args.outdir}")


if __name__ == "__main__":
    main()
