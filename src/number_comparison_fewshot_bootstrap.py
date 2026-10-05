from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.number_comparison import load_beta_table


DEFAULT_BETA_CSV = Path(
    "results/dataset_frequency_0_to_10000/alpha_beta_without_pile/"
    "alpha_beta_models_without_pile.csv"
)

MODEL_LABELS = [
    "Falcon-RW-1B",
    "Falcon-RW-7B",
    "RedPajama-3B",
    "RedPajama-7B",
    "OLMo-7B-2T",
    "OLMo-7B-Twin-2T",
    "StarCoderBase-1B",
    "StarCoderBase-3B",
    "StarCoderBase-7B",
    "OpenLLaMA-3B",
    "OpenLLaMA-7B",
    "OpenLLaMA-13B",
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
    "OpenLLaMA-3B": "OpenLLaMA-3B",
    "OpenLLaMA-7B": "OpenLLaMA-7B",
    "OpenLLaMA-13B": "OpenLLaMA-13B",
}


@dataclass(frozen=True)
class Condition:
    shots: int
    exemplar_set_id: int
    path: Path


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Iterable[float]) -> float:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = _finite(values)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _format_mean_std_ci(mean: float, std: float, ci_low: float, ci_high: float) -> str:
    if not all(math.isfinite(value) for value in (mean, std, ci_low, ci_high)):
        return "--"
    return rf"{mean:.2f}\,$\pm$\,{std:.2f} [{ci_low:.2f}, {ci_high:.2f}]"


def condition_paths(
    sweep_dir: Path,
    zero_shot_dir: Path,
    two_shot_set0_dir: Path,
    four_shot_set0_dir: Path,
) -> list[Condition]:
    conditions = [
        Condition(0, 0, zero_shot_dir),
        Condition(0, 1, sweep_dir / "0shot_set1"),
        Condition(0, 2, sweep_dir / "0shot_set2"),
        Condition(1, 0, sweep_dir / "1shot_set0"),
        Condition(2, 0, two_shot_set0_dir),
        Condition(3, 0, sweep_dir / "3shot_set0"),
        Condition(4, 0, four_shot_set0_dir),
    ]
    for exemplar_set_id in (1, 2):
        for shots in (1, 2, 3, 4):
            conditions.append(
                Condition(shots, exemplar_set_id, sweep_dir / f"{shots}shot_set{exemplar_set_id}")
            )
    return conditions


def load_condition_matrix(path: Path, model_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    rows_path = path / "number_comparison_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)

    rows: list[dict[str, str]] = []
    with open(rows_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("model_label") in model_labels:
                rows.append(row)
    if not rows:
        raise ValueError(f"no rows for expected models in {rows_path}")

    task_ids = sorted({row["task_id"] for row in rows})
    task_index = {task_id: idx for idx, task_id in enumerate(task_ids)}
    model_index = {label: idx for idx, label in enumerate(model_labels)}
    matrix = np.full((len(model_labels), len(task_ids)), np.nan, dtype=float)

    for row in rows:
        model_label = row["model_label"]
        task_id = row["task_id"]
        matrix[model_index[model_label], task_index[task_id]] = float(row["correct"])

    missing = np.argwhere(~np.isfinite(matrix))
    if missing.size:
        first_model, first_task = missing[0]
        raise ValueError(
            f"missing correctness values in {rows_path}; first missing "
            f"model={model_labels[int(first_model)]} task={task_ids[int(first_task)]}"
        )
    return matrix, task_ids


def correlation_metrics(beta: np.ndarray, model_accuracy: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(beta) & np.isfinite(model_accuracy)
    if int(ok.sum()) < 2:
        return {
            "rho_s": float("nan"),
            "r": float("nan"),
            "r2": float("nan"),
            "mean_accuracy": _mean(model_accuracy),
        }
    rho_s = float(spearmanr(beta[ok], model_accuracy[ok]).statistic)
    r = float(pearsonr(beta[ok], model_accuracy[ok]).statistic)
    return {
        "rho_s": rho_s,
        "r": r,
        "r2": r * r if math.isfinite(r) else float("nan"),
        "mean_accuracy": float(np.mean(model_accuracy[ok])),
    }


def bootstrap_ci(
    matrices: list[np.ndarray],
    beta: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if not matrices:
        return {metric: (float("nan"), float("nan")) for metric in ("rho_s", "r", "r2", "mean_accuracy")}
    n_tasks = matrices[0].shape[1]
    if any(matrix.shape[1] != n_tasks for matrix in matrices):
        raise ValueError("all matrices for a shot must use the same task set")

    stacked = np.stack(matrices, axis=0)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "rho_s": [],
        "r": [],
        "r2": [],
        "mean_accuracy": [],
    }
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_tasks, size=n_tasks)
        # Average across fixed exemplar sets and bootstrapped comparison tasks.
        model_accuracy = stacked[:, :, sample].mean(axis=(0, 2))
        metrics = correlation_metrics(beta, model_accuracy)
        for key in values:
            values[key].append(metrics[key])

    out: dict[str, tuple[float, float]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[key] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        else:
            out[key] = (float("nan"), float("nan"))
    return out


def model_accuracy_bootstrap_ci(
    matrices: list[np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not matrices:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    n_tasks = matrices[0].shape[1]
    if any(matrix.shape[1] != n_tasks for matrix in matrices):
        raise ValueError("all matrices for a shot must use the same task set")

    stacked = np.stack(matrices, axis=0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_tasks, size=n_tasks)
        values.append(stacked[:, :, sample].mean(axis=(0, 2)))
    arr = np.asarray(values, dtype=float)
    return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_summary(outdir: Path, rows: list[dict[str, object]]) -> None:
    with open(outdir / "fewshot_summary_mean_std_ci.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\resizebox{\\linewidth}{!}{%\n")
        fh.write("    \\begin{tabular}{@{}lcccc@{}}\n")
        fh.write("        \\toprule\n")
        fh.write(
            "        Shots & $\\rho_s$ & $r$ & $R^2$ & Mean accuracy \\\\\n"
        )
        fh.write("        \\midrule\n")
        for row in rows:
            fh.write(
                f"        {int(row['shots'])} & "
                f"{_format_mean_std_ci(float(row['rho_s_mean']), float(row['rho_s_std']), float(row['rho_s_ci_low']), float(row['rho_s_ci_high']))} & "
                f"{_format_mean_std_ci(float(row['r_mean']), float(row['r_std']), float(row['r_ci_low']), float(row['r_ci_high']))} & "
                f"{_format_mean_std_ci(float(row['r2_mean']), float(row['r2_std']), float(row['r2_ci_low']), float(row['r2_ci_high']))} & "
                f"{_format_mean_std_ci(float(row['mean_accuracy_mean']), float(row['mean_accuracy_std']), float(row['mean_accuracy_ci_low']), float(row['mean_accuracy_ci_high']))} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}}\n")
        fh.write(
            "    \\caption{Few-shot fixed-gap digit-comparison results. "
            "Entries are mean $\\pm$ standard deviation across exemplar sets, "
            "with bootstrap 95\\% confidence intervals over comparison tasks in brackets. "
            "The zero-shot row has one prompt condition.}\n"
        )
        fh.write("    \\label{tab:fewshot-beta-accuracy-bootstrap}\n")
        fh.write("\\end{table}\n")


def write_spearman_summary(
    outdir: Path,
    condition_rows: list[dict[str, object]],
    shot_summary_rows: list[dict[str, object]],
) -> None:
    condition_by_key = {
        (int(row["shots"]), int(row["exemplar_set_id"])): row
        for row in condition_rows
    }
    summary_by_shot = {
        int(row["shots"]): row
        for row in shot_summary_rows
    }
    rows: list[dict[str, object]] = []
    for shots in range(5):
        summary = summary_by_shot[shots]
        row: dict[str, object] = {
            "shots": shots,
            "rho_s_mean": float(summary["rho_s_mean"]),
            "rho_s_std": float(summary["rho_s_std"]),
            "rho_s_ci_low": float(summary["rho_s_ci_low"]),
            "rho_s_ci_high": float(summary["rho_s_ci_high"]),
        }
        for set_id in range(3):
            condition = condition_by_key.get((shots, set_id))
            if condition is None:
                row[f"set{set_id}_rho_s"] = ""
                row[f"set{set_id}_ci_low"] = ""
                row[f"set{set_id}_ci_high"] = ""
            else:
                row[f"set{set_id}_rho_s"] = float(condition["rho_s"])
                row[f"set{set_id}_ci_low"] = float(condition["rho_s_ci_low"])
                row[f"set{set_id}_ci_high"] = float(condition["rho_s_ci_high"])
        rows.append(row)

    write_csv(outdir / "fewshot_spearman_by_exemplar_set.csv", rows)
    with open(outdir / "fewshot_spearman_by_exemplar_set.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\begin{tabular}{@{}lcccc@{}}\n")
        fh.write("        \\toprule\n")
        fh.write("        Shots & Set 0 & Set 1 & Set 2 & Mean $\\pm$ std \\\\\n")
        fh.write("        \\midrule\n")
        for row in rows:
            set_values = []
            for set_id in range(3):
                value = row[f"set{set_id}_rho_s"]
                set_values.append("--" if value == "" else f"{float(value):.2f}")
            fh.write(
                f"        {row['shots']} & {set_values[0]} & {set_values[1]} & "
                f"{set_values[2]} & {float(row['rho_s_mean']):.2f} "
                f"$\\pm$ {float(row['rho_s_std']):.2f} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}\n")
        fh.write(
            "    \\caption{Spearman correlation $\\rho_s$ between PCA compression "
            "factor $\\beta$ and fixed-gap comparison accuracy for each few-shot "
            "exemplar set.}\n"
        )
        fh.write("    \\label{tab:fewshot-spearman-by-exemplar-set}\n")
        fh.write("\\end{table}\n")


def plot_beta_accuracy(
    outdir: Path,
    model_summary_rows: list[dict[str, object]],
    shot_summary_rows: list[dict[str, object]],
    beta_by_model: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    rows_by_shot = {
        int(row["shots"]): row
        for row in shot_summary_rows
    }
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.3), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in model_summary_rows if int(row["shots"]) == shots]
        xs = np.asarray([beta_by_model[str(row["model_label"])] for row in rows], dtype=float)
        ys = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
        yerr = np.asarray([float(row["accuracy_std"]) for row in rows], dtype=float)
        ci_low = np.asarray([float(row["accuracy_ci_low"]) for row in rows], dtype=float)
        ci_high = np.asarray([float(row["accuracy_ci_high"]) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ci_yerr = np.vstack([ys[ok] - ci_low[ok], ci_high[ok] - ys[ok]])
        ax.errorbar(
            xs[ok],
            ys[ok],
            yerr=ci_yerr,
            fmt="none",
            ecolor="#b7b7b7",
            elinewidth=1.8,
            capsize=2.4,
            zorder=1,
        )
        ax.errorbar(
            xs[ok],
            ys[ok],
            yerr=yerr[ok],
            fmt="o",
            color="#276b83",
            ecolor="#8fb7c5",
            elinewidth=1.0,
            capsize=2.5,
            markersize=4.8,
            zorder=3,
        )
        if int(ok.sum()) >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#b84e3a", linewidth=1.2)
        for row in rows:
            x = beta_by_model[str(row["model_label"])]
            y = float(row["accuracy_mean"])
            if math.isfinite(x) and math.isfinite(y):
                ax.annotate(
                    LABEL_TEXT.get(str(row["model_label"]), str(row["model_label"])),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.8,
                )
        rho_s = float(rows_by_shot[shots]["rho_s_mean"])
        r2 = float(rows_by_shot[shots]["r2_mean"])
        ax.set_title(rf"{shots}-shot: $\rho_s={rho_s:.2f}$, $R^2={r2:.2f}$", fontsize=9)
        ax.set_xlabel(r"PCA $\beta$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("comparison accuracy")
    fig.tight_layout()
    fig.savefig(outdir / "fewshot_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fewshot_beta_accuracy_by_shot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_spearman_by_exemplar_set(
    outdir: Path,
    condition_rows: list[dict[str, object]],
    shot_summary_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        0: "#276b83",
        1: "#b84e3a",
        2: "#6f8f3a",
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for set_id in (0, 1, 2):
        rows = sorted(
            (
                row
                for row in condition_rows
                if int(row["exemplar_set_id"]) == set_id
            ),
            key=lambda row: int(row["shots"]),
        )
        if not rows:
            continue
        xs = np.asarray([int(row["shots"]) for row in rows], dtype=float)
        ys = np.asarray([float(row["rho_s"]) for row in rows], dtype=float)
        ci_low = np.asarray([float(row["rho_s_ci_low"]) for row in rows], dtype=float)
        ci_high = np.asarray([float(row["rho_s_ci_high"]) for row in rows], dtype=float)
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.3,
            markersize=4.8,
            color=colors[set_id],
            label=f"set {set_id}",
        )
        ax.fill_between(xs, ci_low, ci_high, color=colors[set_id], alpha=0.12, linewidth=0)

    summary = sorted(shot_summary_rows, key=lambda row: int(row["shots"]))
    xs = np.asarray([int(row["shots"]) for row in summary], dtype=float)
    means = np.asarray([float(row["rho_s_mean"]) for row in summary], dtype=float)
    stds = np.asarray([float(row["rho_s_std"]) for row in summary], dtype=float)
    ci_low = np.asarray([float(row["rho_s_ci_low"]) for row in summary], dtype=float)
    ci_high = np.asarray([float(row["rho_s_ci_high"]) for row in summary], dtype=float)
    ci_yerr = np.vstack([means - ci_low, ci_high - means])
    ax.errorbar(
        xs,
        means,
        yerr=ci_yerr,
        fmt="none",
        ecolor="#b7b7b7",
        elinewidth=2.0,
        capsize=4,
        label="mean bootstrap CI",
        zorder=3,
    )
    ax.errorbar(
        xs,
        means,
        yerr=stds,
        color="#111111",
        linewidth=1.7,
        marker="s",
        markersize=4.4,
        capsize=3,
        label="mean",
        zorder=4,
    )
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.55)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(0.2, 0.9)
    ax.set_xlabel("shots")
    ax.set_ylabel(r"Spearman $\rho_s$")
    ax.set_title(r"$\beta$-accuracy rank correlation by exemplar set")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(outdir / "fewshot_spearman_by_exemplar_set.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fewshot_spearman_by_exemplar_set.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_shot(outdir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    shots = np.asarray([int(row["shots"]) for row in rows], dtype=float)
    panels = [
        ("mean_accuracy", "Mean accuracy"),
        ("rho_s", r"Spearman $\rho_s$"),
        ("r", "Pearson r"),
        ("r2", r"$R^2$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))
    for ax, (metric, title) in zip(axes, panels):
        means = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        stds = np.asarray([float(row[f"{metric}_std"]) for row in rows], dtype=float)
        ci_low = np.asarray([float(row[f"{metric}_ci_low"]) for row in rows], dtype=float)
        ci_high = np.asarray([float(row[f"{metric}_ci_high"]) for row in rows], dtype=float)
        ax.fill_between(shots, ci_low, ci_high, color="#9bb8c2", alpha=0.25, linewidth=0)
        ci_yerr = np.vstack([means - ci_low, ci_high - means])
        ax.errorbar(
            shots,
            means,
            yerr=ci_yerr,
            fmt="none",
            ecolor="#b7b7b7",
            elinewidth=1.8,
            capsize=3,
            zorder=2,
        )
        ax.errorbar(
            shots,
            means,
            yerr=stds,
            marker="o",
            color="#276b83",
            ecolor="#8fb7c5",
            capsize=3,
            linewidth=1.4,
        )
        ax.set_title(title)
        ax.set_xlabel("shots")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fewshot_metrics_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fewshot_metrics_by_shot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def log_to_mlflow(outdir: Path, run_name: str) -> str | None:
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - optional local dependency
        print(f"[mlflow] skipped: {exc}")
        return None

    mlflow.set_experiment("numberline_geometry_expv1")
    with mlflow.start_run(run_name=run_name) as run:
        for path in sorted(outdir.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path))
        with open(outdir / "fewshot_summary_mean_std_ci.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            shots = int(row["shots"])
            for metric in ("rho_s", "r", "r2", "mean_accuracy"):
                value = float(row[f"{metric}_mean"])
                if math.isfinite(value):
                    mlflow.log_metric(f"{shots}shot_{metric}_mean", value)
                std = float(row[f"{metric}_std"])
                if math.isfinite(std):
                    mlflow.log_metric(f"{shots}shot_{metric}_std", std)
        return run.info.run_id


def run_analysis(args: argparse.Namespace) -> None:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    beta_by_model = load_beta_table(args.beta_csv)
    model_labels = [label for label in MODEL_LABELS if label in beta_by_model]
    beta = np.asarray([beta_by_model[label] for label in model_labels], dtype=float)
    if len(model_labels) < 2:
        raise ValueError(f"not enough model beta values in {args.beta_csv}")

    conditions = condition_paths(
        args.sweep_dir,
        args.zero_shot_dir,
        args.two_shot_set0_dir,
        args.four_shot_set0_dir,
    )

    matrices_by_condition: dict[tuple[int, int], np.ndarray] = {}
    task_ids_by_condition: dict[tuple[int, int], list[str]] = {}
    condition_rows: list[dict[str, object]] = []
    per_model_condition_rows: list[dict[str, object]] = []

    for condition in conditions:
        matrix, task_ids = load_condition_matrix(condition.path, model_labels)
        matrices_by_condition[(condition.shots, condition.exemplar_set_id)] = matrix
        task_ids_by_condition[(condition.shots, condition.exemplar_set_id)] = task_ids

        model_accuracy = matrix.mean(axis=1)
        point = correlation_metrics(beta, model_accuracy)
        ci = bootstrap_ci(
            [matrix],
            beta,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + condition.shots * 101 + condition.exemplar_set_id,
        )
        row: dict[str, object] = {
            "shots": condition.shots,
            "exemplar_set_id": condition.exemplar_set_id,
            "n_models": len(model_labels),
            "n_tasks": len(task_ids),
            "path": str(condition.path),
        }
        for metric in ("rho_s", "r", "r2", "mean_accuracy"):
            row[metric] = point[metric]
            row[f"{metric}_ci_low"] = ci[metric][0]
            row[f"{metric}_ci_high"] = ci[metric][1]
        condition_rows.append(row)

        for label, model_acc in zip(model_labels, model_accuracy):
            per_model_condition_rows.append(
                {
                    "shots": condition.shots,
                    "exemplar_set_id": condition.exemplar_set_id,
                    "model_label": label,
                    "pca_beta": beta_by_model[label],
                    "accuracy": float(model_acc),
                }
            )

    shot_summary_rows: list[dict[str, object]] = []
    shot_model_rows: list[dict[str, object]] = []
    condition_by_key = {
        (int(row["shots"]), int(row["exemplar_set_id"])): row
        for row in condition_rows
    }
    for shots in range(5):
        keys = sorted(key for key in matrices_by_condition if key[0] == shots)
        if not keys:
            continue
        matrices = [matrices_by_condition[key] for key in keys]
        pooled_model_accuracy = np.stack(matrices, axis=0).mean(axis=(0, 2))
        pooled_point = correlation_metrics(beta, pooled_model_accuracy)
        pooled_ci = bootstrap_ci(
            matrices,
            beta,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + shots * 1009,
        )
        per_set_points = [
            correlation_metrics(beta, matrices_by_condition[key].mean(axis=1))
            for key in keys
        ]
        row = {
            "shots": shots,
            "n_exemplar_sets": len(keys),
            "n_models": len(model_labels),
            "n_tasks_per_set": matrices[0].shape[1],
        }
        for metric in ("rho_s", "r", "r2", "mean_accuracy"):
            values = [point[metric] for point in per_set_points]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
            row[f"{metric}_pooled"] = pooled_point[metric]
            row[f"{metric}_ci_low"] = _mean(
                condition_by_key[key][f"{metric}_ci_low"] for key in keys
            )
            row[f"{metric}_ci_high"] = _mean(
                condition_by_key[key][f"{metric}_ci_high"] for key in keys
            )
            row[f"{metric}_pooled_ci_low"] = pooled_ci[metric][0]
            row[f"{metric}_pooled_ci_high"] = pooled_ci[metric][1]
        shot_summary_rows.append(row)

        model_ci_low, model_ci_high = model_accuracy_bootstrap_ci(
            matrices,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + shots * 2003,
        )
        for model_idx, label in enumerate(model_labels):
            accs = [float(matrix[model_idx, :].mean()) for matrix in matrices]
            shot_model_rows.append(
                {
                    "shots": shots,
                    "model_label": label,
                    "pca_beta": beta_by_model[label],
                    "accuracy_mean": _mean(accs),
                    "accuracy_std": _std(accs),
                    "accuracy_pooled": float(pooled_model_accuracy[model_idx]),
                    "accuracy_ci_low": float(model_ci_low[model_idx]),
                    "accuracy_ci_high": float(model_ci_high[model_idx]),
                    "n_exemplar_sets": len(keys),
                }
            )

    write_csv(outdir / "fewshot_exemplar_set_metrics.csv", condition_rows)
    write_csv(outdir / "fewshot_model_accuracy_by_exemplar_set.csv", per_model_condition_rows)
    write_csv(outdir / "fewshot_summary_mean_std_ci.csv", shot_summary_rows)
    write_csv(outdir / "fewshot_model_accuracy_by_shot.csv", shot_model_rows)
    write_latex_summary(outdir, shot_summary_rows)
    write_spearman_summary(outdir, condition_rows, shot_summary_rows)
    plot_beta_accuracy(outdir, shot_model_rows, shot_summary_rows, beta_by_model)
    plot_spearman_by_exemplar_set(outdir, condition_rows, shot_summary_rows)
    plot_metrics_by_shot(outdir, shot_summary_rows)

    with open(outdir / "fewshot_bootstrap_summary.json", "w") as fh:
        json.dump(
            {
                "beta_csv": str(args.beta_csv),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "model_labels": model_labels,
                "conditions": [
                    {
                        "shots": condition.shots,
                        "exemplar_set_id": condition.exemplar_set_id,
                        "path": str(condition.path),
                    }
                    for condition in conditions
                ],
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    run_id = None
    if args.log_mlflow:
        run_id = log_to_mlflow(outdir, args.run_name)
        if run_id:
            print(f"[mlflow] run_id={run_id}")

    print(f"[analysis] wrote {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate few-shot number-comparison runs with exemplar-set stds and bootstrap CIs."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("results/number_comparison/fixed_gap_dense_fewshot_exemplar_sweep"),
    )
    parser.add_argument(
        "--zero-shot-dir",
        type=Path,
        default=Path("results/number_comparison/fixed_gap_dense_number_answer"),
    )
    parser.add_argument(
        "--two-shot-set0-dir",
        type=Path,
        default=Path("results/number_comparison/fixed_gap_dense_number_answer_2shot"),
    )
    parser.add_argument(
        "--four-shot-set0-dir",
        type=Path,
        default=Path("results/number_comparison/fixed_gap_dense_number_answer_4shot"),
    )
    parser.add_argument("--beta-csv", type=Path, default=DEFAULT_BETA_CSV)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/number_comparison/fixed_gap_fewshot_exemplar_bootstrap_analysis"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--run-name",
        default="number_comparison_fewshot_exemplar_bootstrap_analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
