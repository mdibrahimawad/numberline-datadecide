from __future__ import annotations

import argparse
import csv
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


DISPLAY_LABELS = {
    "Llama-2-7B": "Llama-2",
    "Pythia-2.8B": "Pythia",
    "GPT2-L": "GPT2-L",
    "Mistral-7B": "Mistral",
    "DeepSeek-Base-7B": "DeepSeek",
    "Qwen1.5-7B": "Qwen1.5",
    "Llama-3.1-8B": "Llama-3.1",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
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


def _ci_text(mean: float, std: float, low: float, high: float) -> str:
    if not all(math.isfinite(value) for value in (mean, std, low, high)):
        return "--"
    return rf"{mean:.2f}\,$\pm$\,{std:.2f} [{low:.2f}, {high:.2f}]"


def _plot_yerr(mean: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    lower = np.maximum(mean - low, 0.0)
    upper = np.maximum(high - mean, 0.0)
    return np.vstack([lower, upper])


def read_beta_table(path: Path, beta_column: str) -> tuple[list[str], dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    labels: list[str] = []
    beta_by_model: dict[str, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            label = row.get("model_label") or row.get("model")
            status = row.get("status", "ok")
            beta = row.get(beta_column) or row.get("pca_beta_mean") or row.get("pca_beta")
            if not label or status != "ok" or beta in {None, ""}:
                continue
            value = float(beta)
            if math.isfinite(value):
                labels.append(label)
                beta_by_model[label] = value
    if len(labels) < 2:
        raise ValueError(f"not enough valid beta values in {path}")
    return labels, beta_by_model


def discover_conditions(root: Path) -> list[Condition]:
    conditions: list[Condition] = []
    for shots in range(5):
        for set_id in range(3):
            path = root / f"{shots}shot_set{set_id}"
            if (path / "number_comparison_rows.csv").exists():
                conditions.append(Condition(shots, set_id, path))
    if not conditions:
        raise FileNotFoundError(f"no shot result directories found under {root}")
    return conditions


def load_condition_arrays(
    condition: Condition,
    model_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows_path = condition.path / "number_comparison_rows.csv"
    rows: list[dict[str, str]] = []
    with open(rows_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("model_label") in model_labels:
                rows.append(row)
    if not rows:
        raise ValueError(f"no rows for requested models in {rows_path}")

    task_ids = sorted({row["task_id"] for row in rows})
    task_index = {task_id: idx for idx, task_id in enumerate(task_ids)}
    model_index = {label: idx for idx, label in enumerate(model_labels)}
    correct = np.full((len(model_labels), len(task_ids)), np.nan, dtype=float)
    margin = np.full((len(model_labels), len(task_ids)), np.nan, dtype=float)

    for row in rows:
        model = row["model_label"]
        task = row["task_id"]
        correct[model_index[model], task_index[task]] = float(row["correct"])
        margin[model_index[model], task_index[task]] = float(row.get("margin") or "nan")

    for name, matrix in (("correct", correct), ("margin", margin)):
        missing = np.argwhere(~np.isfinite(matrix))
        if missing.size:
            model_idx, task_idx = missing[0]
            raise ValueError(
                f"missing {name} values in {rows_path}; first missing "
                f"model={model_labels[int(model_idx)]} task={task_ids[int(task_idx)]}"
            )
    return correct, margin, task_ids


def correlation_metrics(beta: np.ndarray, model_values: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(beta) & np.isfinite(model_values)
    if int(ok.sum()) < 2:
        return {"rho_s": float("nan"), "r": float("nan"), "r2": float("nan")}
    rho_s = float(spearmanr(beta[ok], model_values[ok]).statistic)
    r = float(pearsonr(beta[ok], model_values[ok]).statistic)
    return {"rho_s": rho_s, "r": r, "r2": r * r if math.isfinite(r) else float("nan")}


def bootstrap_metric_ci(
    matrices: list[np.ndarray],
    beta: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if not matrices:
        return {key: (float("nan"), float("nan")) for key in ("rho_s", "r", "r2", "mean_value")}
    n_tasks = matrices[0].shape[1]
    if any(matrix.shape[1] != n_tasks for matrix in matrices):
        raise ValueError("all matrices for a shot must use the same task set")

    stacked = np.stack(matrices, axis=0)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {key: [] for key in ("rho_s", "r", "r2", "mean_value")}
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_tasks, size=n_tasks)
        # Average over exemplar sets and bootstrapped task samples.
        model_values = stacked[:, :, sample].mean(axis=(0, 2))
        metrics = correlation_metrics(beta, model_values)
        for key in ("rho_s", "r", "r2"):
            values[key].append(metrics[key])
        values["mean_value"].append(float(np.mean(model_values)))

    out: dict[str, tuple[float, float]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[key] = (
            (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
            if arr.size
            else (float("nan"), float("nan"))
        )
    return out


def model_bootstrap_ci(
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
    samples = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_tasks, size=n_tasks)
        samples.append(stacked[:, :, sample].mean(axis=(0, 2)))
    arr = np.asarray(samples, dtype=float)
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
    with open(outdir / "comparison_motivation_shot_summary.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\resizebox{\\linewidth}{!}{%\n")
        fh.write("    \\begin{tabular}{@{}lcccc@{}}\n")
        fh.write("        \\toprule\n")
        fh.write("        Shots & $\\rho_s$ & $r$ & $R^2$ & Mean accuracy \\\\\n")
        fh.write("        \\midrule\n")
        for row in rows:
            fh.write(
                f"        {int(row['shots'])} & "
                f"{_ci_text(float(row['rho_s_mean']), float(row['rho_s_std']), float(row['rho_s_ci_low']), float(row['rho_s_ci_high']))} & "
                f"{_ci_text(float(row['r_mean']), float(row['r_std']), float(row['r_ci_low']), float(row['r_ci_high']))} & "
                f"{_ci_text(float(row['r2_mean']), float(row['r2_std']), float(row['r2_ci_low']), float(row['r2_ci_high']))} & "
                f"{_ci_text(float(row['mean_accuracy_mean']), float(row['mean_accuracy_std']), float(row['mean_accuracy_ci_low']), float(row['mean_accuracy_ci_high']))} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}}\n")
        fh.write(
            "    \\caption{Fixed-gap number-comparison results by shot count. "
            "Entries are mean $\\pm$ standard deviation across exemplar sets, "
            "with bootstrap 95\\% confidence intervals over comparison tasks in brackets. "
            "Correlations are computed between PCA compression factor $\\beta$ and model accuracy.}\n"
        )
        fh.write("    \\label{tab:comparison-motivation-shot-summary}\n")
        fh.write("\\end{table}\n")


def write_spearman_by_set(outdir: Path, rows: list[dict[str, object]]) -> None:
    by_key = {(int(row["shots"]), int(row["exemplar_set_id"])): row for row in rows}
    table_rows: list[dict[str, object]] = []
    for shots in range(5):
        values = [
            float(by_key[(shots, set_id)]["rho_s"])
            for set_id in range(3)
            if (shots, set_id) in by_key
        ]
        row: dict[str, object] = {
            "shots": shots,
            "rho_s_mean": _mean(values),
            "rho_s_std": _std(values),
        }
        for set_id in range(3):
            value = by_key.get((shots, set_id), {}).get("rho_s", "")
            row[f"set{set_id}_rho_s"] = value
        table_rows.append(row)
    write_csv(outdir / "comparison_motivation_spearman_by_set.csv", table_rows)

    with open(outdir / "comparison_motivation_spearman_by_set.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\begin{tabular}{@{}lcccc@{}}\n")
        fh.write("        \\toprule\n")
        fh.write("        Shots & Set 0 & Set 1 & Set 2 & Mean $\\pm$ std \\\\\n")
        fh.write("        \\midrule\n")
        for row in table_rows:
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
            "factor $\\beta$ and fixed-gap comparison accuracy for each exemplar set.}\n"
        )
        fh.write("    \\label{tab:comparison-motivation-spearman-by-set}\n")
        fh.write("\\end{table}\n")


def plot_beta_accuracy(
    outdir: Path,
    model_rows: list[dict[str, object]],
    shot_rows: list[dict[str, object]],
    beta_by_model: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    by_shot = {int(row["shots"]): row for row in shot_rows}
    fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.35), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in model_rows if int(row["shots"]) == shots]
        xs = np.asarray([beta_by_model[str(row["model_label"])] for row in rows], dtype=float)
        ys = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
        std = np.asarray([float(row["accuracy_std"]) for row in rows], dtype=float)
        low = np.asarray([float(row["accuracy_ci_low"]) for row in rows], dtype=float)
        high = np.asarray([float(row["accuracy_ci_high"]) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ci = _plot_yerr(ys[ok], low[ok], high[ok])
        ax.errorbar(xs[ok], ys[ok], yerr=ci, fmt="none", color="#b7b7b7", capsize=2.5, zorder=1)
        ax.errorbar(
            xs[ok],
            ys[ok],
            yerr=std[ok],
            fmt="o",
            color="#246b7f",
            ecolor="#8fb5c1",
            capsize=2.5,
            markersize=4.8,
            zorder=3,
        )
        if int(ok.sum()) >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 200)
            ax.plot(xx, coef[0] * xx + coef[1], color="#b24a3a", linewidth=1.1, zorder=2)
        for row in rows:
            x = beta_by_model[str(row["model_label"])]
            y = float(row["accuracy_mean"])
            if math.isfinite(x) and math.isfinite(y):
                ax.annotate(
                    DISPLAY_LABELS.get(str(row["model_label"]), str(row["model_label"])),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.6,
                )
        row = by_shot[shots]
        ax.set_xscale("log")
        ax.set_title(
            rf"{shots}-shot: $\rho_s={float(row['rho_s_mean']):.2f}$, "
            rf"$R^2={float(row['r2_mean']):.2f}$",
            fontsize=9,
        )
        ax.set_xlabel(r"PCA $\beta$ (log scale)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("comparison accuracy")
    fig.tight_layout()
    fig.savefig(outdir / "comparison_motivation_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "comparison_motivation_beta_accuracy_by_shot.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_beta(
    outdir: Path,
    model_rows: list[dict[str, object]],
    shot_rows: list[dict[str, object]],
    beta_by_model: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    by_shot = {int(row["shots"]): row for row in shot_rows}
    fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.35), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in model_rows if int(row["shots"]) == shots]
        xs = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
        ys = np.asarray([beta_by_model[str(row["model_label"])] for row in rows], dtype=float)
        x_std = np.asarray([float(row["accuracy_std"]) for row in rows], dtype=float)
        x_low = np.asarray([float(row["accuracy_ci_low"]) for row in rows], dtype=float)
        x_high = np.asarray([float(row["accuracy_ci_high"]) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        x_ci = _plot_yerr(xs[ok], x_low[ok], x_high[ok])
        ax.errorbar(xs[ok], ys[ok], xerr=x_ci, fmt="none", color="#b7b7b7", capsize=2.5, zorder=1)
        ax.errorbar(
            xs[ok],
            ys[ok],
            xerr=x_std[ok],
            fmt="o",
            color="#246b7f",
            ecolor="#8fb5c1",
            capsize=2.5,
            markersize=4.8,
            zorder=3,
        )
        if int(ok.sum()) >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 200)
            ax.plot(xx, coef[0] * xx + coef[1], color="#b24a3a", linewidth=1.1, zorder=2)
        for row in rows:
            x = float(row["accuracy_mean"])
            y = beta_by_model[str(row["model_label"])]
            if math.isfinite(x) and math.isfinite(y):
                ax.annotate(
                    DISPLAY_LABELS.get(str(row["model_label"]), str(row["model_label"])),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.6,
                )
        row = by_shot[shots]
        ax.set_title(
            rf"{shots}-shot: $\rho_s={float(row['rho_s_mean']):.2f}$, "
            rf"$R^2={float(row['r2_mean']):.2f}$",
            fontsize=9,
        )
        ax.set_xlabel("comparison accuracy")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\beta$")
    fig.tight_layout()
    fig.savefig(outdir / "comparison_motivation_accuracy_beta_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "comparison_motivation_accuracy_beta_by_shot.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_beta_margin(
    outdir: Path,
    model_rows: list[dict[str, object]],
    beta_by_model: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.35), sharey=True)
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in model_rows if int(row["shots"]) == shots]
        xs = np.asarray([beta_by_model[str(row["model_label"])] for row in rows], dtype=float)
        ys = np.asarray([float(row["margin_mean"]) for row in rows], dtype=float)
        low = np.asarray([float(row["margin_ci_low"]) for row in rows], dtype=float)
        high = np.asarray([float(row["margin_ci_high"]) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ci = _plot_yerr(ys[ok], low[ok], high[ok])
        ax.errorbar(xs[ok], ys[ok], yerr=ci, fmt="o", color="#6b5a92", ecolor="#c4badb", capsize=2.5)
        for row in rows:
            x = beta_by_model[str(row["model_label"])]
            y = float(row["margin_mean"])
            if math.isfinite(x) and math.isfinite(y):
                ax.annotate(
                    DISPLAY_LABELS.get(str(row["model_label"]), str(row["model_label"])),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.6,
                )
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.55)
        ax.set_xscale("log")
        ax.set_title(f"{shots}-shot", fontsize=9)
        ax.set_xlabel(r"PCA $\beta$ (log scale)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("mean margin")
    fig.tight_layout()
    fig.savefig(outdir / "comparison_motivation_beta_margin_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "comparison_motivation_beta_margin_by_shot.png", dpi=260, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.25))
    for ax, (metric, title) in zip(axes, panels):
        means = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        std = np.asarray([float(row[f"{metric}_std"]) for row in rows], dtype=float)
        low = np.asarray([float(row[f"{metric}_ci_low"]) for row in rows], dtype=float)
        high = np.asarray([float(row[f"{metric}_ci_high"]) for row in rows], dtype=float)
        ax.fill_between(shots, low, high, color="#98b5bf", alpha=0.24, linewidth=0)
        ci = _plot_yerr(means, low, high)
        ax.errorbar(shots, means, yerr=ci, fmt="none", ecolor="#b7b7b7", capsize=3, zorder=1)
        ax.errorbar(
            shots,
            means,
            yerr=std,
            marker="o",
            color="#246b7f",
            ecolor="#8fb5c1",
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
        ax.set_title(title)
        ax.set_xlabel("shots")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "comparison_motivation_metrics_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "comparison_motivation_metrics_by_shot.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_spearman_by_set(
    outdir: Path,
    condition_rows: list[dict[str, object]],
    shot_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {0: "#246b7f", 1: "#b24a3a", 2: "#6f8f3a"}
    fig, ax = plt.subplots(figsize=(6.35, 3.65))
    for set_id in range(3):
        rows = sorted(
            [row for row in condition_rows if int(row["exemplar_set_id"]) == set_id],
            key=lambda row: int(row["shots"]),
        )
        if not rows:
            continue
        xs = np.asarray([int(row["shots"]) for row in rows], dtype=float)
        ys = np.asarray([float(row["rho_s"]) for row in rows], dtype=float)
        low = np.asarray([float(row["rho_s_ci_low"]) for row in rows], dtype=float)
        high = np.asarray([float(row["rho_s_ci_high"]) for row in rows], dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=1.25, markersize=4.6, color=colors[set_id], label=f"set {set_id}")
        ax.fill_between(xs, low, high, color=colors[set_id], alpha=0.12, linewidth=0)
    shot_rows = sorted(shot_rows, key=lambda row: int(row["shots"]))
    xs = np.asarray([int(row["shots"]) for row in shot_rows], dtype=float)
    means = np.asarray([float(row["rho_s_mean"]) for row in shot_rows], dtype=float)
    std = np.asarray([float(row["rho_s_std"]) for row in shot_rows], dtype=float)
    low = np.asarray([float(row["rho_s_ci_low"]) for row in shot_rows], dtype=float)
    high = np.asarray([float(row["rho_s_ci_high"]) for row in shot_rows], dtype=float)
    ci = _plot_yerr(means, low, high)
    ax.errorbar(xs, means, yerr=ci, fmt="none", ecolor="#b7b7b7", capsize=4, zorder=3)
    ax.errorbar(xs, means, yerr=std, marker="s", color="#111111", capsize=3, linewidth=1.6, label="mean", zorder=4)
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.55)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("shots")
    ax.set_ylabel(r"Spearman $\rho_s$")
    ax.set_title(r"$\beta$-accuracy rank correlation by exemplar set")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(outdir / "comparison_motivation_spearman_by_set.pdf", bbox_inches="tight")
    fig.savefig(outdir / "comparison_motivation_spearman_by_set.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def log_to_mlflow(outdir: Path, run_name: str) -> str | None:
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] skipped: {exc}")
        return None
    mlflow.set_experiment("numberline_geometry_expv1")
    with mlflow.start_run(run_name=run_name) as run:
        for path in sorted(outdir.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path))
        summary = outdir / "comparison_motivation_shot_summary.csv"
        if summary.exists():
            with open(summary, newline="") as fh:
                for row in csv.DictReader(fh):
                    shots = int(row["shots"])
                    for metric in ("rho_s", "r", "r2", "mean_accuracy", "mean_margin"):
                        value = float(row[f"{metric}_mean"])
                        if math.isfinite(value):
                            mlflow.log_metric(f"{shots}shot_{metric}_mean", value)
        return run.info.run_id


def run_analysis(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_labels, beta_by_model = read_beta_table(args.beta_csv, args.beta_column)
    if args.exclude_models:
        excluded = set(args.exclude_models.split(","))
        model_labels = [label for label in model_labels if label not in excluded]
    beta = np.asarray([beta_by_model[label] for label in model_labels], dtype=float)

    conditions = discover_conditions(args.shots_dir)
    correct_by_key: dict[tuple[int, int], np.ndarray] = {}
    margin_by_key: dict[tuple[int, int], np.ndarray] = {}
    condition_rows: list[dict[str, object]] = []
    model_set_rows: list[dict[str, object]] = []

    for condition in conditions:
        correct, margin, task_ids = load_condition_arrays(condition, model_labels)
        key = (condition.shots, condition.exemplar_set_id)
        correct_by_key[key] = correct
        margin_by_key[key] = margin

        acc_by_model = correct.mean(axis=1)
        margin_by_model = margin.mean(axis=1)
        acc_metrics = correlation_metrics(beta, acc_by_model)
        margin_metrics = correlation_metrics(beta, margin_by_model)
        acc_ci = bootstrap_metric_ci(
            [correct],
            beta,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + condition.shots * 101 + condition.exemplar_set_id,
        )
        margin_ci = bootstrap_metric_ci(
            [margin],
            beta,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 50_000 + condition.shots * 101 + condition.exemplar_set_id,
        )
        row: dict[str, object] = {
            "shots": condition.shots,
            "exemplar_set_id": condition.exemplar_set_id,
            "n_models": len(model_labels),
            "n_tasks": len(task_ids),
            "path": str(condition.path),
            "mean_accuracy": float(np.mean(acc_by_model)),
            "mean_margin": float(np.mean(margin_by_model)),
        }
        for metric in ("rho_s", "r", "r2"):
            row[metric] = acc_metrics[metric]
            row[f"{metric}_ci_low"] = acc_ci[metric][0]
            row[f"{metric}_ci_high"] = acc_ci[metric][1]
            row[f"margin_{metric}"] = margin_metrics[metric]
            row[f"margin_{metric}_ci_low"] = margin_ci[metric][0]
            row[f"margin_{metric}_ci_high"] = margin_ci[metric][1]
        row["mean_accuracy_ci_low"] = acc_ci["mean_value"][0]
        row["mean_accuracy_ci_high"] = acc_ci["mean_value"][1]
        row["mean_margin_ci_low"] = margin_ci["mean_value"][0]
        row["mean_margin_ci_high"] = margin_ci["mean_value"][1]
        condition_rows.append(row)

        for label, acc, avg_margin in zip(model_labels, acc_by_model, margin_by_model):
            model_set_rows.append(
                {
                    "shots": condition.shots,
                    "exemplar_set_id": condition.exemplar_set_id,
                    "model_label": label,
                    "pca_beta": beta_by_model[label],
                    "accuracy": float(acc),
                    "mean_margin": float(avg_margin),
                }
            )

    shot_rows: list[dict[str, object]] = []
    model_shot_rows: list[dict[str, object]] = []
    condition_by_key = {(int(row["shots"]), int(row["exemplar_set_id"])): row for row in condition_rows}
    for shots in range(5):
        keys = sorted(key for key in correct_by_key if key[0] == shots)
        if not keys:
            continue
        correct_matrices = [correct_by_key[key] for key in keys]
        margin_matrices = [margin_by_key[key] for key in keys]
        acc_condition_rows = [condition_by_key[key] for key in keys]
        acc_ci = bootstrap_metric_ci(correct_matrices, beta, n_bootstrap=args.n_bootstrap, seed=args.seed + shots * 997)
        margin_ci = bootstrap_metric_ci(margin_matrices, beta, n_bootstrap=args.n_bootstrap, seed=args.seed + 75_000 + shots * 997)
        row = {"shots": shots, "n_exemplar_sets": len(keys)}
        for metric in ("rho_s", "r", "r2"):
            vals = [float(item[metric]) for item in acc_condition_rows]
            row[f"{metric}_mean"] = _mean(vals)
            row[f"{metric}_std"] = _std(vals)
            row[f"{metric}_ci_low"] = acc_ci[metric][0]
            row[f"{metric}_ci_high"] = acc_ci[metric][1]
            margin_vals = [float(item[f"margin_{metric}"]) for item in acc_condition_rows]
            row[f"margin_{metric}_mean"] = _mean(margin_vals)
            row[f"margin_{metric}_std"] = _std(margin_vals)
            row[f"margin_{metric}_ci_low"] = margin_ci[metric][0]
            row[f"margin_{metric}_ci_high"] = margin_ci[metric][1]
        mean_acc = [float(item["mean_accuracy"]) for item in acc_condition_rows]
        mean_margin = [float(item["mean_margin"]) for item in acc_condition_rows]
        row["mean_accuracy_mean"] = _mean(mean_acc)
        row["mean_accuracy_std"] = _std(mean_acc)
        row["mean_accuracy_ci_low"] = acc_ci["mean_value"][0]
        row["mean_accuracy_ci_high"] = acc_ci["mean_value"][1]
        row["mean_margin_mean"] = _mean(mean_margin)
        row["mean_margin_std"] = _std(mean_margin)
        row["mean_margin_ci_low"] = margin_ci["mean_value"][0]
        row["mean_margin_ci_high"] = margin_ci["mean_value"][1]
        shot_rows.append(row)

        acc_low, acc_high = model_bootstrap_ci(correct_matrices, n_bootstrap=args.n_bootstrap, seed=args.seed + 10_000 + shots)
        margin_low, margin_high = model_bootstrap_ci(margin_matrices, n_bootstrap=args.n_bootstrap, seed=args.seed + 20_000 + shots)
        correct_stack = np.stack(correct_matrices, axis=0)
        margin_stack = np.stack(margin_matrices, axis=0)
        for idx, label in enumerate(model_labels):
            acc_values = correct_stack[:, idx, :].mean(axis=1)
            margin_values = margin_stack[:, idx, :].mean(axis=1)
            model_shot_rows.append(
                {
                    "shots": shots,
                    "model_label": label,
                    "pca_beta": beta_by_model[label],
                    "accuracy_mean": float(np.mean(acc_values)),
                    "accuracy_std": float(np.std(acc_values, ddof=1)) if len(acc_values) > 1 else 0.0,
                    "accuracy_ci_low": float(acc_low[idx]),
                    "accuracy_ci_high": float(acc_high[idx]),
                    "margin_mean": float(np.mean(margin_values)),
                    "margin_std": float(np.std(margin_values, ddof=1)) if len(margin_values) > 1 else 0.0,
                    "margin_ci_low": float(margin_low[idx]),
                    "margin_ci_high": float(margin_high[idx]),
                }
            )

    write_csv(args.outdir / "comparison_motivation_condition_metrics.csv", condition_rows)
    write_csv(args.outdir / "comparison_motivation_model_by_set.csv", model_set_rows)
    write_csv(args.outdir / "comparison_motivation_shot_summary.csv", shot_rows)
    write_csv(args.outdir / "comparison_motivation_model_by_shot.csv", model_shot_rows)
    write_latex_summary(args.outdir, shot_rows)
    write_spearman_by_set(args.outdir, condition_rows)
    plot_beta_accuracy(args.outdir, model_shot_rows, shot_rows, beta_by_model)
    plot_accuracy_beta(args.outdir, model_shot_rows, shot_rows, beta_by_model)
    plot_beta_margin(args.outdir, model_shot_rows, beta_by_model)
    plot_metrics_by_shot(args.outdir, shot_rows)
    plot_spearman_by_set(args.outdir, condition_rows, shot_rows)
    run_id = log_to_mlflow(args.outdir, args.mlflow_run_name) if args.log_mlflow else None
    if run_id:
        print(f"[mlflow] logged analysis run_id={run_id}")
    print(f"[analysis] wrote {args.outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-csv", type=Path, default=Path("results/comparison_motivation/beta/comparison_motivation_beta_summary.csv"))
    parser.add_argument("--shots-dir", type=Path, default=Path("results/comparison_motivation/shots"))
    parser.add_argument("--outdir", type=Path, default=Path("results/comparison_motivation/shots/analysis"))
    parser.add_argument("--beta-column", default="pca_beta")
    parser.add_argument("--exclude-models", default="")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--mlflow-run-name", default="comparison_motivation_analysis")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
