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


FAMILIES = {
    "Falcon": ["Falcon-RW-1B", "Falcon-RW-7B"],
    "RedPajama": ["RedPajama-3B", "RedPajama-7B"],
    "StarCoder": ["StarCoderBase-1B", "StarCoderBase-3B", "StarCoderBase-7B"],
    "OLMo": ["OLMo-7B-2T", "OLMo-7B-Twin-2T"],
}

DISPLAY_LABELS = {
    "Falcon-RW-1B": "1B",
    "Falcon-RW-7B": "7B",
    "RedPajama-3B": "3B",
    "RedPajama-7B": "7B",
    "StarCoderBase-1B": "1B",
    "StarCoderBase-3B": "3B",
    "StarCoderBase-7B": "7B",
    "OLMo-7B-2T": "2T",
    "OLMo-7B-Twin-2T": "Twin",
}


@dataclass(frozen=True)
class Condition:
    shots: int
    exemplar_set_id: int
    path: Path


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def read_beta_table(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            rows[model] = {
                "pca_beta": float(row["pca_beta"]),
                "pca_beta_std": float(row.get("pca_beta_std") or 0.0),
                "pls_beta": float(row["pls_beta"]),
                "pls_beta_std": float(row.get("pls_beta_std") or 0.0),
            }
    return rows


def condition_paths(root: Path) -> list[Condition]:
    sweep = root / "fixed_gap_dense_fewshot_exemplar_sweep"
    paths = [
        Condition(0, 0, root / "fixed_gap_dense_number_answer"),
        Condition(0, 1, sweep / "0shot_set1"),
        Condition(0, 2, sweep / "0shot_set2"),
        Condition(1, 0, sweep / "1shot_set0"),
        Condition(1, 1, sweep / "1shot_set1"),
        Condition(1, 2, sweep / "1shot_set2"),
        Condition(2, 0, root / "fixed_gap_dense_number_answer_2shot"),
        Condition(2, 1, sweep / "2shot_set1"),
        Condition(2, 2, sweep / "2shot_set2"),
        Condition(3, 0, sweep / "3shot_set0"),
        Condition(3, 1, sweep / "3shot_set1"),
        Condition(3, 2, sweep / "3shot_set2"),
        Condition(4, 0, root / "fixed_gap_dense_number_answer_4shot"),
        Condition(4, 1, sweep / "4shot_set1"),
        Condition(4, 2, sweep / "4shot_set2"),
    ]
    missing = [item.path for item in paths if not (item.path / "number_comparison_model_summary.csv").exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))
    return paths


def read_condition(condition: Condition) -> list[dict[str, object]]:
    rows = []
    with open(condition.path / "number_comparison_model_summary.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "shots": condition.shots,
                    "exemplar_set_id": condition.exemplar_set_id,
                    "model": row["model_label"],
                    "accuracy": float(row.get("fixed_gap_accuracy") or row["accuracy"]),
                    "mean_margin": float(row.get("fixed_gap_mean_margin") or row.get("mean_margin") or "nan"),
                }
            )
    return rows


def corr_metrics(beta: np.ndarray, values: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(beta) & np.isfinite(values)
    if int(ok.sum()) < 2:
        return {"rho_s": float("nan"), "r": float("nan"), "r2": float("nan")}
    rho = float(spearmanr(beta[ok], values[ok]).statistic)
    if int(ok.sum()) < 3:
        r = 1.0 if (beta[ok][1] - beta[ok][0]) * (values[ok][1] - values[ok][0]) > 0 else -1.0
    else:
        r = float(pearsonr(beta[ok], values[ok]).statistic)
    return {"rho_s": rho, "r": r, "r2": r * r if math.isfinite(r) else float("nan")}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    condition_rows: list[dict[str, object]],
    beta_rows: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_model_shot_set: dict[tuple[str, int, int], dict[str, object]] = {}
    for row in condition_rows:
        by_model_shot_set[(str(row["model"]), int(row["shots"]), int(row["exemplar_set_id"]))] = row

    model_rows: list[dict[str, object]] = []
    for family, models in FAMILIES.items():
        for shots in range(5):
            for model in models:
                set_rows = [
                    by_model_shot_set[(model, shots, set_id)]
                    for set_id in range(3)
                    if (model, shots, set_id) in by_model_shot_set
                ]
                if not set_rows:
                    continue
                beta = beta_rows[model]
                acc_values = [float(row["accuracy"]) for row in set_rows]
                margin_values = [float(row["mean_margin"]) for row in set_rows]
                model_rows.append(
                    {
                        "family": family,
                        "shots": shots,
                        "model": model,
                        "pca_beta": beta["pca_beta"],
                        "pca_beta_std": beta["pca_beta_std"],
                        "pls_beta": beta["pls_beta"],
                        "pls_beta_std": beta["pls_beta_std"],
                        "accuracy_mean": _mean(acc_values),
                        "accuracy_std": _std(acc_values),
                        "mean_margin": _mean(margin_values),
                        "margin_std": _std(margin_values),
                    }
                )

    direction_rows: list[dict[str, object]] = []
    for family, models in FAMILIES.items():
        for shots in range(5):
            rows = [row for row in model_rows if row["family"] == family and int(row["shots"]) == shots]
            if len(rows) < 2:
                continue
            accuracy = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
            for beta_name in ("pca_beta", "pls_beta"):
                beta = np.asarray([float(row[beta_name]) for row in rows], dtype=float)
                metrics = corr_metrics(beta, accuracy)
                best_acc_idx = int(np.nanargmax(accuracy))
                best_beta_idx = int(np.nanargmax(beta))
                direction_rows.append(
                    {
                        "family": family,
                        "shots": shots,
                        "beta_type": beta_name.replace("_beta", ""),
                        "n_models": len(rows),
                        "rho_s": metrics["rho_s"],
                        "r": metrics["r"],
                        "r2": metrics["r2"],
                        "best_accuracy_model": rows[best_acc_idx]["model"],
                        "best_beta_model": rows[best_beta_idx]["model"],
                        "best_model_matches": rows[best_acc_idx]["model"] == rows[best_beta_idx]["model"],
                    }
                )
    return model_rows, direction_rows


def plot_family_grid(
    outdir: Path,
    model_rows: list[dict[str, object]],
    direction_rows: list[dict[str, object]],
    *,
    beta_type: str,
) -> None:
    import matplotlib.pyplot as plt

    beta_col = f"{beta_type}_beta"
    beta_std_col = f"{beta_type}_beta_std"
    direction = {
        (str(row["family"]), int(row["shots"]), str(row["beta_type"])): row
        for row in direction_rows
    }
    fig, axes = plt.subplots(len(FAMILIES), 5, figsize=(15.4, 9.0), squeeze=False)
    for row_idx, (family, models) in enumerate(FAMILIES.items()):
        family_rows = [row for row in model_rows if row["family"] == family]
        y_values = [float(row[beta_col]) for row in family_rows]
        y_margin = max((max(y_values) - min(y_values)) * 0.22, 0.08) if y_values else 0.1
        for shots in range(5):
            ax = axes[row_idx][shots]
            rows = [
                row
                for row in family_rows
                if int(row["shots"]) == shots and row["model"] in models
            ]
            rows = sorted(rows, key=lambda row: models.index(str(row["model"])))
            xs = np.asarray([float(row["accuracy_mean"]) for row in rows], dtype=float)
            ys = np.asarray([float(row[beta_col]) for row in rows], dtype=float)
            xerr = np.asarray([float(row["accuracy_std"]) for row in rows], dtype=float)
            yerr = np.asarray([float(row[beta_std_col]) for row in rows], dtype=float)
            ax.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color="#246b7f",
                ecolor="#9fb6bd",
                capsize=2.5,
                markersize=5.2,
                zorder=3,
            )
            if len(rows) >= 2:
                order = np.argsort(xs)
                ax.plot(xs[order], ys[order], color="#b24a3a", linewidth=1.0, alpha=0.8, zorder=2)
            for item, x, y in zip(rows, xs, ys):
                ax.annotate(
                    DISPLAY_LABELS.get(str(item["model"]), str(item["model"])),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7.0,
                )
            stats = direction.get((family, shots, beta_type))
            if stats:
                ax.set_title(rf"{shots}-shot, $\rho_s={float(stats['rho_s']):.2f}$", fontsize=8.5)
            else:
                ax.set_title(f"{shots}-shot", fontsize=8.5)
            if shots == 0:
                ax.set_ylabel(f"{family}\n{beta_type.upper()} " + r"$\beta$")
            if row_idx == len(FAMILIES) - 1:
                ax.set_xlabel("accuracy")
            ax.grid(True, alpha=0.25)
            if y_values:
                ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
            ax.set_xlim(0.48, 1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"family_accuracy_{beta_type}_beta_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"family_accuracy_{beta_type}_beta_by_shot.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_family_summary(
    outdir: Path,
    direction_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    colors = {"Falcon": "#246b7f", "RedPajama": "#b24a3a", "StarCoder": "#6f8f3a", "OLMo": "#6b5a92"}
    for ax, beta_type in zip(axes, ("pca", "pls")):
        for family in FAMILIES:
            rows = sorted(
                [
                    row
                    for row in direction_rows
                    if row["family"] == family and row["beta_type"] == beta_type
                ],
                key=lambda row: int(row["shots"]),
            )
            xs = [int(row["shots"]) for row in rows]
            ys = [float(row["rho_s"]) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=4.8, color=colors[family], label=family)
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.55)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel("shots")
        ax.set_title(beta_type.upper() + r" $\beta$ vs accuracy")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"within-family Spearman $\rho_s$")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "family_beta_accuracy_spearman_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "family_beta_accuracy_spearman_by_shot.png", dpi=260, bbox_inches="tight")
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
        return run.info.run_id


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    beta_rows = read_beta_table(args.beta_csv)
    conditions = condition_paths(args.comparison_root)
    condition_rows: list[dict[str, object]] = []
    for condition in conditions:
        condition_rows.extend(read_condition(condition))
    model_rows, direction_rows = summarize(condition_rows, beta_rows)
    write_csv(args.outdir / "family_model_by_shot.csv", model_rows)
    write_csv(args.outdir / "family_direction_by_shot.csv", direction_rows)
    plot_family_grid(args.outdir, model_rows, direction_rows, beta_type="pca")
    plot_family_grid(args.outdir, model_rows, direction_rows, beta_type="pls")
    plot_family_summary(args.outdir, direction_rows)
    if args.log_mlflow:
        run_id = log_to_mlflow(args.outdir, args.mlflow_run_name)
        if run_id:
            print(f"[mlflow] logged family analysis run_id={run_id}")
    print(f"[family] wrote {args.outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path, default=Path("results/number_comparison"))
    parser.add_argument(
        "--beta-csv",
        type=Path,
        default=Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/comparison_motivation/family_controls"))
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--mlflow-run-name", default="comparison_motivation_family_controls")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
