from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DOMAIN_BY_MODEL = {
    "Falcon-RW-1B": "Web/general base",
    "Falcon-RW-7B": "Web/general base",
    "RedPajama-3B": "Web/general base",
    "RedPajama-7B": "Web/general base",
    "Pythia-2.8B": "Web/general base",
    "StarCoderBase-1B": "Code",
    "StarCoderBase-3B": "Code",
    "StarCoderBase-7B": "Code",
    "OLMo-7B-2T": "Dolma/OLMo",
    "OLMo-7B-Twin-2T": "Dolma/OLMo",
    "Llama-2-7B": "Modern general base",
    "Mistral-7B": "Modern general base",
    "DeepSeek-Base-7B": "Modern general base",
    "Qwen1.5-7B": "Modern general base",
}

LABEL_BY_MODEL = {
    "Falcon-RW-1B": "Falcon-1B",
    "Falcon-RW-7B": "Falcon-7B",
    "RedPajama-3B": "RPJ-3B",
    "RedPajama-7B": "RPJ-7B",
    "Pythia-2.8B": "Pythia",
    "StarCoderBase-1B": "SCB-1B",
    "StarCoderBase-3B": "SCB-3B",
    "StarCoderBase-7B": "SCB-7B",
    "OLMo-7B-2T": "OLMo-2T",
    "OLMo-7B-Twin-2T": "OLMo-Twin",
    "Llama-2-7B": "Llama-2",
    "Mistral-7B": "Mistral",
    "DeepSeek-Base-7B": "DeepSeek",
    "Qwen1.5-7B": "Qwen",
}

DOMAIN_STYLE = {
    "Web/general base": {"color": "#2f6f8f", "marker": "o"},
    "Code": {"color": "#6f8f3a", "marker": "s"},
    "Dolma/OLMo": {"color": "#6b5a92", "marker": "D"},
    "Modern general base": {"color": "#c7503d", "marker": "^"},
}


def read_family_beta(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            out[model] = {
                "pca_beta": float(row["pca_beta"]),
                "pls_beta": float(row["pls_beta"]),
            }
    return out


def read_new_beta(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            out[row["model_label"]] = {
                "pca_beta": float(row["pca_beta"]),
                "pls_beta": float(row["pls_beta"]),
            }
    return out


def read_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    old_beta = read_family_beta(args.old_beta_csv)
    new_beta = read_new_beta(args.new_beta_csv)
    rows: list[dict[str, object]] = []

    with open(args.old_summary_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            model = row["model_label"]
            shot = int(row["n_shots"])
            if shot not in {0, 2, 4} or model not in DOMAIN_BY_MODEL:
                continue
            rows.append(
                {
                    "model": model,
                    "label": LABEL_BY_MODEL[model],
                    "domain": DOMAIN_BY_MODEL[model],
                    "shots": shot,
                    "pca_beta": float(row["pca_beta"]),
                    "pls_beta": old_beta[model]["pls_beta"],
                    "accuracy": float(row["fixed_gap_accuracy"]),
                    "source": "original",
                }
            )

    old_models = {str(row["model"]) for row in rows}
    with open(args.new_summary_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            model = row["model_label"]
            shot = int(row["shots"])
            if shot not in {0, 2, 4} or model in old_models or model not in DOMAIN_BY_MODEL:
                continue
            if model not in new_beta:
                continue
            rows.append(
                {
                    "model": model,
                    "label": LABEL_BY_MODEL[model],
                    "domain": DOMAIN_BY_MODEL[model],
                    "shots": shot,
                    "pca_beta": new_beta[model]["pca_beta"],
                    "pls_beta": new_beta[model]["pls_beta"],
                    "accuracy": float(row["accuracy_mean"]),
                    "source": "comparison_motivation",
                }
            )
    return rows


def corr_metrics(points: list[dict[str, object]], beta_key: str) -> dict[str, float]:
    xs = np.asarray([float(row[beta_key]) for row in points], dtype=float)
    ys = np.asarray([float(row["accuracy"]) for row in points], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if int(ok.sum()) < 2:
        return {"rho_s": float("nan"), "r": float("nan"), "r2": float("nan")}
    rho = float(spearmanr(xs[ok], ys[ok]).statistic)
    r = float(pearsonr(xs[ok], ys[ok]).statistic) if int(ok.sum()) > 2 else (
        1.0 if (xs[ok][1] - xs[ok][0]) * (ys[ok][1] - ys[ok][0]) > 0 else -1.0
    )
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


def write_correlation_summary(outdir: Path, rows: list[dict[str, object]]) -> None:
    summary: list[dict[str, object]] = []
    domains = ["All"] + list(DOMAIN_STYLE)
    for beta_type in ("pca", "pls"):
        beta_key = f"{beta_type}_beta"
        for shots in (0, 2, 4):
            shot_rows = [row for row in rows if int(row["shots"]) == shots]
            for domain in domains:
                points = shot_rows if domain == "All" else [row for row in shot_rows if row["domain"] == domain]
                if len(points) < 2:
                    continue
                metrics = corr_metrics(points, beta_key)
                summary.append(
                    {
                        "beta_type": beta_type,
                        "shots": shots,
                        "domain": domain,
                        "n_models": len({row["model"] for row in points}),
                        "spearman": metrics["rho_s"],
                        "pearson": metrics["r"],
                        "r2": metrics["r2"],
                    }
                )
    write_csv(outdir / "domain_group_beta_accuracy_correlations.csv", summary)


def add_fit_line(ax, points: list[dict[str, object]], beta_key: str) -> None:
    xs = np.asarray([float(row[beta_key]) for row in points], dtype=float)
    ys = np.asarray([float(row["accuracy"]) for row in points], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if int(ok.sum()) < 2:
        return
    coef = np.polyfit(xs[ok], ys[ok], 1)
    xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 160)
    ax.plot(xx, coef[0] * xx + coef[1], color="#333333", linewidth=1.1, alpha=0.45, zorder=1)


def plot_beta_x(outdir: Path, rows: list[dict[str, object]], *, beta_type: str) -> None:
    import matplotlib.pyplot as plt

    beta_key = f"{beta_type}_beta"
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharey=True)
    for ax, shots in zip(axes, (0, 2, 4)):
        panel = [row for row in rows if int(row["shots"]) == shots]
        add_fit_line(ax, panel, beta_key)
        for domain, style in DOMAIN_STYLE.items():
            points = [row for row in panel if row["domain"] == domain]
            if not points:
                continue
            ax.scatter(
                [float(row[beta_key]) for row in points],
                [float(row["accuracy"]) for row in points],
                s=58,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.55,
                label=domain,
                zorder=3,
            )
        metrics = corr_metrics(panel, beta_key)
        ax.set_title(rf"{shots}-shot: $\rho_s={metrics['rho_s']:.2f}$, $R^2={metrics['r2']:.2f}$", fontsize=9)
        ax.set_xlabel(rf"{beta_type.upper()} compression factor $\beta$")
        ax.grid(True, alpha=0.25)
        for row in panel:
            x = float(row[beta_key])
            y = float(row["accuracy"])
            dx, dy = (4, 4)
            if row["model"] in {"StarCoderBase-3B", "Qwen1.5-7B"}:
                dx = -44
            if row["model"] == "Pythia-2.8B":
                dy = -12
            ax.annotate(str(row["label"]), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.0)
    axes[0].set_ylabel("fixed-gap comparison accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(outdir / f"domain_group_{beta_type}_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"domain_group_{beta_type}_beta_accuracy_by_shot.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_x(outdir: Path, rows: list[dict[str, object]], *, beta_type: str) -> None:
    import matplotlib.pyplot as plt

    beta_key = f"{beta_type}_beta"
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharey=True)
    for ax, shots in zip(axes, (0, 2, 4)):
        panel = [row for row in rows if int(row["shots"]) == shots]
        for domain, style in DOMAIN_STYLE.items():
            points = [row for row in panel if row["domain"] == domain]
            if not points:
                continue
            ax.scatter(
                [float(row["accuracy"]) for row in points],
                [float(row[beta_key]) for row in points],
                s=58,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.55,
                label=domain,
                zorder=3,
            )
        metrics = corr_metrics(panel, beta_key)
        ax.set_title(rf"{shots}-shot: $\rho_s={metrics['rho_s']:.2f}$, $R^2={metrics['r2']:.2f}$", fontsize=9)
        ax.set_xlabel("fixed-gap comparison accuracy")
        ax.grid(True, alpha=0.25)
        for row in panel:
            x = float(row["accuracy"])
            y = float(row[beta_key])
            dx, dy = (4, 4)
            if row["model"] in {"StarCoderBase-3B", "Qwen1.5-7B"}:
                dx = -44
            if row["model"] == "Pythia-2.8B":
                dy = -12
            ax.annotate(str(row["label"]), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.0)
    axes[0].set_ylabel(rf"{beta_type.upper()} compression factor $\beta$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(outdir / f"domain_group_accuracy_{beta_type}_beta_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"domain_group_accuracy_{beta_type}_beta_by_shot.png", dpi=260, bbox_inches="tight")
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
    rows = read_rows(args)
    rows = sorted(rows, key=lambda row: (int(row["shots"]), str(row["domain"]), float(row["pca_beta"])))
    write_csv(args.outdir / "domain_group_model_by_shot.csv", rows)
    write_correlation_summary(args.outdir, rows)
    for beta_type in ("pca", "pls"):
        plot_beta_x(args.outdir, rows, beta_type=beta_type)
        plot_accuracy_x(args.outdir, rows, beta_type=beta_type)
    if args.log_mlflow:
        run_id = log_to_mlflow(args.outdir, args.mlflow_run_name)
        if run_id:
            print(f"[mlflow] logged domain group run_id={run_id}")
    print(f"[domain] wrote {args.outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-summary-csv", type=Path, default=Path("results/number_comparison/fixed_gap_shot_analysis/number_comparison_shot_model_summary.csv"))
    parser.add_argument("--old-beta-csv", type=Path, default=Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"))
    parser.add_argument("--new-summary-csv", type=Path, default=Path("results/comparison_motivation/shots/analysis_no_gpt/comparison_motivation_model_by_shot.csv"))
    parser.add_argument("--new-beta-csv", type=Path, default=Path("results/comparison_motivation/beta/comparison_motivation_beta_summary.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/comparison_motivation/domain_groups"))
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--mlflow-run-name", default="comparison_motivation_domain_groups")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
