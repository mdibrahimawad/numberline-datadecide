from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.number_analysis import C, _apply_neurips_style, _save


@dataclass(frozen=True)
class ModelArtifact:
    label: str
    path: Path


@dataclass(frozen=True)
class FamilySpec:
    dataset: str
    model_family: str
    artifacts: tuple[ModelArtifact, ...]


FAMILY_SPECS = (
    FamilySpec(
        dataset="Stack v1.2",
        model_family="StarCoderBase",
        artifacts=(
            ModelArtifact(
                "StarCoderBase-1B",
                Path("results/figs/geometry/starcoderbase/starcoderbase_1b/_artifacts/results.json"),
            ),
            ModelArtifact(
                "StarCoderBase-3B",
                Path("results/figs/geometry/starcoderbase/starcoderbase_3b/_artifacts/results.json"),
            ),
            ModelArtifact(
                "StarCoderBase-7B",
                Path("results/figs/geometry/starcoderbase/starcoderbase_7b/_artifacts/results.json"),
            ),
        ),
    ),
    FamilySpec(
        dataset="Dolma v1.5 sample",
        model_family="OLMo",
        artifacts=(
            ModelArtifact(
                "OLMo-7B-2T",
                Path("results/figs/geometry/olmo/olmo_7b_2t/_artifacts/results.json"),
            ),
            ModelArtifact(
                "OLMo-7B-Twin-2T",
                Path("results/figs/geometry/olmo/olmo_7b_twin_2t/_artifacts/results.json"),
            ),
        ),
    ),
    FamilySpec(
        dataset="RedPajama-Data-1T",
        model_family="RedPajama",
        artifacts=(
            ModelArtifact(
                "RedPajama-3B",
                Path("results/figs/geometry/redpajama/RedPajama-INCITE-Base-3B-v1/_artifacts/results.json"),
            ),
            ModelArtifact(
                "RedPajama-7B",
                Path("results/figs/geometry/redpajama/RedPajama-INCITE-7B-Base/_artifacts/results.json"),
            ),
        ),
    ),
    FamilySpec(
        dataset="Falcon RefinedWeb",
        model_family="Falcon-RW",
        artifacts=(
            ModelArtifact(
                "Falcon-RW-1B",
                Path("results/figs/geometry/_artifacts_falcon_scatter/tiiuae_falcon-rw-1b/results.json"),
            ),
            ModelArtifact(
                "Falcon-RW-7B",
                Path("results/figs/geometry/_artifacts_falcon_scatter/tiiuae_falcon-rw-7b/results.json"),
            ),
        ),
    ),
)


def _read_alpha(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[row["dataset"]] = {
                "alpha": float(row["value_power_alpha"]),
                "alpha_r2": float(row["value_power_r2"]),
                "rank_s": float(row["zipf_rank_s"]),
                "rank_r2": float(row["zipf_rank_r2"]),
            }
    return out


def _best_metric(payload: dict, method: str, metric: str) -> tuple[int, float, float]:
    layer = int(payload[f"best_layer_{method}"])
    item = payload[method][str(layer)]
    return layer, float(item[f"{metric}_mean"]), float(item[f"{metric}_std"])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(x, dtype=float)), _rankdata(np.asarray(y, dtype=float)))


def collect_rows(alpha_path: Path) -> tuple[list[dict], list[dict]]:
    alpha = _read_alpha(alpha_path)
    family_rows: list[dict] = []
    model_rows: list[dict] = []
    for spec in FAMILY_SPECS:
        if spec.dataset not in alpha:
            raise SystemExit(f"missing dataset alpha row for {spec.dataset} in {alpha_path}")
        pca_betas: list[float] = []
        pls_betas: list[float] = []
        for artifact in spec.artifacts:
            if not artifact.path.exists():
                raise SystemExit(f"missing model artifact: {artifact.path}")
            with open(artifact.path) as fh:
                payload = json.load(fh)
            pca_layer, pca_beta, pca_beta_std = _best_metric(payload, "pca", "beta")
            _, pca_rho, pca_rho_std = _best_metric(payload, "pca", "rho")
            pls_layer, pls_beta, pls_beta_std = _best_metric(payload, "pls", "beta")
            _, pls_rho, pls_rho_std = _best_metric(payload, "pls", "rho")
            pca_betas.append(pca_beta)
            pls_betas.append(pls_beta)
            model_rows.append(
                {
                    "dataset": spec.dataset,
                    "model_family": spec.model_family,
                    "model": artifact.label,
                    "alpha_N": alpha[spec.dataset]["alpha"],
                    "alpha_R2": alpha[spec.dataset]["alpha_r2"],
                    "pca_layer": pca_layer,
                    "pca_beta": pca_beta,
                    "pca_beta_std": pca_beta_std,
                    "pca_rho": pca_rho,
                    "pca_rho_std": pca_rho_std,
                    "pls_layer": pls_layer,
                    "pls_beta": pls_beta,
                    "pls_beta_std": pls_beta_std,
                    "pls_rho": pls_rho,
                    "pls_rho_std": pls_rho_std,
                }
            )
        family_rows.append(
            {
                "dataset": spec.dataset,
                "model_family": spec.model_family,
                "alpha_N": alpha[spec.dataset]["alpha"],
                "alpha_R2": alpha[spec.dataset]["alpha_r2"],
                "zipf_rank_s": alpha[spec.dataset]["rank_s"],
                "zipf_rank_R2": alpha[spec.dataset]["rank_r2"],
                "n_models": len(spec.artifacts),
                "pca_beta_mean": float(np.mean(pca_betas)),
                "pca_beta_sd_across_models": float(np.std(pca_betas, ddof=1)) if len(pca_betas) > 1 else 0.0,
                "pls_beta_mean": float(np.mean(pls_betas)),
                "pls_beta_sd_across_models": float(np.std(pls_betas, ddof=1)) if len(pls_betas) > 1 else 0.0,
                "models": ", ".join(a.label for a in spec.artifacts),
            }
        )
    family_rows.sort(key=lambda row: row["alpha_N"])
    model_rows.sort(key=lambda row: (row["alpha_N"], row["model"]))
    return family_rows, model_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def write_tex(path: Path, family_rows: list[dict]) -> None:
    with open(path, "w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n")
        fh.write("\\toprule\n")
        fh.write("Dataset / family & $\\alpha_N$ & $R^2_N$ & PCA $\\beta$ & PLS $\\beta$ \\\\\n")
        fh.write("\\midrule\n")
        for row in family_rows:
            fh.write(
                f"{row['dataset']} / {row['model_family']} & "
                f"{row['alpha_N']:.2f} & "
                f"{row['alpha_R2']:.2f} & "
                f"{row['pca_beta_mean']:.2f} $\\pm$ {row['pca_beta_sd_across_models']:.2f} & "
                f"{row['pls_beta_mean']:.2f} $\\pm$ {row['pls_beta_sd_across_models']:.2f} \\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
    print(f"  wrote {path}")


def write_summary(path: Path, summary: dict) -> None:
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {path}")


def plot_alpha_beta(outdir: Path, family_rows: list[dict], model_rows: list[dict]) -> None:
    _apply_neurips_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.35))

    colors = {
        "StarCoderBase": C["accent2"],
        "OLMo": C["empirical"],
        "RedPajama": C["accent1"],
        "Falcon-RW": C["zipf"],
    }

    for idx, row in enumerate(family_rows):
        alpha = row["alpha_N"]
        fam = row["model_family"]
        color = colors.get(fam, C["muted"])
        sub = [item for item in model_rows if item["model_family"] == fam]
        offsets = np.linspace(-0.006, 0.006, len(sub)) if len(sub) > 1 else [0.0]
        for offset, item in zip(offsets, sub):
            ax.scatter(
                alpha + float(offset),
                item["pca_beta"],
                s=26,
                color=color,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        ax.errorbar(
            alpha,
            row["pca_beta_mean"],
            yerr=row["pca_beta_sd_across_models"],
            fmt="o",
            ms=5.5,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=3,
            label=fam,
            zorder=4,
        )
        ax.annotate(
            fam,
            (alpha, row["pca_beta_mean"]),
            xytext=(4, 4 if idx % 2 == 0 else -10),
            textcoords="offset points",
            fontsize=8,
        )

    xs = np.array([row["alpha_N"] for row in family_rows], dtype=float)
    ys = np.array([row["pca_beta_mean"] for row in family_rows], dtype=float)
    slope, intercept = np.polyfit(xs, np.log10(ys), 1)
    xline = np.linspace(xs.min() - 0.015, xs.max() + 0.015, 100)
    ax.plot(xline, 10 ** (intercept + slope * xline), color="black", lw=0.8, ls="--")

    ax.set_yscale("log")
    ax.set_xlabel(r"dataset value exponent $\alpha_N$")
    ax.set_ylabel(r"PCA compression factor $\beta$")
    ax.set_title("Dataset alpha vs model beta (Pile/Pythia excluded)")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, outdir, "alpha_beta_without_pile")


def maybe_log_mlflow(outdir: Path, summary: dict, *, experiment: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow.db')}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="alpha_beta_without_pile_order") as run:
        mlflow.set_tags(
            {
                "stage": "alpha_beta_comparison",
                "excluded_dataset": "Pile Uncopyrighted",
                "excluded_model_family": "Pythia",
            }
        )
        for key, value in summary.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                mlflow.log_metric(key, float(value))
        for artifact in [
            "alpha_beta_family_without_pile.csv",
            "alpha_beta_models_without_pile.csv",
            "alpha_beta_without_pile_summary.json",
            "alpha_beta_without_pile.tex",
            "alpha_beta_without_pile.pdf",
            "alpha_beta_without_pile.png",
        ]:
            path = outdir / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
        print(f"  MLflow run_id={run.info.run_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha-summary",
        default="results/dataset_frequency_0_to_10000/all_with_redpajama/frequency_summary_0_to_10000.csv",
    )
    parser.add_argument("--outdir", default="results/dataset_frequency_0_to_10000/alpha_beta_without_pile")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="dataset_frequency_0_to_10000")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    family_rows, model_rows = collect_rows(Path(args.alpha_summary))
    family_alpha = np.array([row["alpha_N"] for row in family_rows], dtype=float)
    family_pca_beta = np.array([row["pca_beta_mean"] for row in family_rows], dtype=float)
    family_pls_beta = np.array([row["pls_beta_mean"] for row in family_rows], dtype=float)
    model_alpha = np.array([row["alpha_N"] for row in model_rows], dtype=float)
    model_pca_beta = np.array([row["pca_beta"] for row in model_rows], dtype=float)
    model_pls_beta = np.array([row["pls_beta"] for row in model_rows], dtype=float)
    summary = {
        "n_families": len(family_rows),
        "n_models": len(model_rows),
        "family_spearman_alpha_pca_beta": _spearman(family_alpha, family_pca_beta),
        "family_spearman_alpha_pls_beta": _spearman(family_alpha, family_pls_beta),
        "family_pearson_alpha_log10_pca_beta": _pearson(family_alpha, np.log10(family_pca_beta)),
        "family_pearson_alpha_log10_pls_beta": _pearson(family_alpha, np.log10(family_pls_beta)),
        "model_spearman_alpha_pca_beta": _spearman(model_alpha, model_pca_beta),
        "model_spearman_alpha_pls_beta": _spearman(model_alpha, model_pls_beta),
        "model_pearson_alpha_log10_pca_beta": _pearson(model_alpha, np.log10(model_pca_beta)),
        "model_pearson_alpha_log10_pls_beta": _pearson(model_alpha, np.log10(model_pls_beta)),
    }

    write_csv(outdir / "alpha_beta_family_without_pile.csv", family_rows)
    write_csv(outdir / "alpha_beta_models_without_pile.csv", model_rows)
    write_tex(outdir / "alpha_beta_without_pile.tex", family_rows)
    write_summary(outdir / "alpha_beta_without_pile_summary.json", summary)
    plot_alpha_beta(outdir, family_rows, model_rows)
    if args.log_mlflow:
        maybe_log_mlflow(outdir, summary, experiment=args.mlflow_experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
