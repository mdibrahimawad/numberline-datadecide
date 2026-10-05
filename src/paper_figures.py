from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.number_analysis import C, _apply_neurips_style, _kl, _save


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path


MODEL_SPECS = [
    ModelSpec("Pythia-2.8B", Path("results/figs/geometry/_artifacts/results.json")),
    ModelSpec(
        "Falcon-RW-1B",
        Path("results/figs/geometry/_artifacts_falcon_scatter/tiiuae_falcon-rw-1b/results.json"),
    ),
    ModelSpec(
        "Falcon-RW-7B",
        Path("results/figs/geometry/_artifacts_falcon_scatter/tiiuae_falcon-rw-7b/results.json"),
    ),
    ModelSpec(
        "RedPajama-3B",
        Path("results/figs/geometry/redpajama/RedPajama-INCITE-Base-3B-v1/_artifacts/results.json"),
    ),
    ModelSpec(
        "RedPajama-7B",
        Path("results/figs/geometry/redpajama/RedPajama-INCITE-7B-Base/_artifacts/results.json"),
    ),
    ModelSpec("OLMo-7B-2T", Path("results/figs/geometry/olmo/olmo_7b_2t/_artifacts/results.json")),
    ModelSpec(
        "OLMo-7B-Twin-2T",
        Path("results/figs/geometry/olmo/olmo_7b_twin_2t/_artifacts/results.json"),
    ),
    ModelSpec(
        "StarCoderBase-1B",
        Path("results/figs/geometry/starcoderbase/starcoderbase_1b/_artifacts/results.json"),
    ),
    ModelSpec(
        "StarCoderBase-3B",
        Path("results/figs/geometry/starcoderbase/starcoderbase_3b/_artifacts/results.json"),
    ),
    ModelSpec(
        "StarCoderBase-7B",
        Path("results/figs/geometry/starcoderbase/starcoderbase_7b/_artifacts/results.json"),
    ),
]


DATASET_SPECS = [
    DatasetSpec("Stack v1.2", Path("results/stack_v1_2/0_to_10000/counts_0_to_10000.csv")),
    DatasetSpec("Dolma v1.5 sample", Path("results/dolma_v1_5_sample/0_to_10000/counts_0_to_10000.csv")),
    DatasetSpec("Pile Uncopyrighted", Path("results/pile_uncopyrighted/0_to_10000/counts_0_to_10000.csv")),
    DatasetSpec("Falcon RefinedWeb", Path("results/refinedweb_full_modal/0_to_10000/counts_0_to_10000.csv")),
    DatasetSpec("RedPajama-1T", Path("results/redpajama_full_modal/0_to_10000/counts_0_to_10000.csv")),
]


def _metric(layer_metrics: dict, layer: int, key: str) -> float:
    return float(layer_metrics[str(layer)][key])


def _projection_xy(result: dict, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = result["projections_pca"][str(layer)]
    xs: list[float] = []
    ys: list[float] = []
    groups: list[int] = []
    for group_s in sorted(payload["answers"], key=lambda g: int(g)):
        group = int(group_s)
        answers = payload["answers"][group_s]
        projections = payload["projections"][group_s]
        for answer, projection in zip(answers, projections):
            xs.append(math.log10(max(float(answer), 1.0)))
            ys.append(float(projection[0] if isinstance(projection, list) else projection))
            groups.append(group)
    return np.array(xs), np.array(ys), np.array(groups)


def _grid(n: int, ncols: int = 5) -> tuple[int, int]:
    return math.ceil(n / ncols), ncols


def _load_models() -> list[tuple[ModelSpec, dict]]:
    out = []
    for spec in MODEL_SPECS:
        if spec.path.exists():
            with open(spec.path) as fh:
                out.append((spec, json.load(fh)))
        else:
            print(f"[paper_figures] missing model artifact: {spec.path}")
    if not out:
        raise SystemExit("no model artifacts found")
    return out


def _model_title(label: str, result: dict, method: str = "pca") -> str:
    best_key = f"best_layer_{method}"
    metrics_key = method
    layer = int(result[best_key])
    metrics = result[metrics_key]
    rho = _metric(metrics, layer, "rho_mean")
    beta = _metric(metrics, layer, "beta_mean")
    return rf"{label}" + "\n" + rf"$L={layer},\ \rho={rho:.2f},\ \beta={beta:.2f}$"


def plot_model_pca_scatter(models: list[tuple[ModelSpec, dict]], outdir: Path) -> None:
    nrows, ncols = _grid(len(models))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 5.8), squeeze=False)
    cmap = mpl.colors.ListedColormap(mpl.colormaps["viridis"](np.linspace(0, 1, 4)))

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (spec, result) in zip(axes.flat, models):
        ax.set_visible(True)
        layer = int(result["best_layer_pca"])
        x, y, group = _projection_xy(result, layer)
        ax.scatter(
            x,
            y,
            c=group,
            cmap=cmap,
            vmin=1,
            vmax=4,
            s=16,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(_model_title(spec.label, result, "pca"), fontsize=10, pad=3)
        ax.set_xlabel(r"$\log_{10}(x)$")
        ax.set_ylabel(r"$T(x)$")
        ax.grid(True, ls=":", alpha=0.5)

    fig.subplots_adjust(wspace=0.32, hspace=0.46)
    _save(fig, outdir, "paper_model_pca_scatter_all")


def _series(result: dict, method: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = result[method]
    layers: list[int] = []
    means: list[float] = []
    stds: list[float] = []
    for layer_s in sorted(data, key=lambda x: int(x)):
        item = data[layer_s]
        mean = float(item[f"{metric}_mean"])
        std = float(item[f"{metric}_std"])
        if math.isnan(mean):
            continue
        layers.append(int(layer_s))
        means.append(mean)
        stds.append(std)
    return np.array(layers), np.array(means), np.array(stds)


def plot_layer_metric(
    models: list[tuple[ModelSpec, dict]],
    outdir: Path,
    metric: str,
    ylabel: str,
    name: str,
    *,
    logy: bool = False,
) -> None:
    nrows, ncols = _grid(len(models))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 5.8), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (spec, result) in zip(axes.flat, models):
        ax.set_visible(True)
        for method, color, label, ls in [
            ("pca", C["empirical"], "PCA", "-"),
            ("pls", C["accent2"], "PLS", "--"),
        ]:
            layers, means, stds = _series(result, method, metric)
            if len(layers) == 0:
                continue
            ax.plot(layers, means, color=color, ls=ls, lw=1.1, label=label)
            ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.12, linewidth=0)
            best_layer = int(result[f"best_layer_{method}"])
            ax.axvline(best_layer, color=color, ls=":", lw=0.7, alpha=0.8)
        ax.set_title(spec.label, fontsize=10, pad=3)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, ls=":", alpha=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(bottom=0.10)
    fig.subplots_adjust(wspace=0.36, hspace=0.46)
    _save(fig, outdir, name)


def _read_counts(path: Path, n_min: int = 0, n_max: int = 10000) -> dict[int, int]:
    counts: dict[int, int] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or {"number", "count"} - set(reader.fieldnames):
            raise SystemExit(f"{path} must contain number,count columns")
        for row in reader:
            n = int(row["number"])
            if n_min <= n <= n_max:
                counts[n] = int(row["count"])
    if not counts:
        raise SystemExit(f"no counts in {path} for [{n_min},{n_max}]")
    return {n: counts.get(n, 0) for n in range(n_min, n_max + 1)}


def _load_datasets(require_all: bool = False) -> list[tuple[DatasetSpec, dict[int, int]]]:
    out = []
    missing = []
    for spec in DATASET_SPECS:
        if spec.path.exists():
            out.append((spec, _read_counts(spec.path)))
        else:
            missing.append(spec.path)
    if missing:
        msg = "\n".join(str(p) for p in missing)
        if require_all:
            raise SystemExit(f"missing required 0_to_10000 dataset counts:\n{msg}")
        print(f"[paper_figures] skipping missing 0_to_10000 dataset counts:\n{msg}")
    if not out:
        raise SystemExit("no 0_to_10000 dataset counts found")
    return out


def _zipf_null(counts: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs], dtype=float)
    total = ys.sum()
    ranks = np.arange(1, len(xs) + 1, dtype=float)
    zipf = total * (1.0 / ranks) / np.sum(1.0 / ranks)
    kl = _kl(ys, zipf)
    return xs, ys, zipf, kl


def plot_dataset_zipf_panels(datasets: list[tuple[DatasetSpec, dict[int, int]]], outdir: Path) -> None:
    nrows, ncols = 1, len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 3.0), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (spec, counts) in zip(axes.flat, datasets):
        ax.set_visible(True)
        xs, ys, zipf, kl = _zipf_null(counts)
        ax.semilogy(xs, ys, color=C["empirical"], lw=0.75, label="empirical")
        ax.semilogy(xs, zipf, color=C["zipf"], lw=1.0, ls="-.", label=r"Zipf $\propto 1/(N+1)$")
        ax.set_title(rf"{spec.label}" + "\n" + rf"$D_{{KL}}={kl:.2f}$", fontsize=10, pad=3)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"count$(N)$")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(bottom=0.10)
    fig.subplots_adjust(wspace=0.32, hspace=0.48)
    _save(fig, outdir, "paper_dataset_zipf_only_panels_0_to_10000")


def plot_dataset_roundness_panels(datasets: list[tuple[DatasetSpec, dict[int, int]]], outdir: Path) -> None:
    nrows, ncols = 1, len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 3.0), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (spec, counts) in zip(axes.flat, datasets):
        ax.set_visible(True)
        xs = np.array(sorted(counts))
        ys = np.array([counts[n] for n in xs])
        is_hundred = (xs > 0) & (xs % 100 == 0)
        is_ten = (xs > 0) & (xs % 10 == 0) & (xs % 100 != 0)
        is_other = ~(is_hundred | is_ten)
        ax.semilogy(xs[is_other], ys[is_other], ".", color=C["muted"], ms=1.6, alpha=0.45, rasterized=True)
        ax.semilogy(xs[is_ten], ys[is_ten], "o", color=C["accent2"], ms=3.5, mfc="none", mew=0.7)
        ax.semilogy(xs[is_hundred], ys[is_hundred], "s", color=C["accent1"], ms=4.2, mfc=C["accent1"])
        ax.set_title(spec.label, fontsize=10, pad=3)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"count$(N)$")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    handles = [
        mpl.lines.Line2D([], [], marker=".", color=C["muted"], linestyle="None", label="other"),
        mpl.lines.Line2D([], [], marker="o", color=C["accent2"], mfc="none", linestyle="None", label="10-multiple"),
        mpl.lines.Line2D([], [], marker="s", color=C["accent1"], linestyle="None", label="100-multiple"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.32, hspace=0.46, bottom=0.18)
    _save(fig, outdir, "paper_dataset_round_numbers_0_to_10000")


def write_dataset_summary(datasets: list[tuple[DatasetSpec, dict[int, int]]], outdir: Path) -> None:
    path = outdir / "paper_dataset_summary_0_to_10000.csv"
    rows = []
    with open(path, "w", newline="") as fh:
        fields = [
            "dataset",
            "total",
            "zipf_kl",
            "single_digit_share",
            "ten_multiple_share",
            "hundred_multiple_share",
            "year_1900_2025_share",
            "n50",
            "n90",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for spec, counts in datasets:
            xs, ys, zipf, kl = _zipf_null(counts)
            total = ys.sum()
            cum = np.cumsum(ys) / total
            row = {
                "dataset": spec.label,
                "total": int(total),
                "zipf_kl": kl,
                "single_digit_share": float(ys[(xs >= 0) & (xs <= 9)].sum() / total),
                "ten_multiple_share": float(ys[(xs > 0) & (xs % 10 == 0) & (xs % 100 != 0)].sum() / total),
                "hundred_multiple_share": float(ys[(xs > 0) & (xs % 100 == 0)].sum() / total),
                "year_1900_2025_share": float(ys[(xs >= 1900) & (xs <= 2025)].sum() / total),
                "n50": int(xs[min(int(np.searchsorted(cum, 0.50)), len(xs) - 1)]),
                "n90": int(xs[min(int(np.searchsorted(cum, 0.90)), len(xs) - 1)]),
            }
            rows.append(row)
            writer.writerow(row)
    print(f"  wrote {path}")

    tex_path = outdir / "paper_dataset_summary_0_to_10000.tex"
    with open(tex_path, "w") as fh:
        fh.write("\\begin{table}[t]\n")
        fh.write("    \\centering\n")
        fh.write("    \\resizebox{\\linewidth}{!}{%\n")
        fh.write("    \\begin{tabular}{lrrrrrrrr}\n")
        fh.write("        \\toprule\n")
        fh.write(
            "        Dataset & Count (B) & $D_{KL}$ & $0$--$9$ & "
            "10-mult. & 100-mult. & 1900--2025 & $N_{50}$ & $N_{90}$ \\\\\n"
        )
        fh.write("        \\midrule\n")
        for row in rows:
            fh.write(
                "        "
                f"{row['dataset']} & "
                f"{int(row['total']) / 1e9:.2f} & "
                f"{float(row['zipf_kl']):.2f} & "
                f"{100 * float(row['single_digit_share']):.2f}\\% & "
                f"{100 * float(row['ten_multiple_share']):.2f}\\% & "
                f"{100 * float(row['hundred_multiple_share']):.2f}\\% & "
                f"{100 * float(row['year_1900_2025_share']):.2f}\\% & "
                f"{int(row['n50'])} & "
                f"{int(row['n90'])} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}%\n")
        fh.write("    }\n")
        fh.write(
            "    \\caption{Integer-frequency statistics for dataset counts "
            "restricted to $0\\leq N\\leq 10000$. Count (B) is the total "
            "integer-occurrence mass inside the range.}\n"
        )
        fh.write("    \\label{tab:dataset-frequency-0-10000}\n")
        fh.write("\\end{table}\n")
    print(f"  wrote {tex_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/paper_figures")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    parser.add_argument("--require-all-datasets", action="store_true")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _apply_neurips_style()

    if not args.datasets_only:
        models = _load_models()
        plot_model_pca_scatter(models, outdir)
        plot_layer_metric(models, outdir, "rho", r"$|\rho_S|$", "paper_model_layer_rho_pca_pls")
        plot_layer_metric(models, outdir, "beta", r"$\beta$", "paper_model_layer_beta_pca_pls", logy=True)
        plot_layer_metric(models, outdir, "ev", r"$\sigma^2$", "paper_model_layer_sigma2_pca_pls")

    if not args.models_only:
        datasets = _load_datasets(require_all=args.require_all_datasets)
        plot_dataset_zipf_panels(datasets, outdir)
        plot_dataset_roundness_panels(datasets, outdir)
        write_dataset_summary(datasets, outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
